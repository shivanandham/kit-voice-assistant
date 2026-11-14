"""The assistant turn: Claude with tools, streamed, with per-conversation content-block memory."""

import asyncio
import contextlib
import json
import logging
import re
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from typing import Any

import anthropic

from app import history, moderation
from app.agent import confirm, prompts, tools
from app.auth import SessionCtx
from app.config import get_settings
from app.kit.client import college_today
from app.knowledge import rag, store
from app.knowledge.mode import knowledge_block

log = logging.getLogger(__name__)

MAX_MODEL_CALLS = 8
CACHE_CONVERSATIONS = 200
MAX_CACHED_MESSAGES = 20  # default history window (settings.history_window)
MODERATION_TIMEOUT = 2.0  # seconds to wait for the moderation classifier
TOOL_GRACE_SECONDS = 5.0
INTERRUPTED_MARK = " [interrupted]"
INTERRUPTED_NOTE = (
    "(Note from the system: the person interrupted your previous reply by starting to speak, "
    "so they heard it only up to the point where it stops.)"
)
REFUSAL_TEXT = "Sorry, I can't help with that one. Is there something else I can do?"
LIMIT_TEXT = "Sorry, that is taking me too long. Could you ask me again in a simpler way?"
ERROR_TEXT = "Sorry, I'm having trouble thinking right now. Please try again in a moment."
NOT_FOUND_TEXT = "Sorry, I can't find that conversation. Please start a new one."
EMPTY_TEXT = "Sorry, I didn't get that. Could you say it again?"
TOOL_LABELS = {
    "get_today": "College system: calendar",
    "get_college_info": "College system: college information",
    "list_courses": "College system: courses",
    "get_my_profile": "College system: profile",
    "get_my_academics": "College system: academic record",
    "get_my_subjects": "College system: subjects",
    "get_attendance_summary": "College system: attendance",
    "get_attendance_sessions": "College system: attendance",
    "get_timetable": "College system: timetable",
    "get_calendar": "College system: calendar",
    "list_my_requests": "College system: leave and on-duty requests",
    "get_request": "College system: leave and on-duty requests",
    "list_placement_drives": "College system: placement drives",
    "list_my_placements": "College system: placements",
    "list_my_classes": "College system: classes",
    "get_class_attendance": "College system: class attendance",
    "list_class_requests": "College system: student requests",
}
_DOC_KEYS = ("ref", "document_title", "version", "section", "heading", "page_start", "page_end")
_BOUNDARY_RE = re.compile(r"[,;:.!?]+[\"')\]]*[ \t]+|\n+")  # pieces end at clause boundaries
_OPEN_MARKUP_RE = re.compile(r"<[^>\n]*$|\[[^\]\n]*$|\]\([^)\n]*$")  # unclosed tag or link
_REF_RE = re.compile(r"D\d+:[A-Za-z0-9][A-Za-z0-9./\-]*")

_client: anthropic.AsyncAnthropic | None = None
_memory: "OrderedDict[str, list[dict]]" = OrderedDict()
_owners: dict[str, str] = {}  # conversation id -> owner_key of the cached entry
_locks: dict[str, asyncio.Lock] = {}
_in_flight: dict[str, int] = {}  # conversation id -> turns running or waiting


def _get_client() -> anthropic.AsyncAnthropic:
    """The shared async Anthropic client, created on first use."""
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(
            api_key=get_settings().anthropic_api_key or None,
            timeout=anthropic.Timeout(60.0, connect=5.0),
            max_retries=2,
        )
    return _client


def _field(block: Any, name: str) -> Any:
    """Read a field from a content block that is either an SDK model or a plain dict."""
    return block.get(name) if isinstance(block, dict) else getattr(block, name, None)


def _blocks(content: Any) -> list:
    """Message content as a list of blocks."""
    return [{"type": "text", "text": content}] if isinstance(content, str) else list(content)


def _append(messages: list[dict], role: str, content: list) -> None:
    """Append a message, merging into the previous one when it has the same role."""
    if not content:
        return
    if messages and messages[-1]["role"] == role:
        messages[-1] = {"role": role, "content": _blocks(messages[-1]["content"]) + list(content)}
    else:
        messages.append({"role": role, "content": list(content)})


def _starts_turn(message: dict) -> bool:
    """True for a user message that is not only carrying tool results."""
    if message["role"] != "user":
        return False
    return all(_field(b, "type") != "tool_result" for b in _blocks(message["content"]))


def _window() -> int:
    """How many messages at most go to the model (settings.history_window, default 20)."""
    return max(2, int(get_settings().history_window))


def _trim(messages: list[dict], limit: int | None = None) -> list[dict]:
    """Keep at most `limit` (the history window) of the latest messages, starting at a real
    user turn so every tool_use keeps its tool_result. If the current turn alone is longer than
    the window, it is kept whole."""
    limit = limit or _window()
    if len(messages) <= limit:
        return messages
    for i in range(len(messages) - limit, len(messages)):
        if _starts_turn(messages[i]):
            return messages[i:]
    starts = [i for i, m in enumerate(messages) if _starts_turn(m)]
    return messages[starts[-1]:] if starts else messages[-1:]


def _strip_thinking(messages: list[dict]) -> list[dict]:
    """Drop thinking blocks from finished turns.

    A thinking block's signature is bound to everything before it, including the system prompt.
    The per-user system block changes between turns (pending actions, the college date) and the
    history window drops old turns, so replaying old thinking would be rejected with a 400.
    Blocks are still passed back unchanged within a turn's tool loop.
    """
    out: list[dict] = []
    for message in messages:
        if message["role"] == "assistant" and not isinstance(message["content"], str):
            content = [b for b in message["content"]
                       if _field(b, "type") not in ("thinking", "redacted_thinking")]
            _append(out, "assistant", content)
        else:
            _append(out, message["role"], _blocks(message["content"]))
    return out


def _remember(conversation_id: str, messages: list[dict], owner_key: str | None = None) -> None:
    """Store the conversation's message list (without thinking) in the bounded in-memory cache.

    Eviction is oldest first and never touches a conversation with a turn in flight, so its lock
    keeps serialising turns.
    """
    _memory[conversation_id] = _trim(_strip_thinking(messages))
    _memory.move_to_end(conversation_id)
    if owner_key is not None:
        _owners[conversation_id] = owner_key
    for old in list(_memory):
        if len(_memory) <= CACHE_CONVERSATIONS:
            break
        if _in_flight.get(old):
            continue
        del _memory[old]
        _owners.pop(old, None)
        _locks.pop(old, None)


async def _load(conversation_id: str, owner_key: str) -> list[dict]:
    """The conversation so far: cached content blocks, or plain text rebuilt from history."""
    cached = _memory.get(conversation_id)
    if cached is not None and _owners.get(conversation_id, owner_key) == owner_key:
        return list(cached)
    messages: list[dict] = []
    for row in await history.recent_messages(conversation_id, limit=_window()):
        text = row["content"]
        interrupted = row["role"] == "assistant" and text.endswith(INTERRUPTED_MARK)
        if interrupted:
            text = text[: -len(INTERRUPTED_MARK)]
        if text.strip():
            _append(messages, row["role"], [{"type": "text", "text": text}])
        if interrupted:
            _append(messages, "user", [{"type": "text", "text": INTERRUPTED_NOTE}])
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    return messages


async def _safe(coro: Any, what: str, default: Any) -> Any:
    """Await a context lookup; log and fall back to a default if it fails."""
    try:
        return await coro
    except Exception:
        log.exception("could not load %s", what)
        return default


async def _request_parts(ctx: SessionCtx, conversation_id: str) -> tuple[list[dict], list[dict]]:
    """Build (tools, system blocks) for this turn."""
    block, today, pending = await asyncio.gather(
        _safe(knowledge_block(ctx), "knowledge block", None),
        _safe(college_today(), "college date", None),
        _safe(confirm.pending_summary(ctx, conversation_id), "pending actions", None),
    )
    knowledge_text = block.text if block else "(The college documents are unavailable right now.)"
    include_search = bool(block and block.mode == "rag")
    specs = tools.tool_specs(ctx.role, include_search=include_search)
    return specs, prompts.system_blocks(ctx, knowledge_text, today, pending)


async def _warm_variants() -> list[tuple[str, list[dict], dict]]:
    """The distinct shared prefixes (label, tools, stable system block) real turns can send.

    The stable block depends on the role and, for students, on the department (lesson notes are
    per department), so one student variant is built per department that has notes plus one for
    departments without any. Variants that render identically are sent once.
    """
    docs = await _safe(store.active_documents(), "documents", [])
    departments = sorted({dep for d in docs if d.get("type") == "lesson_notes"
                          for dep in ((d.get("applies_to") or {}).get("departments") or [])})
    contexts = [("visitor", SessionCtx("warmup", "warmup", "visitor", None, None)),
                ("teacher", SessionCtx("warmup", "warmup", "teacher", None, None)),
                ("student", SessionCtx("warmup", "warmup", "student", {"department": None}, None))]
    contexts += [(f"student {dep}", SessionCtx("warmup", "warmup", "student",
                                                {"department": dep}, None)) for dep in departments]
    variants, seen = [], set()
    for label, ctx in contexts:
        block = await knowledge_block(ctx)
        specs = tools.tool_specs(ctx.role, include_search=block.mode == "rag")
        stable = prompts.system_blocks(ctx, block.text, None, None)[0]
        key = json.dumps([specs, stable], sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            variants.append((label, specs, stable))
    return variants


async def warm_cache() -> None:
    """Write (or refresh) the prompt cache for every shared prefix; never raises.

    Sends a max_tokens 0 request per variant with the same model, effort, tools and stable system
    block as real turns, so the first real turn after startup reads the prefix from cache instead
    of writing it. Call at startup and every four minutes or so (the cache lives five minutes).
    Disabled when settings.llm_keep_warm is false.
    """
    try:
        settings = get_settings()
        if not settings.llm_keep_warm:
            return
        variants = await _warm_variants()

        async def one(label: str, specs: list[dict], stable: dict) -> None:
            started = time.monotonic()
            message = await _get_client().messages.create(
                model=settings.llm_model,
                max_tokens=0,
                system=[stable],
                tools=specs,
                messages=[{"role": "user", "content": "warmup"}],
                output_config={"effort": settings.llm_effort},
                **_thinking(settings),
            )
            usage = message.usage
            log.info("cache warm %s read=%s write=%s in %.2fs", label,
                     getattr(usage, "cache_read_input_tokens", None),
                     getattr(usage, "cache_creation_input_tokens", None),
                     time.monotonic() - started)

        results = await asyncio.gather(*(one(*v) for v in variants), return_exceptions=True)
        for (label, _, _), result in zip(variants, results, strict=True):
            if isinstance(result, BaseException):
                log.warning("cache warm %s failed: %s", label, type(result).__name__)
    except Exception:
        log.exception("cache warm failed")


def _thinking(settings: Any) -> dict:
    """Extra request arguments for the thinking mode: none for the model's default (adaptive),
    or thinking disabled when settings.llm_thinking is "disabled" (shorter time to first text)."""
    if settings.llm_thinking == "disabled":
        return {"thinking": {"type": "disabled"}}
    return {}


def _log_usage(
    message: Any, started: float, first_text: float | None, first_token: float | None = None
) -> None:
    """Log token usage and timing for one model call."""
    usage = message.usage
    log.info(
        "llm call stop=%s in=%s out=%s cache_read=%s cache_write=%s first_token=%s ttft=%s"
        " total=%.2fs",
        message.stop_reason,
        getattr(usage, "input_tokens", None),
        getattr(usage, "output_tokens", None),
        getattr(usage, "cache_read_input_tokens", None),
        getattr(usage, "cache_creation_input_tokens", None),
        f"{first_token - started:.2f}s" if first_token else "-",
        f"{first_text - started:.2f}s" if first_text else "-",
        time.monotonic() - started,
    )


async def _run_one(
    block: Any, ctx: SessionCtx, conversation_id: str, offered: set[str]
) -> tools.ToolResult:
    """Run one tool call, turning unexpected exceptions into an error result.

    A tool name that was not offered in this request is refused without running anything.
    """
    if _field(block, "name") not in offered:
        log.warning("model asked for a tool it was not offered: %s", _field(block, "name"))
        refusal = {"error": "That tool is not available."}
        try:
            await history.log_tool_call(conversation_id, str(_field(block, "name")),
                                        _field(block, "input") or {}, refusal, True)
        except Exception:
            log.exception("could not log a refused tool call")
        return tools.ToolResult(json.dumps(refusal), True)
    started = time.monotonic()
    try:
        result = await tools.run_tool(_field(block, "name"), _field(block, "input") or {}, ctx,
                                      conversation_id)
        log.info("tool %s took %.2fs", _field(block, "name"), time.monotonic() - started)
        return result
    except Exception:
        log.exception("tool %s failed", _field(block, "name"))
        return tools.ToolResult(json.dumps({"error": "The tool failed unexpectedly."}), True)


def _tool_result(block: Any, result: tools.ToolResult) -> dict:
    """A tool_result block answering a tool_use block."""
    out = {"type": "tool_result", "tool_use_id": _field(block, "id"), "content": result.content}
    if result.is_error:
        out["is_error"] = True
    return out


def _error_results(uses: list, reason: str) -> list[dict]:
    """Error tool_result blocks for tool calls that were not run."""
    payload = json.dumps({"error": reason})
    return [_tool_result(b, tools.ToolResult(payload, True)) for b in uses]


async def run_turn(ctx: SessionCtx, conversation_id: str, user_text: str) -> AsyncIterator[dict]:
    """Run one user turn and stream events: text_delta, tool start/end, then done or error.

    The conversation must belong to ctx's owner, otherwise only an error event is sent. Turns on
    one conversation run one at a time; a barge-in's new turn waits for the cancelled one to
    finish saving. The user message is saved before the model is called; if that fails, only an
    error event is sent. The full reply is saved as an assistant message before `done`. If the consuming task is cancelled (barge-in), the model stream is closed, unanswered
    tool calls get error results so the message list stays valid, and the text produced so far is
    saved marked as interrupted.
    """
    try:
        owned = await history.conversation_owned(ctx, conversation_id)
    except Exception:
        log.exception("ownership check failed")
        owned = None
    if not owned:
        yield {"type": "error", "message": NOT_FOUND_TEXT if owned is False else ERROR_TEXT}
        return
    _in_flight[conversation_id] = _in_flight.get(conversation_id, 0) + 1
    try:
        lock = _locks.setdefault(conversation_id, asyncio.Lock())
        async with lock, contextlib.aclosing(_turn(ctx, conversation_id, user_text)) as events:
            async for event in events:
                yield event
    finally:
        _in_flight[conversation_id] -= 1
        if not _in_flight[conversation_id]:
            del _in_flight[conversation_id]
            if conversation_id not in _memory:
                _locks.pop(conversation_id, None)


class _Speech:
    """Turns streamed model text into clean, whole-word pieces and holds back the SOURCES line.

    Text is cut at clause and sentence boundaries (never inside an unclosed HTML tag or markdown
    link, so those can be stripped whole), each piece passes moderation.clean_output, and
    a final "SOURCES: ..." line (any case or spacing, even split across deltas) is captured instead
    of being emitted. Once a piece is blocked, everything after it is dropped.
    """

    def __init__(self, secrets: tuple[str, ...] = ()) -> None:
        self.secrets = secrets
        self.buffer = ""
        self.at_line_start = True
        self.sources_raw = ""
        self.capturing = False
        self.blocked: str | None = None

    def feed(self, text: str) -> list[str]:
        """Add model text; return the cleaned pieces that are ready to emit."""
        if self.blocked:
            return []
        if self.capturing:
            self.sources_raw += text
            return []
        self.buffer += text
        return self._take(final=False)

    def flush(self) -> list[str]:
        """End of a model call: return what is left, and start the next call on a new line."""
        pieces = [] if self.blocked or self.capturing else self._take(final=True)
        self.buffer, self.capturing, self.at_line_start = "", False, True
        return pieces

    def _take(self, final: bool) -> list[str]:
        out: list[str] = []
        while self.buffer and not self.blocked:
            head = re.sub(r"\s+", "", self.buffer[:24]).lower()
            if head.startswith("sources:"):
                self.capturing, self.sources_raw = True, self.sources_raw + self.buffer
                self.buffer = ""
                break
            if not final and head and "sources:".startswith(head):
                break  # might be the start of the SOURCES line; wait for more text
            m = None
            for candidate in _BOUNDARY_RE.finditer(self.buffer):
                if not _OPEN_MARKUP_RE.search(self.buffer[: candidate.start()]):
                    m = candidate
                    break
            if m is None:
                if not final:
                    break
                piece, self.buffer = self.buffer, ""
            else:
                piece, self.buffer = self.buffer[: m.end()], self.buffer[m.end():]
            cleaned = moderation.clean_output(piece, at_line_start=self.at_line_start,
                                              secrets=self.secrets)
            self.at_line_start = piece.endswith("\n")
            if cleaned.blocked:
                self.blocked = cleaned.blocked
                out.append(cleaned.text)
                self.buffer = ""
                break
            if cleaned.text.strip():
                out.append(cleaned.text)
        return out

    def refs(self) -> list[str]:
        """Document refs named on the SOURCES line, in order, without duplicates."""
        found = _REF_RE.findall(self.sources_raw)
        return list(dict.fromkeys(r.rstrip("./-") for r in found))


async def _verdict_gate(task: "asyncio.Task | None", rules: str, deadline: float) -> tuple[str, str]:
    """Wait (until the deadline) for the classifier; return (category, source)."""
    if task is None:
        return rules, "rules"
    try:
        verdict = await asyncio.wait_for(asyncio.shield(task),
                                         max(0.0, deadline - time.monotonic()))
    except TimeoutError:
        log.warning("moderation classifier timed out; using the built-in rules")
        return rules, "rules"
    except Exception:
        log.warning("moderation classifier failed; using the built-in rules", exc_info=True)
        return rules, "rules"
    if verdict is None:
        log.warning("moderation classifier gave no usable verdict; using the built-in rules")
        return rules, "rules"
    return verdict, "model"


async def _log_moderation(conversation_id: str, direction: str, category: str, source: str,
                          action: str) -> None:
    """Record a non-ok moderation decision (category only, never the text)."""
    try:
        await history.log_tool_call(
            conversation_id, "moderation",
            {"direction": direction, "category": category, "source": source},
            {"action": action}, False)
    except Exception:
        log.exception("could not log a moderation decision")


async def _citations(ctx: SessionCtx, refs: list[str], used_tools: list[str]) -> list[dict]:
    """Document sources resolved for this user, then one college-record source per data tool."""
    sources: list[dict] = []
    resolve = getattr(rag, "resolve_refs", None)
    if refs and resolve is not None:
        try:
            for doc in await resolve(refs, ctx):
                sources.append({"kind": "document", **{k: doc.get(k) for k in _DOC_KEYS}})
        except Exception:
            log.exception("could not resolve document refs")
    labels = dict.fromkeys(TOOL_LABELS[name] for name in used_tools if name in TOOL_LABELS)
    sources += [{"kind": "college_record", "label": label} for label in labels]
    return sources


async def _blocked_reply(ctx: SessionCtx, category: str) -> str:
    """The refusal or support line for a blocked turn (support uses the user's documents)."""
    knowledge_text = ""
    if category == "distress":
        block = await _safe(knowledge_block(ctx), "knowledge block", None)
        knowledge_text = block.text if block else ""
    return moderation.refusal_for(category, knowledge_text)


async def _turn(ctx: SessionCtx, conversation_id: str, user_text: str) -> AsyncIterator[dict]:
    """The body of run_turn, run while holding the conversation's lock."""
    settings = get_settings()
    owner = ctx.owner_key
    try:
        messages = await _load(conversation_id, owner)
        await history.add_message(conversation_id, "user", user_text)
    except Exception:
        log.exception("could not load the conversation or save the user message")
        yield {"type": "error", "message": ERROR_TEXT}
        return
    _append(messages, "user", [{"type": "text", "text": user_text}])
    spoken: list[str] = []  # all text emitted this turn
    call_text: list[str] = []  # text of the model call in progress (not yet in `messages`)
    open_uses: list = []  # tool_use blocks appended to messages but not yet answered
    ran: dict[str, tools.ToolResult] = {}  # results of open_uses that have finished
    used_tools: list[str] = []  # successful tool calls this turn, for citations
    speech = _Speech(secrets=tuple(s for s in (ctx.kit_token,) if s))
    rules = moderation.check_input(user_text)
    classifier: asyncio.Task | None = None
    verdict: tuple[str, str] | None = None
    deadline = time.monotonic() + MODERATION_TIMEOUT

    def emit(text: str) -> dict:
        if spoken and not spoken[-1][-1:].isspace() and not text[:1].isspace():
            text = " " + text
        spoken.append(text)
        call_text.append(text)
        return {"type": "text_delta", "text": text}

    async def gate() -> tuple[str, str]:
        nonlocal verdict
        if verdict is None:
            waited = time.monotonic()
            verdict = await _verdict_gate(classifier, rules, deadline)
            log.info("moderation %s (%s), held output for %.2fs", verdict[0], verdict[1],
                     time.monotonic() - waited)
        return verdict

    async def finish(reply: str, citations: list[dict]) -> AsyncIterator[dict]:
        _remember(conversation_id, messages, owner)
        try:
            await history.add_message(conversation_id, "assistant", reply, citations or None)
        except Exception:
            log.exception("could not save the assistant reply")
        if citations:
            yield {"type": "citations", "sources": citations}
            yield {"type": "done", "text": reply, "citations": citations}
        else:
            yield {"type": "done", "text": reply}

    if rules != "ok":
        verdict = (rules, "rules")
    elif settings.moderation_llm:
        classifier = asyncio.ensure_future(
            moderation.classify(_get_client(), settings.llm_model, user_text))

    try:
        if verdict is not None and verdict[0] != "ok":
            category = verdict[0]
        else:
            category = "ok"
            prep_started = time.monotonic()
            specs, system = await _request_parts(ctx, conversation_id)
            offered = {spec["name"] for spec in specs}
            log.info("turn prep %.2fs (documents, college date, pending actions)",
                     time.monotonic() - prep_started)
        for call in range(1, MAX_MODEL_CALLS + 1):
            if category != "ok":
                break
            started, first_text, first_token = time.monotonic(), None, None
            call_text.clear()
            message = None
            async with _get_client().messages.stream(
                model=settings.llm_model,
                max_tokens=settings.llm_max_tokens,
                system=system,
                tools=specs,
                messages=_trim(messages),
                output_config={"effort": settings.llm_effort},
                **_thinking(settings),
                cache_control={"type": "ephemeral"},  # second breakpoint: the conversation tail
            ) as stream:
                async for event in stream:
                    if event.type != "text" or not event.text:
                        continue
                    first_token = first_token or time.monotonic()
                    pieces = speech.feed(event.text)
                    if pieces and (await gate())[0] != "ok":
                        category = verdict[0]
                        break
                    for piece in pieces:
                        first_text = first_text or time.monotonic()
                        yield emit(piece)
                    if speech.blocked:
                        break
                if category == "ok" and not speech.blocked:
                    message = await stream.get_final_message()
                    pieces = speech.flush()
                    if pieces and (await gate())[0] != "ok":
                        category = verdict[0]
                    else:
                        for piece in pieces:
                            first_text = first_text or time.monotonic()
                            yield emit(piece)
            if message is not None:
                _log_usage(message, started, first_text, first_token)
            if category != "ok":
                break  # the model verdict blocked the turn; nothing was said or run
            if speech.blocked:
                await _log_moderation(conversation_id, "output", speech.blocked, "rules",
                                      "replaced")
                _close_open(messages, call_text, [], {})
                break

            if message.stop_reason == "refusal":
                # A refusal can cut a block off mid-way, so keep only the text that was spoken.
                _close_open(messages, call_text, [], {})
                yield emit(REFUSAL_TEXT)
                call_text.clear()
                _append(messages, "assistant", [{"type": "text", "text": REFUSAL_TEXT}])
                break

            uses = [b for b in message.content if _field(b, "type") == "tool_use"]
            if uses and (await gate())[0] != "ok":
                category = verdict[0]
                break
            _append(messages, "assistant", list(message.content))
            call_text.clear()
            if not uses:
                if message.stop_reason not in ("end_turn", "stop_sequence", "max_tokens"):
                    log.warning("unexpected stop_reason %s", message.stop_reason)
                break
            open_uses = list(uses)

            if message.stop_reason == "max_tokens":
                # A truncated tool input must not run; tell the model and let it retry.
                _append(messages, "user", _error_results(
                    uses, "Your reply hit the length limit before this tool call was complete. "
                          "Call the tool again and keep the reply shorter."))
                open_uses = []
                continue
            if call == MAX_MODEL_CALLS:
                _append(messages, "user", _error_results(uses, "Step limit reached; not run."))
                open_uses = []
                yield emit(LIMIT_TEXT)
                _append(messages, "assistant", [{"type": "text", "text": LIMIT_TEXT}])
                call_text.clear()
                break

            async with contextlib.aclosing(
                _run_tools(uses, ctx, conversation_id, ran, offered)
            ) as events:
                async for event in events:
                    yield event
            _append(messages, "user", [_tool_result(b, ran[_field(b, "id")]) for b in uses])
            used_tools += [_field(b, "name") for b in uses if not ran[_field(b, "id")].is_error]
            open_uses, ran = [], {}

        if category != "ok":
            source = verdict[1] if verdict else "rules"
            action = "supported" if category == "distress" else "refused"
            await _log_moderation(conversation_id, "input", category, source, action)
            call_text.clear()
            reply = await _blocked_reply(ctx, category)
            yield emit(reply)
            _append(messages, "assistant", [{"type": "text", "text": reply}])
            call_text.clear()
            async for event in finish(reply, []):
                yield event
            return
        citations = await _citations(ctx, speech.refs(), used_tools)
    except asyncio.CancelledError:
        await asyncio.shield(_save_interrupted(conversation_id, owner, messages, spoken, call_text,
                                               open_uses, ran))
        raise
    except GeneratorExit:
        await _save_interrupted(conversation_id, owner, messages, spoken, call_text, open_uses, ran)
        raise
    except anthropic.APIError as exc:
        log.error("anthropic error: %s %s %s", type(exc).__name__, getattr(exc, "status_code", ""),
                  str(getattr(exc, "message", ""))[:300])
        await _save_partial(conversation_id, owner, messages, spoken, call_text, open_uses, ran)
        yield {"type": "error", "message": ERROR_TEXT}
        return
    except Exception:
        log.exception("assistant turn failed")
        await _save_partial(conversation_id, owner, messages, spoken, call_text, open_uses, ran)
        yield {"type": "error", "message": ERROR_TEXT}
        return
    finally:
        if classifier is not None and not classifier.done():
            classifier.cancel()

    reply = "".join(spoken).strip()
    if not reply:
        # An empty reply would be silence on a voice line; say something instead.
        yield emit(EMPTY_TEXT)
        _append(messages, "assistant", [{"type": "text", "text": EMPTY_TEXT}])
        reply = EMPTY_TEXT
    async for event in finish(reply, citations):
        yield event


async def _run_tools(
    uses: list, ctx: SessionCtx, conversation_id: str, ran: dict[str, tools.ToolResult],
    offered: set[str],
) -> AsyncIterator[dict]:
    """Run tool calls concurrently, storing results in `ran` by tool_use id; yield tool events.

    If cancelled, tools still in flight get a short grace period so a write is not left in an
    unknown state; the cancellation then propagates.
    """
    async def one(block: Any) -> Any:
        ran[_field(block, "id")] = result = await _run_one(block, ctx, conversation_id, offered)
        return block, result

    for b in uses:
        yield {"type": "tool", "name": _field(b, "name"), "state": "start", "is_error": False}
    tasks = [asyncio.ensure_future(one(b)) for b in uses]
    try:
        for fut in asyncio.as_completed(tasks):
            block, result = await fut
            yield {"type": "tool", "name": _field(block, "name"), "state": "end",
                   "is_error": result.is_error}
    except (asyncio.CancelledError, GeneratorExit):
        await asyncio.shield(asyncio.wait(tasks, timeout=TOOL_GRACE_SECONDS))
        for t in tasks:
            t.cancel()
        raise


def _close_open(
    messages: list[dict], call_text: list[str], open_uses: list, ran: dict[str, tools.ToolResult]
) -> None:
    """Make the message list valid: answer open tool calls and keep text spoken mid-call."""
    if open_uses:
        not_run = tools.ToolResult(json.dumps({"error": "Not run: the person interrupted."}), True)
        _append(messages, "user",
                [_tool_result(b, ran.get(_field(b, "id"), not_run)) for b in open_uses])
        open_uses.clear()
    text = "".join(call_text)
    if text.strip():
        _append(messages, "assistant", [{"type": "text", "text": text.rstrip()}])
    call_text.clear()


async def _save_interrupted(conversation_id: str, owner: str, messages: list[dict], spoken: list[str],
                            call_text: list[str], open_uses: list,
                            ran: dict[str, tools.ToolResult]) -> None:
    """After a barge-in: keep the message list valid, note the interruption, save the partial."""
    _close_open(messages, call_text, open_uses, ran)
    _append(messages, "user", [{"type": "text", "text": INTERRUPTED_NOTE}])
    _remember(conversation_id, messages, owner)
    reply = "".join(spoken).strip()
    if reply:
        try:
            await history.add_message(conversation_id, "assistant", reply + INTERRUPTED_MARK)
        except Exception:
            log.exception("could not save interrupted reply")


async def _save_partial(conversation_id: str, owner: str, messages: list[dict], spoken: list[str],
                        call_text: list[str], open_uses: list,
                        ran: dict[str, tools.ToolResult]) -> None:
    """After an error: keep the message list valid and save whatever was spoken."""
    _close_open(messages, call_text, open_uses, ran)
    _remember(conversation_id, messages, owner)
    reply = "".join(spoken).strip()
    if reply:
        try:
            await history.add_message(conversation_id, "assistant", reply)
        except Exception:
            log.exception("could not save partial reply")
