import { lazy, Suspense, useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react"
import { Loader2Icon, MicIcon, PlusIcon, SendIcon, SquareIcon, XIcon } from "lucide-react"

import { Suggestion, Suggestions } from "@/components/ai-elements/suggestion"
import { LevelBars } from "@/components/kit/level-bars"
import { SourceList } from "@/components/kit/source-list"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Conversation, ConversationContent, ConversationEmptyState, ConversationScrollButton,
} from "@/components/ui/conversation"
import { Message, MessageContent } from "@/components/ui/message"
import { Textarea } from "@/components/ui/textarea"
import { toolLabel } from "@/lib/api"
import { usePrefersReducedMotion } from "@/lib/theme"
import { cn } from "@/lib/utils"
import { MAX_TYPED, type VoiceStatus } from "@/voice/engine"
import { useVoice } from "@/voice/useVoice"

const VoiceOrb = lazy(() => import("@/components/kit/voice-orb"))

const LABEL: Record<VoiceStatus, string> = {
  idle: "Ready", listening: "Listening", thinking: "Thinking", speaking: "Speaking",
}
const HINT: Record<VoiceStatus, string> = {
  idle: "Type below, or press the microphone to talk.",
  listening: "Just speak. Press send to send now; Escape closes voice mode.",
  thinking: "Working on it. Press Stop to cancel.",
  speaking: "Press Stop speaking, or just start talking.",
}

function StaticOrb({ status }: { status: VoiceStatus }) {
  return (
    <div
      className={cn(
        "size-full rounded-full bg-radial from-primary/35 via-primary/15 to-transparent ring-1 ring-primary/30",
        status !== "idle" && "from-primary/60 ring-2 ring-primary",
      )}
    />
  )
}

type Props = { suggestions?: string[]; className?: string; tall?: boolean }

export function VoiceAssistant({ suggestions = [], className, tall }: Props) {
  const [s, engine] = useVoice()
  const [text, setText] = useState("")
  const reduced = usePrefersReducedMotion()
  const dark = document.documentElement.classList.contains("dark")
  const left = MAX_TYPED - text.length
  const last = s.messages[s.messages.length - 1]

  // Send now: busy until the final transcript clears the interim text, or two seconds pass.
  const [sending, setSending] = useState(false)
  const sendTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  useEffect(() => {
    if (!s.interim) { setSending(false); clearTimeout(sendTimer.current) }
  }, [s.interim])
  useEffect(() => () => clearTimeout(sendTimer.current), [])
  const canSendNow = s.micOn && !!s.interim
  // Voice mode: focus the send button so Enter sends what was heard; back to the text field on close.
  const voiceSendRef = useRef<HTMLButtonElement>(null)
  useEffect(() => {
    if (s.micOn) voiceSendRef.current?.focus()
    else document.getElementById("vp-text")?.focus({ preventScroll: true })
  }, [s.micOn])
  // Escape leaves voice mode wherever focus is (the send button may be disabled while sending).
  useEffect(() => {
    if (!s.micOn) return
    const onEsc = (e: globalThis.KeyboardEvent) => { if (e.key === "Escape") { e.preventDefault(); engine.leaveVoice() } }
    document.addEventListener("keydown", onEsc)
    return () => document.removeEventListener("keydown", onEsc)
  }, [s.micOn, engine])
  const sendNow = () => {
    if (sending || !engine.finalize()) return
    setSending(true)
    clearTimeout(sendTimer.current)
    sendTimer.current = setTimeout(() => setSending(false), 2000)
  }

  const submit = (e?: FormEvent) => {
    e?.preventDefault()
    if (!text.trim() && canSendNow) { sendNow(); return }
    if (engine.sendText(text)) setText("")
  }
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) submit(e)
  }

  return (
    <section
      aria-label="Voice assistant"
      className={cn("flex flex-col overflow-hidden rounded-2xl border bg-card text-card-foreground shadow-sm", tall ? "h-[calc(100dvh-12rem)] min-h-[26rem]" : "h-[min(70vh,640px)] min-h-[26rem]", className)}
    >
      {/* Compact status line; the conversation gets the room */}
      <div className="flex items-center gap-3 border-b px-4 py-2.5 sm:px-6">
        <div className="size-9 shrink-0" aria-hidden>
          {reduced ? (
            <StaticOrb status={s.status} />
          ) : (
            <Suspense fallback={<StaticOrb status={s.status} />}>
              <VoiceOrb status={s.status} dark={dark}
                inputLevel={() => engine.inputLevel()} outputLevel={() => engine.outputLevel()} />
            </Suspense>
          )}
        </div>
        <div role="status" aria-live="polite" className="flex min-w-0 flex-wrap items-baseline gap-x-2">
          <span className="font-semibold">{LABEL[s.status]}</span>
          {s.status === "listening" && s.hearing && <Badge className="bg-primary/15 text-primary">Hearing you</Badge>}
          <span className="truncate text-sm text-muted-foreground">{HINT[s.status]}</span>
        </div>
        {s.messages.length > 0 && (
          <Button type="button" variant="outline" size="sm" className="ml-auto shrink-0" onClick={() => engine.newConversation()}>
            <PlusIcon aria-hidden />New conversation
          </Button>
        )}
      </div>

      {s.error && (
        <Alert variant="destructive" role="alert" className="rounded-none border-x-0 border-t-0">
          <AlertDescription>{s.error}</AlertDescription>
        </Alert>
      )}

      {/* Conversation log */}
      <Conversation className="min-h-0 flex-1" aria-label="Conversation">
        <ConversationContent className="flex flex-col gap-1 px-4 sm:px-6">
          {s.messages.length === 0 ? (
            <ConversationEmptyState
              title="No messages yet"
              description="Your conversation appears here, with the sources for each answer."
            />
          ) : (
            s.messages.map((m) => {
              const pending = m.confirm && m === last
              return (
                <Message key={m.id} from={m.role} className="py-2">
                  <MessageContent
                    variant="flat"
                    className={cn(
                      "text-[0.95rem] leading-relaxed whitespace-pre-wrap group-[.is-user]:bg-primary/10 group-[.is-user]:rounded-2xl",
                      m.role === "assistant" && "max-w-[92%]",
                      m.confirm && "rounded-xl border border-confirm-border bg-confirm px-4 py-3 text-confirm-foreground",
                    )}
                  >
                    <span className="sr-only">{m.role === "user" ? "You said:" : "Assistant:"}</span>
                    {m.confirm && <span className="text-xs font-semibold">Waiting for your confirmation</span>}
                    <span>{m.text}</span>
                    {m.streaming && !m.text && <Loader2Icon aria-hidden className="size-4 animate-spin" />}
                    {m.role === "assistant" && <SourceList sources={m.sources} />}
                    {pending && (
                      <div className="mt-1 flex flex-wrap gap-2">
                        <Button type="button" size="sm" onClick={() => engine.sendText("Yes, go ahead.")}>Yes, go ahead</Button>
                        <Button type="button" size="sm" variant="outline" onClick={() => engine.sendText("No, cancel it.")}>No, cancel</Button>
                      </div>
                    )}
                  </MessageContent>
                </Message>
              )
            })
          )}
          {s.tool && (
            <p className="flex items-center gap-2 py-1 text-xs text-muted-foreground" aria-live="polite">
              <Loader2Icon aria-hidden className="size-3.5 animate-spin" />
              Checking {toolLabel(s.tool)}…
            </p>
          )}
          {s.status === "thinking" && !s.tool && last?.role === "user" && (
            <p className="flex items-center gap-2 py-1 text-xs text-muted-foreground">
              <Loader2Icon aria-hidden className="size-3.5 animate-spin" /> Thinking…
            </p>
          )}
        </ConversationContent>
        <ConversationScrollButton aria-label="Scroll to latest message" />
      </Conversation>

      {/* Composer: text mode (mic, field, send) or voice mode (close, waveform, stop, send now) */}
      <div className="flex flex-col gap-2 border-t bg-muted/40 px-4 py-3 sm:px-6">
        {suggestions.length > 0 && (
          <Suggestions aria-label="Suggested questions">
            {suggestions.map((q) => (
              <Suggestion key={q} suggestion={q} onClick={(v) => engine.sendText(v)} className="bg-card" />
            ))}
          </Suggestions>
        )}
        {s.micOn ? (
          <div className="flex items-center gap-2" data-testid="voice-composer">
            <Button type="button" size="icon" variant="outline" className="size-11 shrink-0 rounded-full"
              aria-label="Close voice mode" onClick={() => engine.leaveVoice()}>
              <XIcon aria-hidden />
            </Button>
            <div className="flex min-h-11 min-w-0 flex-1 flex-col justify-center rounded-md border bg-card px-3 py-1">
              {s.interim ? (
                <p className="truncate text-sm" aria-live="off"><span className="sr-only">Heard so far: </span>{s.interim}</p>
              ) : (
                <p className="truncate text-xs text-muted-foreground">{s.status === "speaking" ? "Assistant is speaking" : "Listening…"}</p>
              )}
              <LevelBars level={() => (s.status === "speaking" ? engine.outputLevel() : engine.inputLevel())} still={reduced} />
            </div>
            {s.canStop && (
              <Button type="button" variant="secondary" className="h-11 shrink-0 rounded-full"
                onClick={() => engine.stopSpeaking()}>
                <SquareIcon aria-hidden className="fill-current" /><span className="hidden sm:inline">Stop speaking</span><span className="sm:hidden">Stop</span>
              </Button>
            )}
            <Button ref={voiceSendRef} type="button" size="icon" className="size-11 shrink-0 rounded-full"
              aria-label="Send what I said" aria-busy={sending} disabled={!canSendNow || sending} onClick={sendNow}>
              {sending ? <Loader2Icon aria-hidden className="animate-spin" /> : <SendIcon aria-hidden />}
            </Button>
          </div>
        ) : (
          <form onSubmit={submit} className="flex items-end gap-2">
            <Button type="button" size="icon" className="size-11 shrink-0 rounded-full"
              aria-label="Talk: switch to voice" onClick={() => engine.toggleTalk()}>
              <MicIcon aria-hidden />
            </Button>
            <label htmlFor="vp-text" className="sr-only">Type a message</label>
            <Textarea
              id="vp-text"
              rows={1}
              maxLength={MAX_TYPED}
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={onKey}
              placeholder="Type your message"
              autoComplete="off"
              aria-describedby={left < 100 ? "vp-count" : undefined}
              className="max-h-32 min-h-11 resize-none bg-card"
            />
            <Button type="submit" size="icon" className="size-11 shrink-0 rounded-full" aria-label="Send" disabled={!text.trim()}>
              <SendIcon aria-hidden />
            </Button>
          </form>
        )}
        {!s.micOn && left < 100 && (
          <p id="vp-count" className={cn("text-right text-xs text-muted-foreground", left < 20 && "text-destructive")}>
            {left} characters left
          </p>
        )}
      </div>
    </section>
  )
}
