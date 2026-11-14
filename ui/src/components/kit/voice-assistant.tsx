import { lazy, Suspense, useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react"
import { Loader2Icon, MicIcon, MicOffIcon, SendIcon, SquareIcon } from "lucide-react"

import { Suggestion, Suggestions } from "@/components/ai-elements/suggestion"
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
  idle: "Press Start talking and ask your question, or type below.",
  listening: "Just speak. You can talk over the assistant to interrupt.",
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
      className={cn("flex flex-col overflow-hidden rounded-2xl border bg-card text-card-foreground shadow-sm", className)}
    >
      {/* Primary control and live visual */}
      <div className="flex flex-col items-center gap-4 border-b px-4 pt-6 pb-5 sm:flex-row sm:items-center sm:gap-6 sm:px-6">
        <div className="size-32 shrink-0 sm:size-36" aria-hidden>
          {reduced ? (
            <StaticOrb status={s.status} />
          ) : (
            <Suspense fallback={<StaticOrb status={s.status} />}>
              <VoiceOrb status={s.status} dark={dark}
                inputLevel={() => engine.inputLevel()} outputLevel={() => engine.outputLevel()} />
            </Suspense>
          )}
        </div>
        <div className="flex w-full min-w-0 flex-col items-center gap-3 sm:items-start">
          <div role="status" aria-live="polite" className="flex flex-wrap items-center justify-center gap-2 sm:justify-start">
            <span className="text-lg font-semibold">{LABEL[s.status]}</span>
            {s.status === "listening" && s.hearing && (
              <Badge className="bg-primary/15 text-primary">Hearing you</Badge>
            )}
          </div>
          <p className="text-center text-sm text-muted-foreground sm:text-left">{HINT[s.status]}</p>
          <div className="flex flex-wrap items-center justify-center gap-2">
            <Button
              type="button"
              size="lg"
              aria-pressed={s.micOn}
              onClick={() => engine.toggleTalk()}
              variant={s.micOn ? "outline" : "default"}
              className="h-12 rounded-full px-6 text-base"
            >
              {s.micOn ? <MicOffIcon aria-hidden /> : <MicIcon aria-hidden />}
              {s.micOn ? "End conversation" : "Start talking"}
            </Button>
            {s.canStop && (
              <Button type="button" size="lg" variant="secondary" className="h-12 rounded-full px-5"
                onClick={() => engine.stopSpeaking()}>
                <SquareIcon aria-hidden className="fill-current" />
                {s.status === "speaking" ? "Stop speaking" : "Stop"}
              </Button>
            )}
          </div>
        </div>
      </div>

      {s.error && (
        <Alert variant="destructive" role="alert" className="rounded-none border-x-0 border-t-0">
          <AlertDescription>{s.error}</AlertDescription>
        </Alert>
      )}

      {/* Conversation log */}
      <Conversation className={cn("min-h-0", tall ? "h-[min(30rem,52vh)]" : "h-[min(22rem,48vh)]")} aria-label="Conversation">
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
                    variant={m.role === "user" ? "contained" : "flat"}
                    className={cn(
                      "text-[0.95rem] leading-relaxed whitespace-pre-wrap",
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

      {/* Typed input, an equal alternative to voice */}
      <div className="flex flex-col gap-2 border-t bg-muted/40 px-4 py-3 sm:px-6">
        {suggestions.length > 0 && (
          <Suggestions aria-label="Suggested questions">
            {suggestions.map((q) => (
              <Suggestion key={q} suggestion={q} onClick={(v) => engine.sendText(v)} className="bg-card" />
            ))}
          </Suggestions>
        )}
        {canSendNow && (
          <div className="flex items-center gap-2 rounded-lg border border-primary/40 bg-primary/10 px-3 py-2">
            <p className="min-w-0 flex-1 truncate text-sm" aria-live="off">
              <span className="sr-only">Heard so far: </span>“{s.interim}”
            </p>
            <Button type="button" size="sm" className="shrink-0 rounded-full" onClick={sendNow}
              disabled={sending} aria-busy={sending} aria-label="Send what I said">
              {sending ? <Loader2Icon aria-hidden className="animate-spin" /> : <SendIcon aria-hidden />}
              {sending ? "Sending…" : "Send"}
            </Button>
          </div>
        )}
        <form onSubmit={submit} className="flex items-end gap-2">
          <label htmlFor="vp-text" className="sr-only">Type a message</label>
          <Textarea
            id="vp-text"
            rows={1}
            maxLength={MAX_TYPED}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={onKey}
            placeholder="Or type your message"
            autoComplete="off"
            aria-describedby={left < 100 ? "vp-count" : undefined}
            className="max-h-32 min-h-11 resize-none bg-card"
          />
          <Button type="submit" size="icon" className="size-11 shrink-0" aria-label={!text.trim() && canSendNow ? "Send what I said" : "Send"} disabled={(!text.trim() && !canSendNow) || (!text.trim() && sending)}>
            <SendIcon aria-hidden />
          </Button>
        </form>
        {left < 100 && (
          <p id="vp-count" className={cn("text-right text-xs text-muted-foreground", left < 20 && "text-destructive")}>
            {left} characters left
          </p>
        )}
      </div>
    </section>
  )
}
