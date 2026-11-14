import { useEffect, useState } from "react"

import { SiteHeader } from "@/components/kit/site-header"
import { SourceList } from "@/components/kit/source-list"
import { Message, MessageContent } from "@/components/ui/message"
import { Skeleton } from "@/components/ui/skeleton"
import { fmtDate, getJSON, type HistoryDetail, type HistoryItem } from "@/lib/api"
import { cn } from "@/lib/utils"

export function HistoryPage() {
  const [items, setItems] = useState<HistoryItem[] | null>(null)
  const [err, setErr] = useState("")
  const [openId, setOpenId] = useState<string | null>(null)
  const [detail, setDetail] = useState<HistoryDetail | null>(null)
  const [detailState, setDetailState] = useState<"" | "loading" | "error">("")
  useEffect(() => {
    getJSON<HistoryItem[]>("/api/history").then(setItems).catch(() => setErr("Could not load history. Please try again."))
  }, [])
  const open = (id: string) => {
    setOpenId(id)
    setDetail(null)
    setDetailState("loading")
    getJSON<HistoryDetail>("/api/history/" + encodeURIComponent(id))
      .then((d) => { setDetail(d); setDetailState("") })
      .catch(() => setDetailState("error"))
  }
  return (
    <>
      <SiteHeader />
      <main className="mx-auto max-w-6xl px-4 py-6">
        <h1 className="mb-4 text-2xl font-semibold">Past conversations</h1>
        {err ? <p className="text-muted-foreground">{err}</p> : !items ? <Skeleton className="h-40 w-full" /> : !items.length ? (
          <p className="text-muted-foreground">No conversations yet. Talk to the assistant and they will appear here.</p>
        ) : (
          <div className="grid grid-cols-[minmax(0,1fr)] gap-6 md:grid-cols-[18rem_minmax(0,1fr)]">
            <nav aria-label="Conversations" className="rounded-xl border bg-card p-2 md:h-[calc(100dvh-11rem)] md:min-h-80">
              <ul className="flex max-h-[40vh] flex-col gap-1 overflow-y-auto md:h-full md:max-h-none">
                {items.map((it) => (
                  <li key={it.id}>
                    <button type="button" onClick={() => open(it.id)} aria-current={openId === it.id ? "true" : undefined}
                      className={cn("flex w-full flex-col items-start gap-0.5 rounded-lg px-3 py-2 text-left hover:bg-muted",
                        openId === it.id && "bg-secondary")}>
                      <span className="line-clamp-2 text-sm font-medium">{it.title || "Untitled conversation"}</span>
                      <span className="text-xs text-muted-foreground">{fmtDate(it.updated_at)} · {it.message_count} messages</span>
                    </button>
                  </li>
                ))}
              </ul>
            </nav>
            <section aria-label="Transcript" aria-live="polite" className="min-h-64 overflow-y-auto rounded-xl border bg-card p-4 sm:p-6 max-md:max-h-[75vh] md:h-[calc(100dvh-11rem)] md:min-h-80">
              <h2 className="mb-2 text-lg font-semibold">
                {detailState === "loading" ? "Loading…" : detailState === "error" ? "Could not load this conversation." : detail ? (detail.title || "Untitled conversation") : "Select a conversation"}
              </h2>
              {detail && !detail.messages.length && <p className="text-muted-foreground">No messages.</p>}
              {detail?.messages.map((m, i) => {
                const user = m.role === "user"
                return (
                  <Message key={i} from={user ? "user" : "assistant"} className="py-2">
                    <MessageContent variant="flat" className="text-[0.95rem] leading-relaxed whitespace-pre-wrap group-[.is-user]:bg-primary/10 group-[.is-user]:rounded-2xl">
                      <span className="text-xs opacity-70">{user ? "You" : "Assistant"} · {fmtDate(m.created_at)}</span>
                      <span>{m.content}</span>
                      {!user && <SourceList sources={m.citations} />}
                    </MessageContent>
                  </Message>
                )
              })}
            </section>
          </div>
        )}
      </main>
    </>
  )
}
