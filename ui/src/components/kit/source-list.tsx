import { ChevronDownIcon, DatabaseIcon, FileTextIcon } from "lucide-react"

import { Sources, SourcesContent, SourcesTrigger } from "@/components/ai-elements/sources"
import { sourceText } from "@/lib/api"
import type { Source } from "@/voice/engine"

/** Compact "Sources" list under a reply (AI Elements Sources, open by default). */
export function SourceList({ sources }: { sources?: Source[] | null }) {
  const items = (sources ?? []).map((s) => ({ s, text: sourceText(s) })).filter((x) => x.text)
  if (!items.length) return null
  return (
    <Sources {...({ defaultOpen: true } as object)} className="mt-2 mb-0 text-muted-foreground">
      <SourcesTrigger
        count={items.length}
        className="group/src rounded-sm text-xs font-medium hover:text-foreground"
      >
        <span>Sources ({items.length})</span>
        <ChevronDownIcon aria-hidden className="size-3.5 transition-transform group-data-[state=closed]/src:-rotate-90" />
      </SourcesTrigger>
      <SourcesContent className="mt-1.5 w-full gap-1">
        <ul className="flex flex-col gap-1">
          {items.map(({ s, text }, i) => (
            <li key={i} className="flex items-start gap-2 text-xs leading-snug">
              {s.kind === "college_record" ? (
                <DatabaseIcon aria-hidden className="mt-0.5 size-3.5 shrink-0 text-primary" />
              ) : (
                <FileTextIcon aria-hidden className="mt-0.5 size-3.5 shrink-0 text-primary" />
              )}
              <span>{text}</span>
            </li>
          ))}
        </ul>
      </SourcesContent>
    </Sources>
  )
}
