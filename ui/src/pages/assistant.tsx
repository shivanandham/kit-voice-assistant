import { useEffect, useState } from "react"

import { SiteHeader } from "@/components/kit/site-header"
import { isLoggedIn, loadMe, PROMPTS } from "@/lib/me"
import { VoiceAssistant } from "@/components/kit/voice-assistant"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import type { Me } from "@/lib/api"


export function AssistantPage() {
  const [me, setMe] = useState<Me | null>(null)
  useEffect(() => {
    loadMe().then((m) => {
      if (isLoggedIn(m)) setMe(m)
      else location.href = "/login"
    })
  }, [])
  const u = me?.user
  return (
    <>
      <SiteHeader />
      <main className="mx-auto flex max-w-3xl flex-col gap-4 px-4 py-5">
        {!me ? <Skeleton className="h-14 w-full" /> : (
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h1 className="text-2xl font-semibold">Hello, {u?.full_name || u?.login_id}</h1>
              <p className="text-sm text-muted-foreground">Logged in as {u?.login_id}</p>
            </div>
            <div className="flex flex-wrap gap-2">
              <Badge variant="secondary" className="capitalize">{me.role}</Badge>
              {u?.class_code && <Badge variant="outline">Class {u.class_code}</Badge>}
              {u?.department && <Badge variant="outline">{u.department}</Badge>}
            </div>
          </div>
        )}
        {me && <VoiceAssistant tall suggestions={PROMPTS[me.role] ?? []} />}
      </main>
    </>
  )
}
