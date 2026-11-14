import { useEffect, useState } from "react"
import { LogOutIcon } from "lucide-react"

import { NavLink, SiteHeader } from "@/components/kit/site-header"
import { VoiceAssistant } from "@/components/kit/voice-assistant"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { getJSON, logout, type Me } from "@/lib/api"

const PROMPTS: Record<string, string[]> = {
  student: ["What is my attendance?", "What's on my timetable today?", "Apply for leave tomorrow", "Which placement drives can I join?"],
  teacher: ["Show pending leave requests", "How is my class attendance?", "Which classes do I teach?"],
}

export function AssistantPage() {
  const [me, setMe] = useState<Me | null>(null)
  useEffect(() => {
    getJSON<Me>("/api/me").then((m) => {
      if (m.role === "visitor" || !m.user) location.href = "/login"
      else setMe(m)
    }).catch(() => { location.href = "/login" })
  }, [])
  const u = me?.user
  const onLogout = async () => { await logout(); location.href = "/" }
  return (
    <>
      <SiteHeader>
        <NavLink href="/history">History</NavLink>
        <Button type="button" variant="ghost" size="sm" onClick={onLogout}><LogOutIcon aria-hidden />Log out</Button>
      </SiteHeader>
      <main className="mx-auto flex max-w-3xl flex-col gap-5 px-4 py-8">
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
