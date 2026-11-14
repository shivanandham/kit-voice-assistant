import { useEffect, useState } from "react"

import { getJSON, type Me } from "@/lib/api"

let mePromise: Promise<Me | null> | null = null

/** GET /api/me once per page load, shared by the header and the pages. */
export function loadMe(): Promise<Me | null> {
  if (!mePromise) mePromise = getJSON<Me>("/api/me").catch(() => null)
  return mePromise
}

export const isLoggedIn = (me: Me | null | undefined): me is Me => !!me && me.role !== "visitor" && !!me.user

/** undefined while loading, null when unknown (request failed). */
export function useMe(): Me | null | undefined {
  const [me, setMe] = useState<Me | null | undefined>(undefined)
  useEffect(() => { let live = true; loadMe().then((m) => { if (live) setMe(m) }); return () => { live = false } }, [])
  return me
}

export const PROMPTS: Record<string, string[]> = {
  visitor: ["Which courses do you offer?", "What are the fees for B.Tech?", "How do I apply?", "Is there a hostel?"],
  student: ["What is my attendance?", "What's on my timetable today?", "Apply for leave tomorrow", "Which placement drives can I join?"],
  teacher: ["Show pending leave requests", "How is my class attendance?", "Which classes do I teach?"],
}
