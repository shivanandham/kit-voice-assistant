import type { Source } from "@/voice/engine"

export type User = {
  id?: string | number
  login_id?: string
  full_name?: string
  role?: string
  department?: string
  class_id?: string | number
  class_code?: string
  programme?: string
  year_of_study?: number
}
export type Me = { role: "visitor" | "student" | "teacher"; user: User | null }
export type College = {
  name?: string
  tagline?: string
  description?: string
  contact?: { phone?: string; email?: string; website?: string; address?: string }
}
export type Course = {
  code?: string
  name?: string
  department?: string
  min_12th_percent?: number | null
  annual_fee?: number | null
  hostel_fee?: number | null
  seats?: number | null
}
export type HistoryItem = { id: string; title: string | null; started_at: string; updated_at: string; message_count: number }
export type HistoryMessage = { role: string; content: string; created_at: string; citations?: Source[] | null }
export type HistoryDetail = { id: string; title: string | null; messages: HistoryMessage[] }

export async function errorMessage(r: Response): Promise<string> {
  try {
    const j = await r.json()
    return (j.error && j.error.message) || j.message || (typeof j.detail === "string" ? j.detail : "") || `Request failed (${r.status})`
  } catch {
    return `Request failed (${r.status})`
  }
}

export async function getJSON<T>(url: string): Promise<T> {
  const r = await fetch(url, { credentials: "same-origin" })
  if (!r.ok) throw new Error(await errorMessage(r))
  return r.json() as Promise<T>
}

export async function login(login_id: string, password: string): Promise<void> {
  const r = await fetch("/api/login", {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ login_id, password }),
  })
  if (!r.ok) {
    const msg = await errorMessage(r)
    if (r.status === 429) throw new Error(msg || "Too many attempts. Wait a minute and try again.")
    if (r.status === 503) throw new Error(msg || "The college system is unavailable right now. Try again shortly.")
    throw new Error(msg || "Login failed.")
  }
}

export async function logout(): Promise<void> {
  try { await fetch("/api/logout", { method: "POST", credentials: "same-origin" }) } catch { /* ignore */ }
  try { sessionStorage.removeItem("kit_conversation_id") } catch { /* ignore */ }
}

/** "Student Handbook 2026 (2026.1), rule 6.4.3, page 14" or the college record label. */
export function sourceText(s: Source | null | undefined): string {
  if (!s) return ""
  if (s.kind === "college_record") return s.label || "College system"
  const parts: string[] = []
  if (s.document_title) parts.push(s.document_title + (s.version ? " (" + s.version + ")" : ""))
  const title = s.document_title || ""
  // Only the Student Handbook has rules. A circular or notice whose section is its own number
  // (for example "12/2026") gets no section part.
  const ownNumber = !!s.section && (/^\d+\/\d{4}$/.test(s.section) || title.includes(s.section + "/") || title.includes(s.section) && s.section.includes("/"))
  if (s.section && !ownNumber) parts.push((/handbook/i.test(title) ? "rule " : "section ") + s.section)
  else if (!s.section && s.heading) parts.push(s.heading)
  if (s.page_start != null) {
    parts.push(s.page_end != null && s.page_end !== s.page_start ? "pages " + s.page_start + "–" + s.page_end : "page " + s.page_start)
  }
  return parts.join(", ")
}

export function fmtDate(s: string): string {
  const d = new Date(s)
  return isNaN(d.getTime()) ? "" : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })
}

export const rupees = (n: number | null | undefined) =>
  n == null ? "–" : "₹" + Number(n).toLocaleString("en-IN")

export function toolLabel(name: string): string {
  const map: Record<string, string> = {
    get_attendance_summary: "attendance", get_attendance_sessions: "attendance records",
    get_timetable: "the timetable", get_calendar: "the calendar", get_today: "today's date",
    list_placement_drives: "placement drives", list_my_placements: "your placements",
    search_documents: "college documents", get_my_profile: "your profile",
    get_my_academics: "academics", get_my_subjects: "your subjects",
    list_my_requests: "your requests", get_request: "the request", list_courses: "courses",
    get_college_info: "college details", list_my_classes: "your classes",
    get_class_attendance: "class attendance", list_class_requests: "pending requests",
    submit_leave_request: "the leave request", submit_admission_application: "the application",
    register_for_placement_drive: "the registration", cancel_request: "the cancellation",
    approve_request: "the approval", reject_request: "the rejection",
    confirm_pending_action: "the college system", cancel_pending_action: "the pending action",
  }
  return map[name] || name.replace(/_/g, " ")
}
