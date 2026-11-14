import { useEffect, useState } from "react"
import { MailIcon, MapPinIcon, PhoneIcon, GlobeIcon } from "lucide-react"

import { NavLink, SiteHeader } from "@/components/kit/site-header"
import { VoiceAssistant } from "@/components/kit/voice-assistant"
import { Skeleton } from "@/components/ui/skeleton"
import { Table, TableBody, TableCaption, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { getJSON, rupees, type College, type Course } from "@/lib/api"

const VISITOR_PROMPTS = ["Which courses do you offer?", "What are the fees for B.Tech?", "How do I apply?", "Is there a hostel?"]

export function HomePage() {
  const [college, setCollege] = useState<College | null>(null)
  const [collegeErr, setCollegeErr] = useState("")
  const [courses, setCourses] = useState<Course[] | null>(null)
  const [coursesErr, setCoursesErr] = useState("")
  useEffect(() => {
    getJSON<College>("/api/public/college").then(setCollege).catch(() => setCollegeErr("College details are unavailable right now."))
    getJSON<Course[]>("/api/public/courses").then(setCourses).catch(() => setCoursesErr("Courses are unavailable right now."))
  }, [])
  const k = college?.contact ?? {}
  return (
    <>
      <SiteHeader><NavLink href="/history">History</NavLink><NavLink href="/login">Log in</NavLink></SiteHeader>
      <main className="mx-auto max-w-6xl px-4 py-8 sm:py-12">
        <div className="grid grid-cols-[minmax(0,1fr)] items-start gap-8 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
          <div className="flex flex-col gap-5 lg:pt-4">
            <h1 className="text-4xl font-bold sm:text-5xl">{college?.name || "Knotopian Institute of Technology"}</h1>
            {college?.tagline && <p className="text-xl text-primary">{college.tagline}</p>}
            {college ? (
              <p className="max-w-prose leading-relaxed text-muted-foreground">{college.description}</p>
            ) : collegeErr ? (
              <p className="text-muted-foreground">{collegeErr}</p>
            ) : (
              <div className="flex flex-col gap-2"><Skeleton className="h-4 w-full" /><Skeleton className="h-4 w-4/5" /></div>
            )}
            <ul className="flex flex-col gap-2 text-sm">
              {k.phone && <li className="flex items-center gap-2"><PhoneIcon aria-hidden className="size-4 text-primary" />{k.phone}</li>}
              {k.email && <li className="flex items-center gap-2"><MailIcon aria-hidden className="size-4 text-primary" />{k.email}</li>}
              {k.website && <li className="flex items-center gap-2"><GlobeIcon aria-hidden className="size-4 text-primary" />{k.website}</li>}
              {k.address && <li className="flex items-start gap-2"><MapPinIcon aria-hidden className="mt-0.5 size-4 text-primary" />{k.address}</li>}
            </ul>
          </div>
          <div className="flex min-w-0 flex-col gap-3">
            <h2 className="text-lg font-semibold">Talk to us</h2>
            <p className="-mt-2 text-sm text-muted-foreground">Ask about courses, fees and admissions by voice or text.</p>
            <VoiceAssistant suggestions={VISITOR_PROMPTS} />
          </div>
        </div>

        <section aria-labelledby="courses-h" className="mt-14">
          <h2 id="courses-h" className="mb-4 text-2xl font-semibold">Courses</h2>
          {coursesErr ? <p className="text-muted-foreground">{coursesErr}</p> : !courses ? (
            <Skeleton className="h-40 w-full" />
          ) : !courses.length ? <p className="text-muted-foreground">No courses listed.</p> : (
            <div className="rounded-xl border bg-card">
              <Table>
                <TableCaption className="pb-3">Fees in rupees per year</TableCaption>
                <TableHeader>
                  <TableRow>
                    <TableHead scope="col">Code</TableHead><TableHead scope="col">Course</TableHead>
                    <TableHead scope="col">Department</TableHead><TableHead scope="col" className="text-right">Min 12th %</TableHead>
                    <TableHead scope="col" className="text-right">Annual fee</TableHead><TableHead scope="col" className="text-right">Hostel fee</TableHead>
                    <TableHead scope="col" className="text-right">Seats</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {courses.map((c, i) => (
                    <TableRow key={c.code ?? i}>
                      <TableCell className="font-medium">{c.code}</TableCell>
                      <TableCell>{c.name}</TableCell>
                      <TableCell className="text-muted-foreground">{c.department}</TableCell>
                      <TableCell className="text-right tabular-nums">{c.min_12th_percent != null ? c.min_12th_percent + "%" : "–"}</TableCell>
                      <TableCell className="text-right tabular-nums">{rupees(c.annual_fee)}</TableCell>
                      <TableCell className="text-right tabular-nums">{rupees(c.hostel_fee)}</TableCell>
                      <TableCell className="text-right tabular-nums">{c.seats ?? "–"}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </section>
      </main>
    </>
  )
}
