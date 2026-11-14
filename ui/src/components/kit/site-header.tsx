import type { ReactNode } from "react"
import { LogOutIcon, MoonIcon, SunIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { logout } from "@/lib/api"
import { isLoggedIn, useMe } from "@/lib/me"
import { useTheme } from "@/lib/theme"

/** Shared header: shows the login state from GET /api/me on every page. */
export function SiteHeader() {
  const [theme, toggle] = useTheme()
  const me = useMe()
  const onLogout = async () => { await logout(); location.href = "/" }
  const u = isLoggedIn(me) ? me.user : null
  return (
    <header className="border-b bg-card/80 backdrop-blur">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-4 py-3">
        <a href={u ? "/assistant" : "/"} className="flex items-center gap-2 rounded-md font-semibold">
          <span aria-hidden className="grid size-8 place-items-center rounded-lg bg-primary text-sm font-bold text-primary-foreground">K</span>
          <span>KIT <span className="font-normal text-muted-foreground">Assistant</span></span>
        </a>
        <nav aria-label="Main" className="flex flex-wrap items-center gap-1">
          {me === undefined ? (
            <Skeleton aria-label="Loading" className="h-8 w-40" />
          ) : u && me ? (
            <>
              <span className="mr-1 hidden text-sm text-muted-foreground sm:inline">
                {u.full_name || u.login_id} · <span className="capitalize">{me.role}</span>
                {me.role === "student" && u.class_code ? " · " + u.class_code : ""}
              </span>
              <NavLink href="/assistant">My assistant</NavLink>
              <NavLink href="/history">History</NavLink>
              <Button type="button" variant="ghost" size="sm" onClick={onLogout}><LogOutIcon aria-hidden />Log out</Button>
            </>
          ) : (
            <>
              {location.pathname !== "/" && <NavLink href="/">Home</NavLink>}
              <NavLink href="/history">History</NavLink>
              <NavLink href="/login">Log in</NavLink>
            </>
          )}
          <Button type="button" variant="ghost" size="icon" onClick={toggle}
            aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}>
            {theme === "dark" ? <SunIcon aria-hidden /> : <MoonIcon aria-hidden />}
          </Button>
        </nav>
      </div>
    </header>
  )
}

export function NavLink({ href, children }: { href: string; children: ReactNode }) {
  const current = location.pathname === href
  return (
    <Button asChild variant="ghost" size="sm">
      <a href={href} aria-current={current ? "page" : undefined}>{children}</a>
    </Button>
  )
}
