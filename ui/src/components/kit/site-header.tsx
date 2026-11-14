import type { ReactNode } from "react"
import { MoonIcon, SunIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { useTheme } from "@/lib/theme"

export function SiteHeader({ children }: { children?: ReactNode }) {
  const [theme, toggle] = useTheme()
  return (
    <header className="border-b bg-card/80 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-3 px-4 py-3">
        <a href="/" className="flex items-center gap-2 rounded-md font-semibold">
          <span aria-hidden className="grid size-8 place-items-center rounded-lg bg-primary text-sm font-bold text-primary-foreground">K</span>
          <span>KIT <span className="font-normal text-muted-foreground">Assistant</span></span>
        </a>
        <nav aria-label="Main" className="flex items-center gap-1">
          {children}
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
