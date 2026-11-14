import { useEffect, useState } from "react"

export type Theme = "light" | "dark"
const KEY = "kit_theme"

// Light by default whatever the system says; dark only when chosen with the toggle.
function systemTheme(): Theme {
  return "light"
}

function stored(): Theme | null {
  try {
    const v = localStorage.getItem(KEY)
    return v === "light" || v === "dark" ? v : null
  } catch {
    return null
  }
}

/** Apply the saved or system theme before the first render (no inline script under the CSP). */
export function applyInitialTheme(): void {
  document.documentElement.classList.toggle("dark", (stored() ?? systemTheme()) === "dark")
}

export function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(() => stored() ?? systemTheme())
  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark")
  }, [theme])
  const toggle = () => {
    const next: Theme = theme === "dark" ? "light" : "dark"
    try { localStorage.setItem(KEY, next) } catch { /* per-viewer convenience only */ }
    setTheme(next)
  }
  return [theme, toggle]
}

export function usePrefersReducedMotion(): boolean {
  const q = "(prefers-reduced-motion: reduce)"
  const [reduced, setReduced] = useState(() => !!window.matchMedia && window.matchMedia(q).matches)
  useEffect(() => {
    if (!window.matchMedia) return
    const mq = window.matchMedia(q)
    const on = () => setReduced(mq.matches)
    mq.addEventListener("change", on)
    return () => mq.removeEventListener("change", on)
  }, [])
  return reduced
}
