import { useEffect, useRef } from "react"

const BARS = 32

/** Scrolling bar waveform driven by a 0..1 level meter (no inline style attributes: CSSOM only). */
export function LevelBars({ level, still }: { level: () => number; still?: boolean }) {
  const box = useRef<HTMLDivElement>(null)
  const levelRef = useRef(level)
  levelRef.current = level
  useEffect(() => {
    const bars = Array.from(box.current?.children ?? []) as HTMLElement[]
    const hist = new Array(BARS).fill(0.08)
    if (still) { bars.forEach((b) => { b.style.transform = "scaleY(0.25)" }); return }
    let raf = 0
    let last = 0
    const tick = (t: number) => {
      if (t - last > 60) {
        last = t
        hist.shift()
        hist.push(Math.max(0.08, Math.min(1, levelRef.current())))
        bars.forEach((b, i) => { b.style.transform = `scaleY(${hist[i]})` })
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [still])
  return (
    <div ref={box} data-testid="waveform" aria-hidden className="flex h-5 items-center gap-[3px]">
      {Array.from({ length: BARS }, (_, i) => (
        <span key={i} className="h-full w-[3px] origin-center scale-y-[0.08] rounded-full bg-primary" />
      ))}
    </div>
  )
}
