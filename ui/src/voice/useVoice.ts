import { useEffect, useState, useSyncExternalStore } from "react"

import { VoiceEngine, type VoiceSnapshot } from "@/voice/engine"

/** One engine per mounted assistant; React re-renders from its snapshots. */
export function useVoice(): [VoiceSnapshot, VoiceEngine] {
  const [engine] = useState(() => new VoiceEngine())
  useEffect(() => () => engine.destroy(), [engine])
  const snap = useSyncExternalStore(
    (cb) => engine.subscribe(cb),
    () => engine.getSnapshot(),
  )
  return [snap, engine]
}
