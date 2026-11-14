import { Orb, type AgentState } from "@/components/ui/orb"
import type { VoiceStatus } from "@/voice/engine"

const AGENT: Record<VoiceStatus, AgentState> = {
  idle: null, listening: "listening", thinking: "thinking", speaking: "talking",
}

type Props = {
  status: VoiceStatus
  dark: boolean
  inputLevel: () => number
  outputLevel: () => number
}

/** ElevenLabs UI Orb driven by the engine: live levels while listening or speaking. */
export default function VoiceOrb({ status, dark, inputLevel, outputLevel }: Props) {
  const live = status === "listening" || status === "speaking"
  return (
    <Orb
      className="size-full"
      colors={dark ? ["#60a5fa", "#4f46e5"] : ["#bfdbfe", "#818cf8"]}
      agentState={AGENT[status]}
      volumeMode={live ? "manual" : "auto"}
      getInputVolume={() => (status === "listening" ? 0.25 + inputLevel() * 0.9 : 0.5)}
      getOutputVolume={() => (status === "speaking" ? 0.45 + outputLevel() * 1.2 : 0.4)}
    />
  )
}
