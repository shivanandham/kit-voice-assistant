// Framework-agnostic voice engine (no DOM rendering). Ported from frontend/static/voice.js:
// microphone capture through the pcm-encoder AudioWorklet (voice gate), 16 kHz PCM upload,
// gapless 24 kHz playback, barge-in, reconnect with backoff, close codes 4401/4403/4429,
// idle/timeout handling, typed input and citations. The audio and socket logic is unchanged;
// only the DOM updates were replaced by a state snapshot that subscribers render.

export type VoiceStatus = "idle" | "listening" | "thinking" | "speaking"

export type DocumentSource = {
  kind: "document"
  ref?: string
  document_title?: string
  version?: string
  section?: string | null
  heading?: string | null
  page_start?: number | null
  page_end?: number | null
}
export type RecordSource = { kind: "college_record"; label?: string }
export type Source = DocumentSource | RecordSource

export type ChatMessage = {
  id: number
  role: "user" | "assistant"
  text: string
  sources?: Source[]
  /** a write tool ran in this turn and the reply asks a question: an action awaits confirmation */
  confirm?: boolean
  streaming?: boolean
}

export type VoiceSnapshot = {
  status: VoiceStatus
  hearing: boolean
  micOn: boolean
  interim: string
  messages: ChatMessage[]
  error: string
  tool: string | null
  canStop: boolean
}

const STORE_KEY = "kit_conversation_id"
const PLAY_RATE = 24000
export const MAX_TYPED = 1000
const WORKLET_URL = import.meta.env.BASE_URL + "pcm-worklet.js"
// Tools that only stage a pending action; the reply that follows asks the user to confirm.
const WRITE_TOOLS = new Set([
  "submit_admission_application", "submit_leave_request", "cancel_request",
  "register_for_placement_drive", "approve_request", "reject_request",
])

type Listener = (s: VoiceSnapshot) => void

export class VoiceEngine {
  private listeners = new Set<Listener>()
  private snap: VoiceSnapshot = {
    status: "idle", hearing: false, micOn: false, interim: "", messages: [], error: "",
    tool: null, canStop: false,
  }
  private nextId = 1

  // ---------- state (same variables as voice.js) ----------
  private ws: WebSocket | null = null
  private wsReady = false
  private wanted = false // we want a socket (reconnect when it drops)
  private attempts = 0
  private reconnectTimer: ReturnType<typeof setTimeout> | undefined
  private pendingText: string[] = [] // typed messages waiting for ready
  private micOn = false
  private stream: MediaStream | null = null
  private micNode: AudioWorkletNode | null = null
  private micSource: MediaStreamAudioSourceNode | null = null
  private silent: GainNode | null = null
  private ctx: AudioContext | null = null
  private nextTime = 0
  private sources = new Set<AudioBufferSourceNode>()
  private replyAudio = false // between audio_start and audio_end
  private dropAudio = false // ignore audio frames until next audio_start
  private leftover: number | null = null // odd trailing byte
  private thinking = false
  private asstId: number | null = null // the assistant message being streamed
  private hearing = false // worklet voice gate is open
  private pendingSources: Source[] | null = null
  private stopReconnect = false // server said the session timed out
  private writeToolThisTurn = false
  // Level meters for the visual only (taps; they do not change the audio path).
  private inAnalyser: AnalyserNode | null = null
  private outAnalyser: AnalyserNode | null = null
  private levelBuf = new Float32Array(512)
  private onPageHide = () => { this.closeSocket(); this.stopMic() }

  constructor() {
    window.addEventListener("pagehide", this.onPageHide)
  }

  // ---------- public API ----------
  subscribe(fn: Listener): () => void {
    this.listeners.add(fn)
    fn(this.snap)
    return () => { this.listeners.delete(fn) }
  }
  getSnapshot(): VoiceSnapshot { return this.snap }

  /** Talk button: start the microphone, or end the conversation when it is on. */
  toggleTalk(): void {
    if (this.micOn) {
      this.send({ type: "interrupt" })
      this.stopMic()
      this.flushPlayback()
      this.closeSocket()
      this.thinking = false
      try { sessionStorage.removeItem(STORE_KEY) } catch { /* storage unavailable */ }
      this.refreshState()
    } else {
      void this.startMic()
    }
  }

  /** Stop speaking button. */
  stopSpeaking(): void {
    this.send({ type: "interrupt" })
    this.flushPlayback()
    this.thinking = false
    this.refreshState()
  }

  /** Typed input. Returns false when there is nothing to send. */
  sendText(raw: string): boolean {
    const text = raw.trim().slice(0, MAX_TYPED)
    if (!text) return false
    try { this.ensureCtx() } catch { /* text still works without audio */ }
    this.addMessage("user", text)
    this.asstId = null
    this.thinking = true
    this.dropAudio = false
    this.writeToolThisTurn = false
    this.showError("")
    if (this.wsReady) this.send({ type: "text", text })
    else { this.pendingText.push(text); this.connect() }
    this.refreshState()
    return true
  }

  /** Send now: ask the server to end the current utterance and start the turn with what it has
   * heard so far (CONTRACT "Send now"). Returns false when there is nothing to send. */
  finalize(): boolean {
    if (!this.micOn || !this.snap.interim || !this.ws || this.ws.readyState !== WebSocket.OPEN) return false
    this.send({ type: "finalize" })
    return true
  }

  /** Microphone level 0..1 for the visual. */
  inputLevel(): number { return this.level(this.inAnalyser) }
  /** Playback level 0..1 for the visual. */
  outputLevel(): number { return this.level(this.outAnalyser) }

  destroy(): void {
    window.removeEventListener("pagehide", this.onPageHide)
    this.closeSocket()
    this.stopMic()
    this.listeners.clear()
  }

  // ---------- snapshot helpers ----------
  private emit(patch: Partial<VoiceSnapshot>) {
    this.snap = { ...this.snap, ...patch }
    for (const fn of this.listeners) fn(this.snap)
  }
  private setState(s: VoiceStatus) {
    this.emit({ status: s, hearing: s === "listening" && this.hearing, micOn: this.micOn,
      canStop: s === "thinking" || s === "speaking" })
  }
  private refreshState() {
    if (this.isPlaying()) this.setState("speaking")
    else if (this.thinking) this.setState("thinking")
    else if (this.micOn) this.setState("listening")
    else this.setState("idle")
  }
  private showError(msg: string) { this.emit({ error: msg }) }
  private setTool(name: string | null) { this.emit({ tool: name }) }
  private addMessage(role: "user" | "assistant", text: string): number {
    const id = this.nextId++
    this.emit({ messages: [...this.snap.messages, { id, role, text, streaming: role === "assistant" }] })
    return id
  }
  private updateMessage(id: number, fn: (m: ChatMessage) => ChatMessage) {
    this.emit({ messages: this.snap.messages.map((m) => (m.id === id ? fn(m) : m)) })
  }
  private level(a: AnalyserNode | null): number {
    if (!a) return 0
    const buf = this.levelBuf.length >= a.fftSize ? this.levelBuf : (this.levelBuf = new Float32Array(a.fftSize))
    const view = buf.subarray(0, a.fftSize)
    a.getFloatTimeDomainData(view)
    let sum = 0
    for (let i = 0; i < view.length; i++) sum += view[i] * view[i]
    return Math.min(1, Math.sqrt(sum / view.length) * 4)
  }

  // ---------- audio context ----------
  private ensureCtx(): AudioContext {
    const AC = window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
    if (!AC) throw new Error("This browser does not support Web Audio.")
    if (!this.ctx || this.ctx.state === "closed") {
      this.ctx = new AC()
      this.nextTime = 0
      this.outAnalyser = this.ctx.createAnalyser()
      this.outAnalyser.fftSize = 512
    }
    if (this.ctx.state === "suspended") this.ctx.resume().catch(() => {})
    return this.ctx
  }

  // ---------- playback ----------
  private isPlaying(): boolean {
    return this.sources.size > 0 || !!(this.ctx && this.replyAudio && this.nextTime > this.ctx.currentTime)
  }
  private flushPlayback() {
    this.dropAudio = true
    this.replyAudio = false
    this.leftover = null
    for (const s of this.sources) {
      s.onended = null
      try { s.stop() } catch { /* already stopped */ }
    }
    this.sources.clear()
    this.nextTime = 0
    this.refreshState()
  }
  private enqueue(buf: ArrayBuffer) {
    const ctx = this.ctx
    if (this.dropAudio || !ctx) return
    let bytes = new Uint8Array(buf)
    if (this.leftover !== null) {
      const m = new Uint8Array(bytes.length + 1)
      m[0] = this.leftover
      m.set(bytes, 1)
      bytes = m
      this.leftover = null
    }
    if (bytes.length % 2) {
      this.leftover = bytes[bytes.length - 1]
      bytes = bytes.subarray(0, bytes.length - 1)
    }
    const n = bytes.length / 2
    if (!n) return
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.length)
    const ab = ctx.createBuffer(1, n, PLAY_RATE)
    const ch = ab.getChannelData(0)
    for (let i = 0; i < n; i++) ch[i] = dv.getInt16(i * 2, true) / 32768
    const src = ctx.createBufferSource()
    src.buffer = ab
    src.connect(ctx.destination)
    if (this.outAnalyser) src.connect(this.outAnalyser)
    const startAt = Math.max(this.nextTime, ctx.currentTime + 0.03)
    src.start(startAt)
    this.nextTime = startAt + ab.duration
    this.sources.add(src)
    this.thinking = false
    src.onended = () => {
      this.sources.delete(src)
      if (!this.sources.size) this.refreshState()
    }
    this.refreshState()
  }

  // ---------- socket ----------
  private wsUrl(): string {
    return (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws/voice"
  }
  private connect() {
    this.wanted = true
    this.stopReconnect = false
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) return
    clearTimeout(this.reconnectTimer)
    const sock = new WebSocket(this.wsUrl())
    sock.binaryType = "arraybuffer"
    this.ws = sock
    this.wsReady = false
    sock.onopen = () => {
      let id: string | null = null
      try { id = sessionStorage.getItem(STORE_KEY) } catch { /* storage unavailable */ }
      sock.send(JSON.stringify({ type: "start", conversation_id: id || null }))
    }
    sock.onmessage = (ev) => {
      if (sock !== this.ws) return
      if (typeof ev.data === "string") {
        let m: Record<string, unknown>
        try { m = JSON.parse(ev.data) } catch { return }
        this.onEvent(m)
      } else {
        this.enqueue(ev.data as ArrayBuffer)
      }
    }
    sock.onclose = (ev) => {
      if (sock !== this.ws) return
      this.wsReady = false
      this.ws = null
      if (ev.code === 4403 || ev.code === 4429) {
        this.wanted = false
        this.showError(ev.code === 4403
          ? "This page is not allowed to connect to the assistant (origin not allowed)."
          : "Too many connections. Please close other tabs and try again shortly.")
        this.stopMic()
        return
      }
      if (this.stopReconnect) {
        this.wanted = false
        this.stopMic()
        this.thinking = false
        this.refreshState()
        return
      }
      if (ev.code === 4401) {
        this.wanted = false
        this.showError("Your session has expired. Please reload the page.")
        this.stopMic()
        return
      }
      if (!this.wanted) return
      this.attempts++
      const delay = Math.min(8000, 500 * 2 ** (this.attempts - 1))
      this.showError("Connection lost. Reconnecting…")
      this.reconnectTimer = setTimeout(() => this.connect(), delay)
    }
    sock.onerror = () => {}
  }
  private closeSocket() {
    this.wanted = false
    clearTimeout(this.reconnectTimer)
    const s = this.ws
    this.ws = null
    this.wsReady = false
    if (s) { s.onclose = null; try { s.close() } catch { /* already closed */ } }
  }
  private send(obj: unknown) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(obj))
  }

  private finishAssistant(text: string | undefined) {
    let id = this.asstId
    if (id === null && text) id = this.addMessage("assistant", text)
    if (id === null && this.pendingSources) id = this.addMessage("assistant", "")
    if (id !== null) {
      const sources = this.pendingSources && this.pendingSources.length ? this.pendingSources : undefined
      const confirm = this.writeToolThisTurn
      this.updateMessage(id, (m) => {
        const finalText = text || m.text
        return { ...m, text: finalText, sources, streaming: false,
          confirm: confirm && /\?\s*$/.test(finalText.trim()) }
      })
    }
    this.pendingSources = null
    this.asstId = null
  }
  private dropStreaming() {
    if (this.asstId !== null) this.updateMessage(this.asstId, (m) => ({ ...m, streaming: false }))
    this.asstId = null
  }

  private onEvent(m: Record<string, unknown>) {
    switch (m.type) {
      case "ready":
        this.wsReady = true
        this.attempts = 0
        this.showError("")
        try { if (m.conversation_id) sessionStorage.setItem(STORE_KEY, String(m.conversation_id)) } catch { /* storage unavailable */ }
        for (const t of this.pendingText) this.send({ type: "text", text: t })
        this.pendingText = []
        break
      case "transcript": {
        const text = String(m.text || "").trim()
        if (text && this.isPlaying()) {
          this.flushPlayback()
          this.send({ type: "interrupt" })
        }
        if (m.final) {
          this.emit({ interim: "" })
          if (text) {
            this.addMessage("user", text)
            this.asstId = null
            this.thinking = true
            this.writeToolThisTurn = false
            this.refreshState()
          }
        } else {
          this.emit({ interim: text })
        }
        break
      }
      case "assistant_delta": {
        this.thinking = false
        const delta = String(m.text || "")
        if (this.asstId === null) this.asstId = this.addMessage("assistant", "")
        this.updateMessage(this.asstId, (msg) => ({ ...msg, text: msg.text + delta }))
        this.refreshState()
        break
      }
      case "assistant_done":
        this.finishAssistant(m.text ? String(m.text) : undefined)
        this.thinking = false
        this.setTool(null)
        this.refreshState()
        break
      case "citations":
        this.pendingSources = Array.isArray(m.sources) ? (m.sources as Source[]) : null
        break
      case "tool":
        if (m.state === "start") {
          const name = String(m.name || "")
          if (WRITE_TOOLS.has(name)) this.writeToolThisTurn = true
          this.setTool(name)
        } else {
          this.setTool(null)
        }
        break
      case "audio_start":
        this.dropAudio = false
        this.replyAudio = true
        this.leftover = null
        this.thinking = false
        break
      case "audio_end":
        this.replyAudio = false
        this.refreshState()
        break
      case "interrupted":
        this.flushPlayback()
        this.dropStreaming()
        this.pendingSources = null
        this.replyAudio = false
        this.thinking = false
        this.setTool(null)
        this.refreshState()
        break
      case "error": {
        const msg = String(m.message || "")
        this.showError(msg || "Something went wrong.")
        if (/timed out|time out|idle/i.test(msg)) {
          this.stopReconnect = true
          this.showError(msg + " Press Talk to start again.")
        }
        this.pendingSources = null
        this.dropStreaming()
        this.setTool(null)
        this.replyAudio = false
        this.thinking = false
        this.refreshState()
        break
      }
    }
  }

  // ---------- microphone ----------
  private async startMic() {
    this.showError("")
    if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      this.showError("The microphone needs a secure connection (https or localhost). You can still type a message below.")
      return
    }
    let c: AudioContext
    try {
      c = this.ensureCtx() // inside the click handler for autoplay rules
    } catch (e) {
      this.showError((e as Error).message)
      return
    }
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      })
    } catch (e) {
      this.stream = null
      const err = e as { name?: string; message?: string } | null
      if (err && (err.name === "NotAllowedError" || err.name === "SecurityError")) {
        this.showError("Microphone permission was denied. Allow it in the browser settings, or type a message below.")
      } else if (err && (err.name === "NotFoundError" || err.name === "OverconstrainedError")) {
        this.showError("No microphone was found. You can type a message below.")
      } else {
        this.showError("Could not open the microphone: " + ((err && err.message) || "unknown error"))
      }
      return
    }
    try {
      await c.audioWorklet.addModule(WORKLET_URL)
      this.micSource = c.createMediaStreamSource(this.stream)
      this.micNode = new AudioWorkletNode(c, "pcm-encoder", { numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1, channelCountMode: "explicit" })
      this.micNode.port.onmessage = (ev) => {
        if (!(ev.data instanceof ArrayBuffer)) {
          this.hearing = !!(ev.data && ev.data.gate)
          this.refreshState()
          return
        }
        if (this.wsReady && this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(ev.data)
      }
      this.silent = c.createGain()
      this.silent.gain.value = 0
      this.micSource.connect(this.micNode)
      this.micNode.connect(this.silent)
      this.silent.connect(c.destination)
      this.inAnalyser = c.createAnalyser()
      this.inAnalyser.fftSize = 512
      this.micSource.connect(this.inAnalyser)
    } catch (e) {
      this.stopMic()
      this.showError("Audio capture is not supported in this browser: " + (((e as Error) && (e as Error).message) || ""))
      return
    }
    this.micOn = true
    this.connect()
    this.refreshState()
  }
  private stopMic() {
    this.micOn = false
    this.hearing = false
    if (this.micNode) { this.micNode.port.onmessage = null; try { this.micNode.disconnect() } catch { /* ignore */ } }
    if (this.micSource) { try { this.micSource.disconnect() } catch { /* ignore */ } }
    if (this.silent) { try { this.silent.disconnect() } catch { /* ignore */ } }
    if (this.inAnalyser) { try { this.inAnalyser.disconnect() } catch { /* ignore */ } }
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop())
    this.micNode = this.micSource = this.silent = this.stream = this.inAnalyser = null
    this.emit({ interim: "" })
    this.refreshState()
  }
}
