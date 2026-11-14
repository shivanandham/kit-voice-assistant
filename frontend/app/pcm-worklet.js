// Converts microphone audio (any context sample rate) to 16 kHz mono PCM16 LE, ~40 ms frames.
const TARGET_RATE = 16000;
const FRAME_SAMPLES = 640; // 40 ms at 16 kHz
// Voice gate thresholds (RMS on a 0..1 scale). Biased towards letting audio through.
const MIN_SPEECH_RMS = 0.005; // absolute minimum level (about -46 dBFS) so electrical noise never opens the gate
const SPEECH_RATIO = 2.5; // a frame is speech when its level exceeds the noise floor by this factor
const FLOOR_WINDOW_FRAMES = 150; // noise floor = minimum frame level over the last 6 s
const OPEN_FRAMES = 2; // consecutive speech frames needed to open the gate
const PREROLL_FRAMES = 12; // frames kept while closed and flushed on open (about 480 ms, includes the opening frames)
const HANGOVER_FRAMES = 30; // keep the gate open 1.2 s after the last speech frame

class PcmEncoder extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / TARGET_RATE; // input samples per output sample
    this.buf = new Float32Array(0); // unconsumed input samples
    this.pos = 0; // fractional read position inside buf
    this.frame = new Int16Array(FRAME_SAMPLES);
    this.filled = 0;
    this.levels = []; // recent frame levels for the noise floor
    this.preroll = [];
    this.open = false;
    this.run = 0; // consecutive speech frames
    this.hang = 0;
  }

  // Decide whether to upload this 40 ms frame (Int16Array); posts frames and gate events.
  gate(frame) {
    let sum = 0;
    for (let i = 0; i < frame.length; i++) { const v = frame[i] / 32768; sum += v * v; }
    const level = Math.sqrt(sum / frame.length);
    this.levels.push(level);
    if (this.levels.length > FLOOR_WINDOW_FRAMES) this.levels.shift();
    let floor = Infinity;
    for (const l of this.levels) if (l < floor) floor = l;
    const speech = level > Math.max(MIN_SPEECH_RMS, floor * SPEECH_RATIO);
    if (this.open) {
      this.port.postMessage(frame.buffer, [frame.buffer]);
      this.hang = speech ? HANGOVER_FRAMES : this.hang - 1;
      if (this.hang <= 0) {
        this.open = false;
        this.run = 0;
        this.port.postMessage({ gate: false });
      }
      return;
    }
    this.preroll.push(frame);
    if (this.preroll.length > PREROLL_FRAMES) this.preroll.shift();
    this.run = speech ? this.run + 1 : 0;
    if (this.run >= OPEN_FRAMES) {
      this.open = true;
      this.hang = HANGOVER_FRAMES;
      this.port.postMessage({ gate: true });
      for (const f of this.preroll) this.port.postMessage(f.buffer, [f.buffer]);
      this.preroll = [];
    }
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    const merged = new Float32Array(this.buf.length + ch.length);
    merged.set(this.buf);
    merged.set(ch, this.buf.length);
    const buf = merged;
    let pos = this.pos;
    // Box-filter average over each output sample's input window (anti-aliasing for downsampling).
    while (Math.floor(pos + this.ratio) <= buf.length - (this.ratio > 1 ? 0 : 1)) {
      const start = Math.floor(pos);
      let end = Math.floor(pos + this.ratio);
      if (end <= start) end = start + 1;
      if (end > buf.length) break;
      let sum = 0;
      for (let i = start; i < end; i++) sum += buf[i];
      const s = Math.max(-1, Math.min(1, sum / (end - start)));
      this.frame[this.filled++] = s < 0 ? s * 0x8000 : s * 0x7fff;
      pos += this.ratio;
      if (this.filled === FRAME_SAMPLES) {
        const done = this.frame;
        this.frame = new Int16Array(FRAME_SAMPLES);
        this.gate(done);
        this.filled = 0;
      }
    }
    const drop = Math.min(Math.floor(pos), buf.length);
    this.buf = buf.slice(drop);
    this.pos = pos - drop;
    return true;
  }
}

registerProcessor("pcm-encoder", PcmEncoder);
