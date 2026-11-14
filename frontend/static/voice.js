// Reusable voice assistant panel. Usage: createVoicePanel(containerElement)
import { el, renderSources } from "/static/api.js";

const STORE_KEY = "kit_conversation_id";
const PLAY_RATE = 24000;

export function createVoicePanel(root) {
  const AC = window.AudioContext || window.webkitAudioContext;

  // ---------- DOM ----------
  root.textContent = "";
  const stateEl = el("p", "vp-state state-idle", "Idle");
  stateEl.setAttribute("role", "status");
  stateEl.setAttribute("aria-live", "polite");
  const talkBtn = el("button", "vp-talk", "Talk");
  talkBtn.type = "button";
  talkBtn.setAttribute("aria-pressed", "false");
  const stopBtn = el("button", "vp-stop", "Stop");
  stopBtn.type = "button";
  stopBtn.setAttribute("aria-label", "Stop the assistant speaking");
  const errorEl = el("p", "vp-error");
  errorEl.setAttribute("role", "alert");
  errorEl.hidden = true;
  const interimEl = el("p", "vp-interim");
  interimEl.setAttribute("aria-live", "off");
  const toolEl = el("p", "vp-tool");
  toolEl.hidden = true;
  const logEl = el("div", "vp-log");
  logEl.setAttribute("role", "log");
  logEl.setAttribute("aria-label", "Conversation");
  const form = el("form", "vp-form");
  const input = el("input");
  input.type = "text";
  input.id = "vp-text-" + Math.random().toString(36).slice(2, 8);
  input.autocomplete = "off";
  input.maxLength = 1000;
  input.placeholder = "Or type your message";
  const label = el("label", "vp-label", "Type a message");
  label.htmlFor = input.id;
  const sendBtn = el("button", "vp-send", "Send");
  sendBtn.type = "submit";
  const counter = el("span", "vp-count");
  counter.hidden = true;
  form.append(input, sendBtn);
  const controls = el("div", "vp-controls");
  controls.append(talkBtn, stopBtn);
  root.append(stateEl, controls, errorEl, logEl, interimEl, toolEl, label, form, counter);

  // ---------- state ----------
  let ws = null;
  let wsReady = false;
  let wanted = false; // we want a socket (reconnect when it drops)
  let attempts = 0;
  let reconnectTimer = null;
  let pendingText = []; // typed messages waiting for ready
  let micOn = false;
  let stream = null, micNode = null, micSource = null, silent = null;
  let ctx = null;
  let nextTime = 0;
  const sources = new Set();
  let replyAudio = false; // between audio_start and audio_end
  let dropAudio = false; // ignore audio frames until next audio_start
  let leftover = null; // odd trailing byte
  let thinking = false;
  let asstBubble = null;
  let hearing = false; // worklet voice gate is open
  let pendingSources = null;
  let stopReconnect = false; // server said the session timed out

  function setState(s) {
    const names = { idle: "Idle", listening: "Listening", thinking: "Thinking", speaking: "Speaking" };
    stateEl.textContent = names[s] + (s === "listening" && hearing ? " (hearing you)" : "");
    stateEl.className = "vp-state state-" + s;
  }
  function refreshState() {
    if (isPlaying()) setState("speaking");
    else if (thinking) setState("thinking");
    else if (micOn) setState("listening");
    else setState("idle");
  }
  function showError(msg) {
    errorEl.textContent = msg;
    errorEl.hidden = !msg;
  }
  function addBubble(role, text) {
    const b = el("div", "msg msg-" + role);
    b.append(el("span", "who", role === "user" ? "You" : "Assistant"));
    const t = el("span", "txt", text);
    b.append(t);
    logEl.append(b);
    logEl.scrollTop = logEl.scrollHeight;
    return t;
  }

  // ---------- audio context ----------
  function ensureCtx() {
    if (!AC) throw new Error("This browser does not support Web Audio.");
    if (!ctx || ctx.state === "closed") {
      ctx = new AC();
      nextTime = 0;
    }
    if (ctx.state === "suspended") ctx.resume().catch(() => {});
    return ctx;
  }

  // ---------- playback ----------
  function isPlaying() {
    return sources.size > 0 || (ctx && replyAudio && nextTime > ctx.currentTime);
  }
  function flushPlayback() {
    dropAudio = true;
    replyAudio = false;
    leftover = null;
    for (const s of sources) {
      s.onended = null;
      try { s.stop(); } catch {}
    }
    sources.clear();
    nextTime = 0;
    refreshState();
  }
  function enqueue(buf) {
    if (dropAudio || !ctx) return;
    let bytes = new Uint8Array(buf);
    if (leftover !== null) {
      const m = new Uint8Array(bytes.length + 1);
      m[0] = leftover;
      m.set(bytes, 1);
      bytes = m;
      leftover = null;
    }
    if (bytes.length % 2) {
      leftover = bytes[bytes.length - 1];
      bytes = bytes.subarray(0, bytes.length - 1);
    }
    const n = bytes.length / 2;
    if (!n) return;
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.length);
    const ab = ctx.createBuffer(1, n, PLAY_RATE);
    const ch = ab.getChannelData(0);
    for (let i = 0; i < n; i++) ch[i] = dv.getInt16(i * 2, true) / 32768;
    const src = ctx.createBufferSource();
    src.buffer = ab;
    src.connect(ctx.destination);
    const startAt = Math.max(nextTime, ctx.currentTime + 0.03);
    src.start(startAt);
    nextTime = startAt + ab.duration;
    sources.add(src);
    thinking = false;
    src.onended = () => {
      sources.delete(src);
      if (!sources.size) refreshState();
    };
    refreshState();
  }

  // ---------- socket ----------
  function wsUrl() {
    return (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws/voice";
  }
  function connect() {
    wanted = true;
    stopReconnect = false;
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
    clearTimeout(reconnectTimer);
    const sock = new WebSocket(wsUrl());
    sock.binaryType = "arraybuffer";
    ws = sock;
    wsReady = false;
    sock.onopen = () => {
      let id = null;
      try { id = sessionStorage.getItem(STORE_KEY); } catch {}
      sock.send(JSON.stringify({ type: "start", conversation_id: id || null }));
    };
    sock.onmessage = (ev) => {
      if (sock !== ws) return;
      if (typeof ev.data === "string") {
        let m;
        try { m = JSON.parse(ev.data); } catch { return; }
        onEvent(m);
      } else {
        enqueue(ev.data);
      }
    };
    sock.onclose = (ev) => {
      if (sock !== ws) return;
      wsReady = false;
      ws = null;
      if (ev.code === 4403 || ev.code === 4429) {
        wanted = false;
        showError(ev.code === 4403 ? "This page is not allowed to connect to the assistant (origin not allowed)." : "Too many connections. Please close other tabs and try again shortly.");
        stopMic();
        return;
      }
      if (stopReconnect) {
        wanted = false;
        stopMic();
        thinking = false;
        refreshState();
        return;
      }
      if (ev.code === 4401) {
        wanted = false;
        showError("Your session has expired. Please reload the page.");
        stopMic();
        return;
      }
      if (!wanted) return;
      attempts++;
      const delay = Math.min(8000, 500 * 2 ** (attempts - 1));
      showError("Connection lost. Reconnecting…");
      reconnectTimer = setTimeout(connect, delay);
    };
    sock.onerror = () => {};
  }
  function closeSocket() {
    wanted = false;
    clearTimeout(reconnectTimer);
    const s = ws;
    ws = null;
    wsReady = false;
    if (s) { s.onclose = null; try { s.close(); } catch {} }
  }
  function send(obj) {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
  }

  function onEvent(m) {
    switch (m.type) {
      case "ready":
        wsReady = true;
        attempts = 0;
        showError("");
        try { if (m.conversation_id) sessionStorage.setItem(STORE_KEY, m.conversation_id); } catch {}
        for (const t of pendingText) send({ type: "text", text: t });
        pendingText = [];
        break;
      case "transcript": {
        const text = (m.text || "").trim();
        if (text && isPlaying()) {
          flushPlayback();
          send({ type: "interrupt" });
        }
        if (m.final) {
          interimEl.textContent = "";
          if (text) {
            addBubble("user", text);
            asstBubble = null;
            thinking = true;
            refreshState();
          }
        } else {
          interimEl.textContent = text;
        }
        break;
      }
      case "assistant_delta":
        thinking = false;
        if (!asstBubble) asstBubble = addBubble("assistant", "");
        asstBubble.textContent += m.text || "";
        logEl.scrollTop = logEl.scrollHeight;
        refreshState();
        break;
      case "assistant_done":
        if (!asstBubble && m.text) asstBubble = addBubble("assistant", m.text);
        else if (asstBubble && m.text) asstBubble.textContent = m.text;
        if (!asstBubble && pendingSources) asstBubble = addBubble("assistant", "");
        if (pendingSources) {
          const host = (asstBubble || addBubble("assistant", m.text || "")).parentNode;
          renderSources(host, pendingSources);
          pendingSources = null;
        }
        asstBubble = null;
        thinking = false;
        toolEl.hidden = true;
        refreshState();
        break;
      case "citations":
        pendingSources = Array.isArray(m.sources) ? m.sources : null;
        break;
      case "tool":
        if (m.state === "start") {
          toolEl.textContent = "Checking " + String(m.name || "").replace(/_/g, " ") + "…";
          toolEl.hidden = false;
        } else {
          toolEl.hidden = true;
        }
        break;
      case "audio_start":
        dropAudio = false;
        replyAudio = true;
        leftover = null;
        thinking = false;
        break;
      case "audio_end":
        replyAudio = false;
        refreshState();
        break;
      case "interrupted":
        flushPlayback();
        asstBubble = null;
        pendingSources = null;
        replyAudio = false;
        thinking = false;
        toolEl.hidden = true;
        refreshState();
        break;
      case "error":
        showError(m.message || "Something went wrong.");
        if (/timed out|time out|idle/i.test(m.message || "")) {
          stopReconnect = true;
          showError((m.message || "") + " Press Talk to start again.");
        }
        pendingSources = null;
        asstBubble = null;
        toolEl.hidden = true;
        replyAudio = false;
        thinking = false;
        refreshState();
        break;
    }
  }

  // ---------- microphone ----------
  async function startMic() {
    showError("");
    if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showError("The microphone needs a secure connection (https or localhost). You can still type a message below.");
      return;
    }
    let c;
    try {
      c = ensureCtx(); // inside the click handler for autoplay rules
    } catch (e) {
      showError(e.message);
      return;
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
    } catch (e) {
      stream = null;
      if (e && (e.name === "NotAllowedError" || e.name === "SecurityError")) {
        showError("Microphone permission was denied. Allow it in the browser settings, or type a message below.");
      } else if (e && (e.name === "NotFoundError" || e.name === "OverconstrainedError")) {
        showError("No microphone was found. You can type a message below.");
      } else {
        showError("Could not open the microphone: " + ((e && e.message) || "unknown error"));
      }
      return;
    }
    try {
      await c.audioWorklet.addModule("/static/pcm-worklet.js");
      micSource = c.createMediaStreamSource(stream);
      micNode = new AudioWorkletNode(c, "pcm-encoder", { numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1, channelCountMode: "explicit" });
      micNode.port.onmessage = (ev) => {
        if (!(ev.data instanceof ArrayBuffer)) {
          hearing = !!(ev.data && ev.data.gate);
          refreshState();
          return;
        }
        if (wsReady && ws && ws.readyState === WebSocket.OPEN) ws.send(ev.data);
      };
      silent = c.createGain();
      silent.gain.value = 0;
      micSource.connect(micNode);
      micNode.connect(silent);
      silent.connect(c.destination);
    } catch (e) {
      stopMic();
      showError("Audio capture is not supported in this browser: " + ((e && e.message) || ""));
      return;
    }
    micOn = true;
    talkBtn.textContent = "End conversation";
    talkBtn.setAttribute("aria-pressed", "true");
    talkBtn.classList.add("active");
    connect();
    refreshState();
  }
  function stopMic() {
    micOn = false;
    hearing = false;
    if (micNode) { micNode.port.onmessage = null; try { micNode.disconnect(); } catch {} }
    if (micSource) { try { micSource.disconnect(); } catch {} }
    if (silent) { try { silent.disconnect(); } catch {} }
    if (stream) stream.getTracks().forEach((t) => t.stop());
    micNode = micSource = silent = stream = null;
    talkBtn.textContent = "Talk";
    talkBtn.setAttribute("aria-pressed", "false");
    talkBtn.classList.remove("active");
    interimEl.textContent = "";
    refreshState();
  }

  talkBtn.addEventListener("click", () => {
    if (micOn) {
      send({ type: "interrupt" });
      stopMic();
      flushPlayback();
      closeSocket();
      thinking = false;
      try { sessionStorage.removeItem(STORE_KEY); } catch {}
      refreshState();
    } else {
      startMic();
    }
  });
  stopBtn.addEventListener("click", () => {
    send({ type: "interrupt" });
    flushPlayback();
    thinking = false;
    refreshState();
  });
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    try { ensureCtx(); } catch {} // text still works without audio
    input.value = "";
    counter.hidden = true;
    addBubble("user", text);
    asstBubble = null;
    thinking = true;
    dropAudio = false;
    showError("");
    if (wsReady) send({ type: "text", text });
    else { pendingText.push(text); connect(); }
    refreshState();
  });

  input.addEventListener("input", () => {
    const left = 1000 - input.value.length;
    counter.hidden = left >= 100;
    counter.textContent = left + " characters left";
  });
  window.addEventListener("pagehide", () => { closeSocket(); stopMic(); });
  refreshState();
  return { destroy() { closeSocket(); stopMic(); } };
}
