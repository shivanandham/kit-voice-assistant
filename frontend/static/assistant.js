import { getJSON } from "/static/api.js";
import { createVoicePanel } from "/static/voice.js";

(async () => {
  try {
    const me = await getJSON("/api/me");
    if (me.role === "visitor" || !me.user) { location.href = "/login"; return; }
    const u = me.user;
    document.getElementById("who").textContent = `Logged in as ${u.full_name || u.login_id} (${me.role})`;
  } catch {
    location.href = "/login";
    return;
  }
  createVoicePanel(document.getElementById("voice"));
})();

document.getElementById("logout").addEventListener("click", async () => {
  try { await fetch("/api/logout", { method: "POST", credentials: "same-origin" }); } catch {}
  try { sessionStorage.removeItem("kit_conversation_id"); } catch {}
  location.href = "/";
});
