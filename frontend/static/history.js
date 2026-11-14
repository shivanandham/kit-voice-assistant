import { getJSON, el, fmtDate, renderSources } from "/static/api.js";

const msg = document.getElementById("hist-msg");
const wrap = document.getElementById("hist");
const list = document.getElementById("hist-list");
const tr = document.getElementById("tr");
const trTitle = document.getElementById("tr-title");

async function open(id, btn) {
  list.querySelectorAll("button").forEach((b) => b.removeAttribute("aria-current"));
  btn.setAttribute("aria-current", "true");
  tr.textContent = "";
  trTitle.textContent = "Loading…";
  try {
    const c = await getJSON("/api/history/" + encodeURIComponent(id));
    trTitle.textContent = c.title || "Untitled conversation";
    if (!c.messages.length) tr.append(el("p", "muted", "No messages."));
    for (const m of c.messages) {
      const b = el("div", "msg msg-" + (m.role === "user" ? "user" : "assistant"));
      b.append(el("span", "who", (m.role === "user" ? "You" : "Assistant") + " - " + fmtDate(m.created_at)), el("span", "txt", m.content));
      if (m.role !== "user") renderSources(b, m.citations);
      tr.append(b);
    }
  } catch (e) {
    trTitle.textContent = "Could not load this conversation.";
  }
}

getJSON("/api/history").then((items) => {
  if (!items.length) { msg.textContent = "No conversations yet. Talk to the assistant and they will appear here."; return; }
  msg.hidden = true;
  wrap.hidden = false;
  for (const it of items) {
    const li = document.createElement("li");
    const b = el("button");
    b.type = "button";
    b.append(el("span", "t", it.title || "Untitled conversation"), el("span", "s", `${fmtDate(it.updated_at)} - ${it.message_count} messages`));
    b.addEventListener("click", () => open(it.id, b));
    li.append(b);
    list.append(li);
  }
}).catch(() => { msg.textContent = "Could not load history. Please try again."; });
