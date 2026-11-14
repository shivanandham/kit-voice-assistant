import { getJSON, el } from "/static/api.js";
import { createVoicePanel } from "/static/voice.js";

createVoicePanel(document.getElementById("voice"));
const rupees = (n) => (n == null ? "-" : "₹" + Number(n).toLocaleString("en-IN"));

getJSON("/api/public/college").then((c) => {
  document.getElementById("college-name").textContent = c.name || "Knotopian Institute of Technology";
  document.getElementById("college-tagline").textContent = c.tagline || "";
  document.getElementById("college-desc").textContent = c.description || "";
  const k = c.contact || {};
  const parts = [k.phone && "Phone: " + k.phone, k.email && "Email: " + k.email, k.website && "Web: " + k.website, k.address && "Address: " + k.address].filter(Boolean);
  document.getElementById("college-contact").textContent = parts.join("  |  ");
}).catch(() => { document.getElementById("college-desc").textContent = "College details are unavailable right now."; });

getJSON("/api/public/courses").then((rows) => {
  const msg = document.getElementById("courses-msg");
  if (!rows.length) { msg.textContent = "No courses listed."; return; }
  const tb = document.querySelector("#courses tbody");
  for (const r of rows) {
    const tr = document.createElement("tr");
    const cells = [[r.code], [r.name], [r.department], [r.min_12th_percent != null ? r.min_12th_percent + "%" : "-", 1], [rupees(r.annual_fee), 1], [rupees(r.hostel_fee), 1], [r.seats ?? "-", 1]];
    for (const [v, num] of cells) tr.append(el("td", num ? "num" : "", String(v ?? "")));
    tb.append(tr);
  }
  msg.hidden = true;
  document.getElementById("courses").hidden = false;
}).catch(() => { document.getElementById("courses-msg").textContent = "Courses are unavailable right now."; });
