export async function getJSON(url) {
  const r = await fetch(url, { credentials: "same-origin" });
  if (!r.ok) throw new Error(await errorMessage(r));
  return r.json();
}

export async function errorMessage(r) {
  try {
    const j = await r.json();
    return (j.error && j.error.message) || j.message || (typeof j.detail === "string" ? j.detail : "") || `Request failed (${r.status})`;
  } catch {
    return `Request failed (${r.status})`;
  }
}

export function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

export function sourceText(s) {
  if (!s) return "";
  if (s.kind === "college_record") return s.label || "College system";
  const parts = [];
  if (s.document_title) parts.push(s.document_title + (s.version ? " (" + s.version + ")" : ""));
  if (s.section) parts.push("rule " + s.section);
  else if (s.heading) parts.push(s.heading);
  if (s.page_start != null) {
    parts.push(s.page_end != null && s.page_end !== s.page_start ? "pages " + s.page_start + "\u2013" + s.page_end : "page " + s.page_start);
  }
  return parts.join(", ");
}

// Append a compact "Sources" list to a message element.
export function renderSources(container, sources) {
  if (!Array.isArray(sources) || !sources.length) return;
  const box = el("div", "sources");
  box.append(el("span", "sources-h", "Sources"));
  const ul = el("ul");
  for (const s of sources) {
    const t = sourceText(s);
    if (t) ul.append(el("li", "", t));
  }
  box.append(ul);
  container.append(box);
}

export function fmtDate(s) {
  const d = new Date(s);
  return isNaN(d) ? "" : d.toLocaleString();
}
