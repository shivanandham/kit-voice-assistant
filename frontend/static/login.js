import { errorMessage } from "/static/api.js";

const form = document.getElementById("login-form");
const err = document.getElementById("login-error");
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  err.hidden = true;
  const btn = form.querySelector("button");
  btn.disabled = true;
  try {
    const r = await fetch("/api/login", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ login_id: form.login_id.value.trim(), password: form.password.value }),
    });
    if (!r.ok) throw new Error(await errorMessage(r));
    location.href = "/assistant";
  } catch (ex) {
    err.textContent = ex.message || "Login failed.";
    err.hidden = false;
    btn.disabled = false;
  }
});
