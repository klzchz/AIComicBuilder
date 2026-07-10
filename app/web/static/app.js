/* AI Comic Builder — client helpers (port of src/stores/* client logic).
 *
 * Model providers + default model selection live in localStorage (the source
 * app used a zustand `persist` store the same way); every generate call sends
 * the resolved modelConfig to POST /api/projects/{id}/generate, matching the
 * source API contract: { action, payload, modelConfig }.
 */
(function () {
  const STORE_KEY = "aicb-model-store";

  // ── model store ──────────────────────────────────────────────────────────
  function store() {
    try {
      return JSON.parse(localStorage.getItem(STORE_KEY)) || defaults();
    } catch {
      return defaults();
    }
  }
  function defaults() {
    return { providers: [], defaultTextModel: null, defaultImageModel: null, defaultVideoModel: null };
  }
  function saveStore(s) {
    localStorage.setItem(STORE_KEY, JSON.stringify(s));
  }
  function resolveRef(s, ref) {
    if (!ref) return null;
    const p = (s.providers || []).find((x) => x.id === ref.providerId);
    if (!p) return null;
    return {
      protocol: p.protocol,
      baseUrl: p.baseUrl || "",
      apiKey: p.apiKey || "",
      ...(p.secretKey ? { secretKey: p.secretKey } : {}),
      modelId: ref.modelId,
    };
  }
  function modelConfig() {
    const s = store();
    return {
      text: resolveRef(s, s.defaultTextModel),
      image: resolveRef(s, s.defaultImageModel),
      video: resolveRef(s, s.defaultVideoModel),
    };
  }

  // ── toasts ───────────────────────────────────────────────────────────────
  function toast(msg, type) {
    const box = document.getElementById("toasts");
    if (!box) return alert(msg);
    const el = document.createElement("div");
    el.className =
      "rounded-lg border px-3 py-2 text-sm shadow-lg " +
      (type === "error"
        ? "border-red-800 bg-red-950 text-red-200"
        : "border-emerald-800 bg-emerald-950 text-emerald-200");
    el.textContent = msg;
    box.appendChild(el);
    setTimeout(() => el.remove(), type === "error" ? 8000 : 4000);
  }

  // ── fetch helpers ────────────────────────────────────────────────────────
  async function api(url, opts = {}) {
    const res = await fetch(url, {
      headers: opts.body instanceof FormData ? {} : { "Content-Type": "application/json" },
      ...opts,
      body: opts.body instanceof FormData ? opts.body : opts.body ? JSON.stringify(opts.body) : undefined,
    });
    const text = await res.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch { data = { raw: text }; }
    if (!res.ok) {
      const msg = (data && (data.error || data.detail)) || res.status + " " + res.statusText;
      throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    }
    return data;
  }

  /**
   * POST /api/projects/{id}/generate. Handles both plain-JSON responses and
   * streamed text responses (script outline / generate / parse stream text).
   * opts.onText(chunk) receives streamed text if the response is not JSON.
   */
  async function generate(projectId, action, payload = {}, opts = {}) {
    const res = await fetch(`/api/projects/${projectId}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, payload, modelConfig: modelConfig() }),
    });
    const ctype = res.headers.get("content-type") || "";
    if (!res.ok) {
      let msg = res.status + " " + res.statusText;
      try {
        const err = await res.json();
        msg = err.error || err.detail || msg;
      } catch {}
      throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    }
    if (ctype.includes("application/json")) {
      return res.json();
    }
    // streamed text
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let full = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = dec.decode(value, { stream: true });
      full += chunk;
      if (opts.onText) opts.onText(chunk, full);
    }
    return { text: full };
  }

  /** Run a button-bound async action with busy state + error toast. */
  async function run(btn, fn, busyLabel) {
    const original = btn ? btn.innerHTML : null;
    if (btn) {
      btn.disabled = true;
      btn.innerHTML = '<span class="spin">&#9696;</span> ' + (busyLabel || "Working...");
    }
    try {
      await fn();
    } catch (e) {
      console.error(e);
      toast(e.message || "Request failed", "error");
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = original;
      }
    }
  }

  /** Kick HTMX status polling (partials poll while body[data-polling="1"]). */
  function startPolling() {
    document.body.dataset.polling = "1";
    document.querySelectorAll("[data-poll]").forEach((el) => window.htmx && htmx.trigger(el, "poll"));
  }
  function stopPolling() {
    document.body.dataset.polling = "0";
  }

  function requireModel(kind) {
    const cfg = modelConfig();
    if (!cfg[kind]) {
      toast(
        `Please configure a ${kind} model first (Settings → ${kind} models).`,
        "error"
      );
      window.setTimeout(() => (window.location.href = "/settings"), 1200);
      return false;
    }
    return true;
  }

  window.AICB = { store, saveStore, modelConfig, api, generate, toast, run, startPolling, stopPolling, requireModel };
})();
