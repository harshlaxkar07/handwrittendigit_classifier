/* Handwritten Digit Classifier — drawing pad, upload and prediction display. */
(function () {
  "use strict";

  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  const esc = (v) =>
    String(v ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
    );

  const ICONS = {
    pen: '<path d="M12 19l7-7 3 3-7 7-3-3Z"/><path d="m18 13-1.5-7.5L2 2l3.5 14.5L13 18Z"/><path d="m2 2 7.586 7.586"/><circle cx="11" cy="11" r="2"/>',
    upload: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M7 10l5-5 5 5"/><path d="M12 5v12"/>',
    trash: '<path d="M3 6h18"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>',
    sparkles: '<path d="m12 3 1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9Z"/><path d="M19 15l.9 2.1L22 18l-2.1.9L19 21l-.9-2.1L16 18l2.1-.9Z"/>',
    download: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M7 10l5 5 5-5"/><path d="M12 15V3"/>',
    alert: '<path d="m21.7 18-8-14a2 2 0 0 0-3.4 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.7-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.9 4.9 1.4 1.4"/><path d="m17.7 17.7 1.4 1.4"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.3 17.7-1.4 1.4"/><path d="m19.1 4.9-1.4 1.4"/>',
    moon: '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',
    brain: '<path d="M12 5a3 3 0 1 0-5.9.8A3 3 0 0 0 4 12a3 3 0 0 0 2 2.8A3 3 0 0 0 12 19Z"/><path d="M12 5a3 3 0 1 1 5.9.8A3 3 0 0 1 20 12a3 3 0 0 1-2 2.8A3 3 0 0 1 12 19Z"/><path d="M12 5v14"/>',
    grid: '<rect width="7" height="7" x="3" y="3" rx="1"/><rect width="7" height="7" x="14" y="3" rx="1"/><rect width="7" height="7" x="14" y="14" rx="1"/><rect width="7" height="7" x="3" y="14" rx="1"/>',
  };

  const icon = (name, size) =>
    `<svg viewBox="0 0 24 24" width="${size || 24}" height="${size || 24}" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${
      ICONS[name] || ICONS.grid
    }</svg>`;

  const pct = (n) => (n === null || n === undefined || isNaN(n) ? "—" : (n * 100).toFixed(1) + "%");

  /* ---------------- theme ---------------- */
  function initTheme() {
    let stored = null;
    try { stored = localStorage.getItem("digit-theme"); } catch (_) { /* unavailable */ }
    const prefersLight = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches;
    apply(stored || (prefersLight ? "light" : "dark"));

    $("#themeToggle").addEventListener("click", () => {
      const next = document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light";
      try { localStorage.setItem("digit-theme", next); } catch (_) { /* unavailable */ }
      apply(next);
    });

    function apply(theme) {
      document.documentElement.setAttribute("data-theme", theme);
      $("#themeToggle").innerHTML = icon(theme === "light" ? "moon" : "sun", 18);
    }
  }

  /* ---------------- drawing pad ---------------- */
  const pad = $("#pad");
  const ctx = pad.getContext("2d", { willReadFrequently: true });
  let drawing = false;
  let inked = false;

  function sizePad() {
    const size = 280;
    pad.width = size;
    pad.height = size;
    clearPad();
  }

  function clearPad() {
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, pad.width, pad.height);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 22;
    inked = false;
    pad.classList.remove("inked");
  }

  function point(e) {
    const rect = pad.getBoundingClientRect();
    const source = e.touches && e.touches.length ? e.touches[0] : e;
    return {
      x: ((source.clientX - rect.left) / rect.width) * pad.width,
      y: ((source.clientY - rect.top) / rect.height) * pad.height,
    };
  }

  function start(e) {
    e.preventDefault();
    drawing = true;
    inked = true;
    pad.classList.add("inked");
    const p = point(e);
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
    ctx.lineTo(p.x + 0.1, p.y + 0.1);
    ctx.stroke();
  }

  function move(e) {
    if (!drawing) return;
    e.preventDefault();
    const p = point(e);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
  }

  function end() { drawing = false; }

  ["mousedown", "touchstart"].forEach((ev) => pad.addEventListener(ev, start, { passive: false }));
  ["mousemove", "touchmove"].forEach((ev) => pad.addEventListener(ev, move, { passive: false }));
  ["mouseup", "mouseleave", "touchend", "touchcancel"].forEach((ev) => pad.addEventListener(ev, end));

  /* ---------------- models ---------------- */
  let currentModel = "cnn";

  async function loadModels() {
    try {
      const data = await fetch("/models").then((r) => r.json());
      currentModel = data.default || currentModel;

      const host = $("#modelPicker");
      if (!data.models || data.models.length < 2) {
        host.classList.add("hidden");
        return;
      }

      host.innerHTML = data.models
        .map(
          (m) =>
            `<button class="${m.id === currentModel ? "active" : ""}" data-model="${esc(m.id)}" title="${esc(
              m.parameters.toLocaleString()
            )} parameters">${esc(m.label)}</button>`
        )
        .join("");

      $$("#modelPicker button").forEach((b) =>
        b.addEventListener("click", () => {
          currentModel = b.dataset.model;
          $$("#modelPicker button").forEach((x) => x.classList.toggle("active", x === b));
        })
      );
    } catch (_) {
      $("#modelPicker").classList.add("hidden");
    }
  }

  /* ---------------- prediction ---------------- */
  async function predict(body, btn) {
    if (btn) btn.classList.add("loading");
    showLoading();

    try {
      const response = await fetch("/predict", { method: "POST", body });
      const data = await response.json();

      if (!response.ok) {
        showError(data.error || "The image could not be classified.");
        return;
      }
      showResult(data);
    } catch (_) {
      showError("Cannot reach the server. Confirm the application is running.");
    } finally {
      if (btn) btn.classList.remove("loading");
    }
  }

  function predictFromPad(e) {
    if (!inked) {
      showError("Draw a digit on the pad first.");
      return;
    }
    const form = new FormData();
    form.append("image", pad.toDataURL("image/png"));
    form.append("model", currentModel);
    predict(form, e && e.currentTarget);
  }

  function predictFromFile(file) {
    const form = new FormData();
    form.append("file", file);
    form.append("model", currentModel);

    const reader = new FileReader();
    reader.onload = () => { $("#sourcePreview").innerHTML = `<img src="${reader.result}" alt="Uploaded image">`; };
    reader.readAsDataURL(file);

    predict(form);
  }

  /* ---------------- rendering ---------------- */
  function showLoading() {
    $("#result").innerHTML = `
      <div class="card-body">
        <div class="empty" style="padding:34px 20px">
          <div class="empty-icon">${icon("brain", 23)}</div>
          <h3>Classifying…</h3>
          <p>The image is being preprocessed and passed through the model.</p>
        </div>
      </div>`;
  }

  function showError(message) {
    $("#result").innerHTML = `
      <div class="card-body">
        <div class="alert">${icon("alert", 17)}<div>${esc(message)}</div></div>
        <p class="muted small" style="margin-top:12px">
          A clear, well-centred digit with good contrast against the background works best.
        </p>
      </div>`;
  }

  function showResult(data) {
    const probs = data.probabilities || [];
    const confidence = data.confidence || 0;
    const meterClass = confidence > 0.85 ? "ok" : confidence > 0.6 ? "" : "warn";

    $("#result").innerHTML = `
      <div class="card-body">
        <div class="verdict">
          <div class="verdict-digit">${esc(data.prediction)}</div>
          <div class="verdict-body">
            <div class="verdict-label">Predicted digit</div>
            <div class="verdict-conf">${pct(confidence)}</div>
            <div class="verdict-model">confidence · ${esc(
              data.model === "cnn" ? "convolutional network" : "linear model"
            )}</div>
            <div class="meter ${meterClass}"><span style="width:${confidence * 100}%"></span></div>
          </div>
        </div>

        ${
          probs.length
            ? `<h3 style="font-size:.73rem;text-transform:uppercase;letter-spacing:.08em;color:var(--text-dim);margin:22px 0 11px">
                 Probability across all ten digits
               </h3>
               <div class="probs">${probs
                 .map(
                   (p, d) => `
                 <div class="prob-row ${d === data.prediction ? "top" : ""}">
                   <div class="prob-digit">${d}</div>
                   <div class="prob-bar"><div class="prob-fill" style="width:${Math.max(p * 100, 0.6)}%"></div></div>
                   <div class="prob-pct">${(p * 100).toFixed(1)}%</div>
                 </div>`
                 )
                 .join("")}</div>`
            : ""
        }

        ${
          data.preprocessed
            ? `<h3 style="font-size:.73rem;text-transform:uppercase;letter-spacing:.08em;color:var(--text-dim);margin:22px 0 11px">
                 What the model actually saw
               </h3>
               <div class="preview">
                 <img src="data:image/png;base64,${data.preprocessed}" alt="Preprocessed 28 by 28 image">
                 <div class="preview-body">
                   The digit is isolated, cropped to its bounding box, scaled to fit a 20&times;20 area
                   and padded out to 28&times;28 — exactly the shape the model was trained on.
                   <div style="margin-top:10px">
                     <a class="btn" href="/download_preprocessed">${icon("download", 15)} Download</a>
                   </div>
                 </div>
               </div>`
            : ""
        }
      </div>`;
  }

  /* ---------------- upload wiring ---------------- */
  function setupDropzone() {
    const zone = $("#dropzone");
    const input = $("#fileInput");

    zone.addEventListener("click", (e) => { if (e.target !== input) input.click(); });
    zone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
    });
    input.addEventListener("change", () => {
      if (input.files[0]) predictFromFile(input.files[0]);
      input.value = "";
    });

    ["dragenter", "dragover"].forEach((ev) =>
      zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("over"); })
    );
    ["dragleave", "drop"].forEach((ev) =>
      zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove("over"); })
    );
    zone.addEventListener("drop", (e) => {
      const file = e.dataTransfer.files[0];
      if (file) predictFromFile(file);
    });
  }

  /* ---------------- boot ---------------- */
  function init() {
    $$("[data-icon]").forEach((n) => {
      n.innerHTML = icon(n.dataset.icon, n.dataset.size || 24);
    });

    initTheme();
    sizePad();
    setupDropzone();
    loadModels();

    $("#clearPad").addEventListener("click", () => {
      clearPad();
      showEmpty();
    });
    $("#predictPad").addEventListener("click", predictFromPad);

    $$("[data-tab]").forEach((b) =>
      b.addEventListener("click", () => {
        const tab = b.dataset.tab;
        $$("[data-tab]").forEach((x) => x.classList.toggle("active", x === b));
        $("#drawPanel").classList.toggle("hidden", tab !== "draw");
        $("#uploadPanel").classList.toggle("hidden", tab !== "upload");
      })
    );

    showEmpty();
  }

  function showEmpty() {
    $("#result").innerHTML = $("#emptyTemplate").innerHTML;
    $$("[data-icon]", $("#result")).forEach((n) => {
      n.innerHTML = icon(n.dataset.icon, n.dataset.size || 24);
    });
  }

  document.addEventListener("DOMContentLoaded", init);
})();
