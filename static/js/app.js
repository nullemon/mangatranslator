document.addEventListener("DOMContentLoaded", () => {

  /* ── DOM refs ── */
  const dropZone       = document.getElementById("dropZone");
  const fileInput      = document.getElementById("fileInput");
  const previewRow     = document.getElementById("previewRow");
  const previewImg     = document.getElementById("previewImg");
  const fileName       = document.getElementById("fileName");
  const fileSize       = document.getElementById("fileSize");
  const goBtn          = document.getElementById("goBtn");
  const clearBtn       = document.getElementById("clearBtn");
  const engineSelect   = document.getElementById("engine");
  const apiKeyInput    = document.getElementById("apiKey");
  const apiKeyLabel    = document.getElementById("apiKeyLabel");
  const targetLang     = document.getElementById("targetLang");
  const modelSelect    = document.getElementById("model");
  const smartMode      = document.getElementById("smartMode");
  const fontSelect     = document.getElementById("fontSelect");
  const fontUpload     = document.getElementById("fontUpload");
  const enhancePanel   = document.getElementById("enhancePanel");
  const enhanceProvider= document.getElementById("enhanceProvider");
  const enhanceKey     = document.getElementById("enhanceKey");
  const enhanceKeyLabel= document.getElementById("enhanceKeyLabel");
  const enhanceModel   = document.getElementById("enhanceModel");
  const enhancePrompt  = document.getElementById("enhancePrompt");
  const processingSec  = document.getElementById("processingSection");
  const compLabelLeft  = document.getElementById("compLabelLeft");
  const compLabelRight = document.getElementById("compLabelRight");

  const uploadSection     = document.getElementById("uploadSection");
  const processingSection = document.getElementById("processingSection");
  const resultSection     = document.getElementById("resultSection");
  const errorSection      = document.getElementById("errorSection");

  const progressFill = document.getElementById("progressFill");
  const progressMsg  = document.getElementById("progressMsg");

  const downloadBtn = document.getElementById("downloadBtn");
  const newBtn      = document.getElementById("newBtn");
  const retryBtn    = document.getElementById("retryBtn");
  const errorMsg    = document.getElementById("errorMsg");
  const applyBtn    = document.getElementById("applyBtn");

  let selectedFile = null;
  let currentTaskId = null;
  let pollTimer = null;
  let workflow = "scan-translate";
  let currentItems = [];          // editable translation items
  const excluded = new Set();     // ids the user rejected

  const ENHANCE_MODELS = { gemini: "gemini-2.5-flash-image", openai: "gpt-image-1" };

  const ENGINE_CONFIG = {
    claude: {
      label: "Claude API Key",
      placeholder: "sk-ant-...",
      storageKey: "manga_key_claude",
      models: [
        { value: "claude-sonnet-4-6", text: "Sonnet 4.6 (Fast)" },
        { value: "claude-opus-4-6",   text: "Opus 4.6 (Best)" },
        { value: "claude-haiku-4-5-20251001", text: "Haiku 4.5 (Cheap)" },
      ],
    },
    gemini: {
      label: "Gemini API Key",
      placeholder: "AIza...",
      storageKey: "manga_key_gemini",
      models: [
        { value: "gemini-2.5-flash", text: "Gemini 2.5 Flash (Fast)" },
        { value: "gemini-2.5-pro",   text: "Gemini 2.5 Pro (Best)" },
        { value: "gemini-2.0-flash", text: "Gemini 2.0 Flash (Lite)" },
      ],
    },
  };

  /* ══════════════════════════════════════════════════════════════════
     WORKFLOW PICKER
     ══════════════════════════════════════════════════════════════════ */
  const wfCards = document.querySelectorAll(".wf-card");

  function needsScan(wf)      { return wf === "raw-scan-translate" || wf === "raw-scan"; }
  function needsTranslate(wf) { return wf !== "raw-scan"; }

  function setWorkflow(wf) {
    workflow = wf;
    wfCards.forEach(c => c.classList.toggle("active", c.dataset.wf === wf));

    // Show/hide enhance panel
    enhancePanel.style.display = needsScan(wf) ? "" : "none";

    // Show/hide translation settings
    document.querySelector(".settings-bar").style.display = needsTranslate(wf) ? "" : "none";

    // Adapt Go button text
    const labels = {
      "scan-translate": "Translate",
      "raw-scan-translate": "Enhance & Translate",
      "raw-translate": "Translate Raw",
      "raw-scan": "Enhance to Scan",
    };
    goBtn.textContent = labels[wf] || "Go";

    localStorage.setItem("manga_workflow", wf);
  }

  wfCards.forEach(card => {
    card.addEventListener("click", () => setWorkflow(card.dataset.wf));
  });

  setWorkflow(localStorage.getItem("manga_workflow") || "scan-translate");

  /* ══════════════════════════════════════════════════════════════════
     ENGINE SWITCHING (Claude / Gemini for translation)
     ══════════════════════════════════════════════════════════════════ */
  function setEngine(eng) {
    const cfg = ENGINE_CONFIG[eng];
    if (!cfg) return;

    // Save current key before switching
    const prev = ENGINE_CONFIG[engineSelect.value];
    if (prev) localStorage.setItem(prev.storageKey, apiKeyInput.value);

    engineSelect.value = eng;
    apiKeyLabel.textContent = cfg.label;
    apiKeyInput.placeholder = cfg.placeholder;
    apiKeyInput.value = localStorage.getItem(cfg.storageKey) || "";

    // Populate model dropdown
    modelSelect.innerHTML = "";
    for (const m of cfg.models) {
      const opt = document.createElement("option");
      opt.value = m.value;
      opt.textContent = m.text;
      modelSelect.appendChild(opt);
    }

    localStorage.setItem("manga_engine", eng);
  }

  engineSelect.addEventListener("change", () => setEngine(engineSelect.value));
  apiKeyInput.addEventListener("change", () => {
    const cfg = ENGINE_CONFIG[engineSelect.value];
    if (cfg) localStorage.setItem(cfg.storageKey, apiKeyInput.value);
  });
  setEngine(localStorage.getItem("manga_engine") || "gemini");

  /* ══════════════════════════════════════════════════════════════════
     FONTS
     ══════════════════════════════════════════════════════════════════ */
  async function loadFonts() {
    try {
      const res = await fetch("/api/fonts");
      const data = await res.json();
      fontSelect.innerHTML = '<option value="">Auto-detect</option>';
      for (const f of data.fonts) {
        const opt = document.createElement("option");
        opt.value = f;
        opt.textContent = f.replace(/\.(ttf|otf)$/i, "");
        fontSelect.appendChild(opt);
      }
    } catch (_) {}
  }
  loadFonts();

  fontUpload.addEventListener("change", async () => {
    const file = fontUpload.files[0];
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch("/api/upload-font", { method: "POST", body: form });
      if (res.ok) {
        await loadFonts();
        fontSelect.value = file.name;
      }
    } catch (_) {}
    fontUpload.value = "";
  });

  /* ══════════════════════════════════════════════════════════════════
     ENHANCEMENT SETTINGS (persisted per-provider)
     ══════════════════════════════════════════════════════════════════ */
  function enhKeyName() { return "manga_enh_key_" + enhanceProvider.value; }

  async function initEnhance() {
    let defaultPrompt = "Convert this rough manga sketch into a clean, professional black-and-white manga scan with crisp inked line art, screentones, pure whites and deep blacks. Keep the exact same composition, panels, characters, and text.";
    try {
      const res = await fetch("/api/enhance-prompt");
      if (res.ok) {
        const data = await res.json();
        if (data.prompt) defaultPrompt = data.prompt;
        if (data.models) Object.assign(ENHANCE_MODELS, data.models);
      }
    } catch (_) {}

    enhanceProvider.value = localStorage.getItem("manga_enh_provider") || "gemini";
    enhancePrompt.value   = localStorage.getItem("manga_enh_prompt") || defaultPrompt;
    syncEnhanceFields();
  }

  function syncEnhanceFields() {
    const p = enhanceProvider.value;
    enhanceModel.placeholder = ENHANCE_MODELS[p] || "";
    enhanceModel.value = localStorage.getItem("manga_enh_model_" + p) || "";
    enhanceKey.value   = localStorage.getItem(enhKeyName()) || "";
    enhanceKeyLabel.textContent = p === "openai" ? "OpenAI API Key" : "Gemini API Key";
  }

  enhanceProvider.addEventListener("change", () => {
    localStorage.setItem("manga_enh_provider", enhanceProvider.value);
    syncEnhanceFields();
  });
  enhanceKey.addEventListener("change", () => localStorage.setItem(enhKeyName(), enhanceKey.value));
  enhanceModel.addEventListener("change", () => localStorage.setItem("manga_enh_model_" + enhanceProvider.value, enhanceModel.value));
  enhancePrompt.addEventListener("change", () => localStorage.setItem("manga_enh_prompt", enhancePrompt.value));
  initEnhance();

  /* ══════════════════════════════════════════════════════════════════
     DRAG & DROP / FILE SELECT
     ══════════════════════════════════════════════════════════════════ */
  dropZone.addEventListener("click", () => fileInput.click());

  dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", e => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });

  function handleFile(file) {
    if (!file.type.startsWith("image/")) return;
    selectedFile = file;
    previewImg.src = URL.createObjectURL(file);
    fileName.textContent = file.name;
    fileSize.textContent = formatBytes(file.size);
    dropZone.style.display = "none";
    previewRow.style.display = "flex";
  }

  clearBtn.addEventListener("click", () => {
    selectedFile = null;
    fileInput.value = "";
    previewRow.style.display = "none";
    dropZone.style.display = "";
  });

  /* ══════════════════════════════════════════════════════════════════
     GO BUTTON — dispatches based on workflow
     ══════════════════════════════════════════════════════════════════ */
  goBtn.addEventListener("click", startWorkflow);

  async function startWorkflow() {
    if (!selectedFile) return;

    const wantScan = needsScan(workflow);
    const wantTranslate = needsTranslate(workflow);

    // Validate keys
    if (wantTranslate) {
      const key = apiKeyInput.value.trim();
      if (!key) { apiKeyInput.focus(); apiKeyInput.style.borderColor = "#f87171"; return; }
      apiKeyInput.style.borderColor = "";
    }
    if (wantScan) {
      const key = enhanceKey.value.trim();
      if (!key) { enhanceKey.focus(); enhanceKey.style.borderColor = "#f87171"; return; }
      enhanceKey.style.borderColor = "";
    }

    showSection("processing");
    resetProgress();

    if (wantScan && wantTranslate) {
      // Raw → Scan → Translate
      const form = new FormData();
      form.append("file", selectedFile);
      form.append("api_key", apiKeyInput.value.trim());
      form.append("target_lang", targetLang.value);
      form.append("provider", engineSelect.value);
      form.append("model", modelSelect.value);
      form.append("smart_mode", smartMode.checked ? "true" : "false");
      form.append("font", fontSelect.value);
      form.append("enhance", "true");
      form.append("enhance_provider", enhanceProvider.value);
      form.append("enhance_key", enhanceKey.value.trim());
      form.append("enhance_prompt", enhancePrompt.value);
      form.append("enhance_model", enhanceModel.value);
      await submit("/api/translate", form);

    } else if (wantScan) {
      // Raw → Scan only
      const form = new FormData();
      form.append("file", selectedFile);
      form.append("provider", enhanceProvider.value);
      form.append("api_key", enhanceKey.value.trim());
      form.append("prompt", enhancePrompt.value);
      form.append("model", enhanceModel.value);
      await submit("/api/enhance", form);

    } else {
      // Scan → Translate or Raw → Translate (same endpoint, no enhance)
      const form = new FormData();
      form.append("file", selectedFile);
      form.append("api_key", apiKeyInput.value.trim());
      form.append("target_lang", targetLang.value);
      form.append("provider", engineSelect.value);
      form.append("model", modelSelect.value);
      form.append("smart_mode", smartMode.checked ? "true" : "false");
      form.append("font", fontSelect.value);
      form.append("enhance", "false");
      await submit("/api/translate", form);
    }
  }

  async function submit(url, form) {
    try {
      const res = await fetch(url, { method: "POST", body: form });
      if (!res.ok) {
        let msg = res.statusText;
        try { msg = (await res.json()).detail || msg; } catch (_) {}
        throw new Error(msg);
      }
      const data = await res.json();
      currentTaskId = data.task_id;
      pollStatus();
    } catch (e) {
      showError(e.message);
    }
  }

  function pollStatus() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(async () => {
      try {
        const res = await fetch(`/api/status/${currentTaskId}`);
        const data = await res.json();
        updateProgress(data);
        if (data.status === "done") {
          clearInterval(pollTimer);
          showResult(data);
        } else if (data.status === "error") {
          clearInterval(pollTimer);
          showError(data.message);
        }
      } catch (e) {
        clearInterval(pollTimer);
        showError("Lost connection to server");
      }
    }, 600);
  }

  /* ══════════════════════════════════════════════════════════════════
     PROGRESS
     ══════════════════════════════════════════════════════════════════ */
  function resetProgress() {
    progressFill.style.width = "0%";
    progressMsg.textContent = "Starting...";
    document.querySelectorAll(".step").forEach(s => s.classList.remove("active", "done"));
    document.querySelectorAll(".step-line").forEach(l => l.classList.remove("done"));
  }

  function updateProgress(data) {
    const pct = data.progress || 0;
    progressFill.style.width = pct + "%";
    progressMsg.textContent = data.message || "";
    progressMsg.classList.toggle("processing-pulse", pct > 0 && pct < 100);

    const currentStep = data.step || 0;
    document.querySelectorAll(".step").forEach(el => {
      const s = parseInt(el.dataset.step);
      el.classList.toggle("done", s < currentStep);
      el.classList.toggle("active", s === currentStep);
    });
    const lines = document.querySelectorAll(".step-line");
    lines.forEach((l, i) => l.classList.toggle("done", i + 1 < currentStep));
  }

  /* ══════════════════════════════════════════════════════════════════
     RESULT
     ══════════════════════════════════════════════════════════════════ */
  function showResult(data) {
    const origUrl  = data.original_url;
    const transUrl = data.output_url;

    document.getElementById("origImg").src = origUrl;
    document.getElementById("transImg").src = transUrl;
    document.getElementById("origFull").src = origUrl;
    document.getElementById("transFull").src = transUrl;

    const scanOnly = workflow === "raw-scan";
    compLabelLeft.textContent  = scanOnly ? "Rough" : "Original";
    compLabelRight.textContent = scanOnly ? "Manga Scan" : "Translated";
    document.querySelector('.tab[data-tab="translated"]').textContent = scanOnly ? "Scan" : "Translated";
    document.querySelector('.tab[data-tab="details"]').style.display = scanOnly ? "none" : "";

    excluded.clear();
    buildTranslationsList(data.result);
    showSection("result");
    initComparison();
  }

  function buildTranslationsList(result) {
    const el = document.getElementById("translationsList");
    el.innerHTML = "";

    // Prefer the rich `items` (with placement + ids); fall back to translations
    currentItems = (result && result.items) ? result.items.slice() : [];
    if (currentItems.length === 0 && result && result.translations) {
      currentItems = Object.entries(result.translations).map(([id, t]) => ({
        id: Number(id), original: t.original, translation: t.translation,
        type: t.type, placed: true,
      }));
    }

    if (currentItems.length === 0) {
      el.innerHTML = '<p style="color:var(--text-dim)">No text regions found</p>';
      return;
    }

    for (const it of currentItems) {
      const id = it.id;
      const isExcluded = excluded.has(String(id));
      const skipped = !it.placed && !isExcluded;  // detected but not put in a bubble

      const div = document.createElement("div");
      div.className = "tl-item" + (isExcluded ? " excluded" : "");
      div.dataset.id = id;

      let badge = "";
      if (isExcluded) badge = '<span class="tl-badge skip">skipped by you</span>';
      else if (skipped) badge = '<span class="tl-badge warn">not in a bubble</span>';
      else badge = '<span class="tl-badge ok">in bubble</span>';

      div.innerHTML = `
        <div class="tl-header">
          <span class="tl-id">#${id}</span>
          <span class="tl-type">${esc(it.type || "dialogue")}</span>
          ${badge}
          <button class="tl-x" title="Skip this bubble" data-id="${id}">✕</button>
        </div>
        <div class="tl-original">${esc(it.original || "")}</div>
        <textarea class="tl-edit" data-id="${id}" rows="2"
          ${isExcluded ? "disabled" : ""}>${esc(it.translation || "")}</textarea>
      `;
      el.appendChild(div);
    }

    // wire up X buttons
    el.querySelectorAll(".tl-x").forEach(btn => {
      btn.addEventListener("click", () => {
        const id = btn.dataset.id;
        if (excluded.has(id)) excluded.delete(id); else excluded.add(id);
        buildTranslationsList(collectResult());
      });
    });
  }

  // snapshot current edits back into an items structure for re-render
  function collectResult() {
    document.querySelectorAll(".tl-edit").forEach(t => {
      const it = currentItems.find(i => String(i.id) === t.dataset.id);
      if (it) it.translation = t.value;
    });
    return { items: currentItems };
  }

  async function applyChanges() {
    if (!currentTaskId) return;
    collectResult();
    const edits = {};
    currentItems.forEach(it => { edits[it.id] = it.translation; });

    applyBtn.disabled = true;
    applyBtn.textContent = "Re-rendering...";
    try {
      const res = await fetch(`/api/rerender/${currentTaskId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ excluded: [...excluded], edits }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const data = await res.json();
      refreshResultImages();
      buildTranslationsList({ items: data.items });
    } catch (e) {
      showError(e.message);
    } finally {
      applyBtn.disabled = false;
      applyBtn.textContent = "Apply & Re-render";
    }
  }

  function refreshResultImages() {
    const ts = Date.now();
    const url = `/api/result/${currentTaskId}?t=${ts}`;
    document.getElementById("transImg").src = url;
    document.getElementById("transFull").src = url;
  }

  /* ══════════════════════════════════════════════════════════════════
     COMPARISON SLIDER
     ══════════════════════════════════════════════════════════════════ */
  function initComparison() {
    const container = document.getElementById("comparisonContainer");
    const overlay   = document.getElementById("compOverlay");
    const slider    = document.getElementById("compSlider");
    let dragging = false;

    function setPosition(x) {
      const rect = container.getBoundingClientRect();
      let pct = ((x - rect.left) / rect.width) * 100;
      pct = Math.max(0, Math.min(100, pct));
      overlay.style.width = pct + "%";
      slider.style.left = pct + "%";
    }

    function onStart(e) {
      dragging = true;
      setPosition(e.touches ? e.touches[0].clientX : e.clientX);
    }
    function onMove(e) {
      if (!dragging) return;
      e.preventDefault();
      setPosition(e.touches ? e.touches[0].clientX : e.clientX);
    }
    function onEnd() { dragging = false; }

    container.addEventListener("mousedown", onStart);
    container.addEventListener("touchstart", onStart, { passive: true });
    document.addEventListener("mousemove", onMove);
    document.addEventListener("touchmove", onMove, { passive: false });
    document.addEventListener("mouseup", onEnd);
    document.addEventListener("touchend", onEnd);

    setPosition(container.getBoundingClientRect().left + container.getBoundingClientRect().width / 2);
  }

  /* ══════════════════════════════════════════════════════════════════
     TABS
     ══════════════════════════════════════════════════════════════════ */
  document.querySelectorAll(".tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
      tab.classList.add("active");
      document.getElementById("panel-" + tab.dataset.tab).classList.add("active");
    });
  });

  /* ══════════════════════════════════════════════════════════════════
     DOWNLOAD / NEW / RETRY
     ══════════════════════════════════════════════════════════════════ */
  downloadBtn.addEventListener("click", () => {
    if (!currentTaskId) return;
    const a = document.createElement("a");
    a.href = `/api/result/${currentTaskId}`;
    a.download = "translated_page.png";
    a.click();
  });

  applyBtn.addEventListener("click", applyChanges);
  newBtn.addEventListener("click", resetAll);
  retryBtn.addEventListener("click", () => {
    showSection("upload");
    if (selectedFile) { dropZone.style.display = "none"; previewRow.style.display = "flex"; }
  });

  function resetAll() {
    selectedFile = null;
    currentTaskId = null;
    fileInput.value = "";
    previewRow.style.display = "none";
    dropZone.style.display = "";
    showSection("upload");
  }

  /* ══════════════════════════════════════════════════════════════════
     SECTION TOGGLE / ERROR / HELPERS
     ══════════════════════════════════════════════════════════════════ */
  function showSection(name) {
    uploadSection.style.display     = name === "upload"     ? "" : "none";
    processingSection.style.display = name === "processing" ? "" : "none";
    resultSection.style.display     = name === "result"     ? "" : "none";
    errorSection.style.display      = name === "error"      ? "" : "none";
    const scanOnly = workflow === "raw-scan";
    processingSec.classList.toggle("enhance-mode", scanOnly);
  }

  function showError(msg) { errorMsg.textContent = msg; showSection("error"); }

  function formatBytes(b) {
    if (b < 1024) return b + " B";
    if (b < 1048576) return (b / 1024).toFixed(1) + " KB";
    return (b / 1048576).toFixed(1) + " MB";
  }
  function esc(s) { const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }

});
