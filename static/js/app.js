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
  const enhanceProvider= document.getElementById("enhanceProvider");
  const enhanceKey     = document.getElementById("enhanceKey");
  const enhanceKeyLabel= document.getElementById("enhanceKeyLabel");
  const enhanceModel   = document.getElementById("enhanceModel");
  const enhancePrompt  = document.getElementById("enhancePrompt");
  const enhancePanel   = document.getElementById("enhancePanel");

  const uploadSection  = document.getElementById("uploadSection");
  const resultSection  = document.getElementById("resultSection");
  const errorSection   = document.getElementById("errorSection");

  const batchBar       = document.getElementById("batchBar");
  const batchStatus    = document.getElementById("batchStatus");
  const batchProgressFill = document.getElementById("batchProgressFill");
  const pageStrip      = document.getElementById("pageStrip");
  const addPagesBtn    = document.getElementById("addPagesBtn");
  const zipBtn         = document.getElementById("zipBtn");

  const pageProcessing = document.getElementById("pageProcessing");
  const pageResult     = document.getElementById("pageResult");
  const stepsEl        = pageProcessing.querySelector(".steps");
  const progressBarWrap= document.getElementById("progressBarWrap");
  const progressFill   = document.getElementById("progressFill");
  const progressMsg    = document.getElementById("progressMsg");
  const retryPageBtn   = document.getElementById("retryPageBtn");

  const compLabelLeft  = document.getElementById("compLabelLeft");
  const compLabelRight = document.getElementById("compLabelRight");
  const origImg  = document.getElementById("origImg");
  const transImg = document.getElementById("transImg");
  const origFull = document.getElementById("origFull");
  const transFull= document.getElementById("transFull");
  const tabTranslated = document.querySelector('.tab[data-tab="translated"]');
  const detailsTab    = document.querySelector('.tab[data-tab="details"]');
  const fontScale     = document.getElementById("fontScale");
  const fontScaleVal  = document.getElementById("fontScaleVal");
  const moveToggle    = document.getElementById("moveToggle");
  const moveHint      = document.getElementById("moveHint");
  const moveApply     = document.getElementById("moveApply");
  const moveLayer     = document.getElementById("moveLayer");

  fontScale.addEventListener("input", () => {
    fontScaleVal.textContent = parseFloat(fontScale.value).toFixed(1) + "x";
  });

  const downloadBtn = document.getElementById("downloadBtn");
  const newBtn      = document.getElementById("newBtn");
  const retryBtn    = document.getElementById("retryBtn");
  const errorMsg    = document.getElementById("errorMsg");
  const applyBtn    = document.getElementById("applyBtn");

  const themeToggle      = document.getElementById("themeToggle");
  const apiKeyStatus     = document.getElementById("apiKeyStatus");
  const apiKeyReveal     = document.getElementById("apiKeyReveal");
  const enhanceKeyStatus = document.getElementById("enhanceKeyStatus");
  const enhanceKeyReveal = document.getElementById("enhanceKeyReveal");

  /* ── State ── */
  let workflow = "scan-translate";
  let pages = [];          // page objects
  let activeUid = null;
  let uidCounter = 0;
  let running = 0;
  // Process one page at a time. The local GPU stack (segmentation + manga-ocr
  // + LaMa) and the cloud enhancer both give consistent, full-quality results
  // only when a page has the resources to itself; running pages in parallel
  // caused GPU out-of-memory fallbacks and enhancer rate-limits, so some pages
  // came out translated/clean and others didn't. Sequential = uniform output.
  const MAX_CONCURRENT = 1;

  const ENHANCE_MODELS = { gemini: "gemini-2.5-flash-image", openai: "gpt-image-1" };
  const ENGINE_CONFIG = {
    claude: {
      label: "Claude API Key", placeholder: "sk-ant-...", storageKey: "manga_key_claude",
      models: [
        { value: "claude-sonnet-4-6", text: "Sonnet 4.6 (Fast)" },
        { value: "claude-opus-4-6",   text: "Opus 4.6 (Best)" },
        { value: "claude-haiku-4-5-20251001", text: "Haiku 4.5 (Cheap)" },
      ],
    },
    gemini: {
      label: "Gemini API Key", placeholder: "AIza...", storageKey: "manga_key_gemini",
      models: [
        { value: "gemini-2.5-flash", text: "Gemini 2.5 Flash (Fast)" },
        { value: "gemini-2.5-pro",   text: "Gemini 2.5 Pro (Best)" },
        { value: "gemini-2.0-flash", text: "Gemini 2.0 Flash (Lite)" },
      ],
    },
  };

  const getActive = () => pages.find(p => p.uid === activeUid) || null;
  const needsScan = wf => wf === "raw-scan-translate" || wf === "raw-scan";
  const needsTranslate = wf => wf !== "raw-scan";

  /* ══ THEME ══ */
  function applyTheme(t) {
    document.documentElement.setAttribute("data-theme", t);
    localStorage.setItem("manga_theme", t);
  }
  if (themeToggle) themeToggle.addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
    applyTheme(cur === "light" ? "dark" : "light");
  });

  /* ══ API KEY: persist + reveal + saved badge ══ */
  function showSaved(el, has) { if (el) el.classList.toggle("on", !!has); }
  function bindReveal(btn, input) {
    if (!btn || !input) return;
    btn.addEventListener("click", () => {
      input.type = input.type === "password" ? "text" : "password";
    });
  }
  bindReveal(apiKeyReveal, apiKeyInput);
  bindReveal(enhanceKeyReveal, enhanceKey);

  /* ══ WORKFLOW PICKER ══ */
  const wfCards = document.querySelectorAll(".wf-card");
  function setWorkflow(wf) {
    workflow = wf;
    wfCards.forEach(c => c.classList.toggle("active", c.dataset.wf === wf));
    enhancePanel.style.display = needsScan(wf) ? "" : "none";
    document.querySelector(".settings-bar").style.display = needsTranslate(wf) ? "" : "none";
    goBtn.textContent = {
      "scan-translate": "Translate", "raw-scan-translate": "Enhance & Translate",
      "raw-translate": "Translate Raw", "raw-scan": "Enhance to Scan",
    }[wf] || "Go";
    localStorage.setItem("manga_workflow", wf);
  }
  wfCards.forEach(card => card.addEventListener("click", () => setWorkflow(card.dataset.wf)));
  setWorkflow(localStorage.getItem("manga_workflow") || "scan-translate");

  /* ══ ENGINE SWITCHING ══ */
  function setEngine(eng) {
    const cfg = ENGINE_CONFIG[eng];
    if (!cfg) return;
    const prev = ENGINE_CONFIG[engineSelect.value];
    if (prev) localStorage.setItem(prev.storageKey, apiKeyInput.value);
    engineSelect.value = eng;
    apiKeyLabel.textContent = cfg.label;
    apiKeyInput.placeholder = cfg.placeholder;
    apiKeyInput.value = localStorage.getItem(cfg.storageKey) || "";
    showSaved(apiKeyStatus, apiKeyInput.value.trim());
    modelSelect.innerHTML = "";
    for (const m of cfg.models) {
      const opt = document.createElement("option");
      opt.value = m.value; opt.textContent = m.text;
      modelSelect.appendChild(opt);
    }
    localStorage.setItem("manga_engine", eng);
  }
  engineSelect.addEventListener("change", () => setEngine(engineSelect.value));
  apiKeyInput.addEventListener("input", () => {
    const cfg = ENGINE_CONFIG[engineSelect.value];
    if (cfg) localStorage.setItem(cfg.storageKey, apiKeyInput.value);
    showSaved(apiKeyStatus, apiKeyInput.value.trim());
  });
  setEngine(localStorage.getItem("manga_engine") || "gemini");

  /* ══ FONTS ══ */
  async function loadFonts() {
    try {
      const res = await fetch("/api/fonts");
      const data = await res.json();
      fontSelect.innerHTML = '<option value="">Auto-detect</option>';
      for (const f of data.fonts) {
        const opt = document.createElement("option");
        opt.value = f; opt.textContent = f.replace(/\.(ttf|otf)$/i, "");
        fontSelect.appendChild(opt);
      }
    } catch (_) {}
  }
  loadFonts();
  fontUpload.addEventListener("change", async () => {
    const files = [...fontUpload.files];
    if (!files.length) return;
    let last = "";
    for (const file of files) {
      const form = new FormData(); form.append("file", file);
      try {
        const res = await fetch("/api/upload-font", { method: "POST", body: form });
        if (res.ok) last = file.name;
      } catch (_) {}
    }
    await loadFonts();
    if (last) fontSelect.value = last;
    fontUpload.value = "";
  });

  /* ══ ENHANCEMENT SETTINGS ══ */
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
    showSaved(enhanceKeyStatus, enhanceKey.value.trim());
    enhanceKeyLabel.textContent = p === "openai" ? "OpenAI API Key" : "Gemini API Key";
  }
  enhanceProvider.addEventListener("change", () => {
    localStorage.setItem("manga_enh_provider", enhanceProvider.value);
    syncEnhanceFields();
  });
  enhanceKey.addEventListener("input", () => {
    localStorage.setItem(enhKeyName(), enhanceKey.value);
    showSaved(enhanceKeyStatus, enhanceKey.value.trim());
  });
  enhanceModel.addEventListener("change", () => localStorage.setItem("manga_enh_model_" + enhanceProvider.value, enhanceModel.value));
  enhancePrompt.addEventListener("change", () => localStorage.setItem("manga_enh_prompt", enhancePrompt.value));
  initEnhance();

  /* ══ FILE SELECTION ══ */
  dropZone.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", e => {
    e.preventDefault(); dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) addFiles(fileInput.files);
    fileInput.value = "";
  });

  function addFiles(fileList) {
    const incoming = [...fileList].filter(f => f.type.startsWith("image/"));
    if (!incoming.length) return;
    // natural sort by filename so chapter order is preserved
    incoming.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));

    const startedBatch = resultSection.style.display !== "none";
    for (const file of incoming) {
      pages.push({
        uid: ++uidCounter, file, name: file.name, size: file.size,
        thumb: URL.createObjectURL(file), taskId: null, status: "pending",
        progress: 0, step: 0, message: "", result: null, items: [],
        excluded: new Set(), offsets: {}, error: "", rev: 0,
      });
    }

    if (startedBatch) {
      // already running — enqueue & process the new pages
      pages.forEach(p => { if (p.status === "pending") p.status = "queued"; });
      renderStrip(); updateBatch(); pump();
    } else {
      showUploadPreview();
    }
  }

  function showUploadPreview() {
    if (!pages.length) return;
    previewImg.src = pages[0].thumb;
    if (pages.length === 1) {
      fileName.textContent = pages[0].name;
      fileSize.textContent = formatBytes(pages[0].size);
    } else {
      fileName.textContent = `${pages.length} pages selected`;
      fileSize.textContent = "Sorted by filename · " + pages.map(p => p.name).slice(0, 3).join(", ") + (pages.length > 3 ? "…" : "");
    }
    dropZone.style.display = "none";
    previewRow.style.display = "flex";
  }

  clearBtn.addEventListener("click", resetAll);

  /* ══ START / QUEUE ══ */
  goBtn.addEventListener("click", startBatch);

  function startBatch() {
    if (!pages.length) return;
    if (needsTranslate(workflow) && !apiKeyInput.value.trim()) {
      apiKeyInput.focus(); apiKeyInput.style.borderColor = "#f87171"; return;
    }
    apiKeyInput.style.borderColor = "";
    if (needsScan(workflow) && !enhanceKey.value.trim()) {
      enhancePanel.scrollIntoView({ behavior: "smooth" });
      enhanceKey.focus(); enhanceKey.style.borderColor = "#f87171"; return;
    }
    enhanceKey.style.borderColor = "";

    pages.forEach(p => { if (p.status === "pending") p.status = "queued"; });
    activeUid = pages[0].uid;
    showSection("result");
    renderStrip(); updateBatch(); renderActivePage();
    pump();
  }

  function pump() {
    while (running < MAX_CONCURRENT) {
      const next = pages.find(p => p.status === "queued");
      if (!next) break;
      next.status = "processing";
      running++;
      processPage(next).finally(() => { running--; pump(); });
    }
    renderStrip();
  }

  function buildRequest(file) {
    const f = new FormData();
    f.append("file", file);
    if (needsTranslate(workflow)) {
      f.append("api_key", apiKeyInput.value.trim());
      f.append("target_lang", targetLang.value);
      f.append("provider", engineSelect.value);
      f.append("model", modelSelect.value);
      f.append("smart_mode", smartMode.checked ? "true" : "false");
      f.append("font", fontSelect.value);
      f.append("enhance", needsScan(workflow) ? "true" : "false");
      if (needsScan(workflow)) {
        f.append("enhance_provider", enhanceProvider.value);
        f.append("enhance_key", enhanceKey.value.trim());
        f.append("enhance_prompt", enhancePrompt.value);
        f.append("enhance_model", enhanceModel.value);
      }
      return { url: "/api/translate", form: f };
    }
    f.append("provider", enhanceProvider.value);
    f.append("api_key", enhanceKey.value.trim());
    f.append("prompt", enhancePrompt.value);
    f.append("model", enhanceModel.value);
    return { url: "/api/enhance", form: f };
  }

  async function processPage(page) {
    page.error = ""; page.progress = 0; page.step = 0; page.message = "Queued";
    try {
      const { url, form } = buildRequest(page.file);
      const res = await fetch(url, { method: "POST", body: form });
      if (!res.ok) {
        let msg = res.statusText;
        try { msg = (await res.json()).detail || msg; } catch (_) {}
        throw new Error(msg);
      }
      page.taskId = (await res.json()).task_id;
      await pollPage(page);
    } catch (e) {
      page.status = "error"; page.error = e.message;
    } finally {
      renderStrip(); updateBatch();
      if (page.uid === activeUid) renderActivePage();
    }
  }

  function pollPage(page) {
    return new Promise(resolve => {
      const tick = async () => {
        try {
          const r = await fetch(`/api/status/${page.taskId}`);
          const d = await r.json();
          page.progress = d.progress || 0; page.step = d.step || 0; page.message = d.message || "";
          if (d.status === "done") {
            page.status = "done"; page.result = d;
            page.items = (d.result && d.result.items) ? d.result.items : [];
            page.excluded = new Set();
            page.rev++;
            renderStrip(); updateBatch();
            if (page.uid === activeUid) renderActivePage();
            return resolve();
          }
          if (d.status === "error") {
            page.status = "error"; page.error = d.message || "Failed";
            return resolve();
          }
          renderStrip();
          if (page.uid === activeUid) renderActivePage();
          setTimeout(tick, 600);
        } catch (e) {
          page.status = "error"; page.error = "Lost connection to server";
          resolve();
        }
      };
      tick();
    });
  }

  retryPageBtn.addEventListener("click", () => {
    const p = getActive();
    if (!p) return;
    p.status = "queued"; renderActivePage(); renderStrip(); updateBatch(); pump();
  });

  /* ══ PAGE STRIP ══ */
  function renderStrip() {
    const multi = pages.length > 1;
    batchBar.style.display = multi ? "" : "none";
    pageStrip.style.display = multi ? "" : "none";
    if (!multi) return;

    pageStrip.innerHTML = "";
    pages.forEach((p, i) => {
      const chip = document.createElement("div");
      chip.className = "pg-chip" + (p.uid === activeUid ? " active" : "");
      chip.dataset.uid = p.uid;
      const src = (p.status === "done" && p.taskId) ? `/api/result/${p.taskId}?t=${p.rev}` : p.thumb;
      chip.innerHTML = `
        <div class="pg-thumb"><img src="${src}" alt=""></div>
        <span class="pg-idx">${i + 1}</span>
        <span class="pg-dot ${p.status}"></span>
        <div class="pg-tools">
          <button data-act="left" title="Move left" ${i === 0 ? "disabled" : ""}>‹</button>
          <button data-act="right" title="Move right" ${i === pages.length - 1 ? "disabled" : ""}>›</button>
          <button data-act="remove" title="Remove">✕</button>
        </div>`;
      chip.querySelector(".pg-thumb").addEventListener("click", () => { activeUid = p.uid; renderStrip(); renderActivePage(); });
      chip.querySelector(".pg-idx").addEventListener("click", () => { activeUid = p.uid; renderStrip(); renderActivePage(); });
      chip.querySelectorAll(".pg-tools button").forEach(b => {
        b.addEventListener("click", e => { e.stopPropagation(); stripAction(p.uid, b.dataset.act); });
      });
      pageStrip.appendChild(chip);
    });
  }

  function stripAction(uid, act) {
    const i = pages.findIndex(p => p.uid === uid);
    if (i < 0) return;
    if (act === "left" && i > 0) { [pages[i - 1], pages[i]] = [pages[i], pages[i - 1]]; }
    else if (act === "right" && i < pages.length - 1) { [pages[i + 1], pages[i]] = [pages[i], pages[i + 1]]; }
    else if (act === "remove") {
      const wasActive = pages[i].uid === activeUid;
      pages.splice(i, 1);
      if (!pages.length) { resetAll(); return; }
      if (wasActive) activeUid = pages[Math.min(i, pages.length - 1)].uid;
    }
    renderStrip(); updateBatch(); renderActivePage();
  }

  function updateBatch() {
    const done = pages.filter(p => p.status === "done").length;
    const err = pages.filter(p => p.status === "error").length;
    batchStatus.textContent = `${done} / ${pages.length} done` + (err ? ` · ${err} failed` : "");
    batchProgressFill.style.width = (pages.length ? (done / pages.length * 100) : 0) + "%";
    zipBtn.disabled = done === 0;
  }

  addPagesBtn.addEventListener("click", () => fileInput.click());

  /* ══ ACTIVE PAGE RENDER ══ */
  function renderActivePage() {
    const p = getActive();
    if (!p) return;
    const scanOnly = workflow === "raw-scan";

    if (p.status === "done") {
      pageProcessing.style.display = "none";
      pageResult.style.display = "";
      const bust = `?t=${p.rev}`;
      origImg.src = `/api/original/${p.taskId}`;
      transImg.src = `/api/result/${p.taskId}${bust}`;
      origFull.src = origImg.src;
      transFull.src = transImg.src;
      compLabelLeft.textContent  = scanOnly ? "Rough" : "Original";
      compLabelRight.textContent = scanOnly ? "Manga Scan" : "Translated";
      tabTranslated.textContent  = scanOnly ? "Scan" : "Translated";
      detailsTab.style.display   = scanOnly ? "none" : "";
      buildTranslationsList(p);
      initComparison();
    } else if (p.status === "error") {
      pageResult.style.display = "none";
      pageProcessing.style.display = "";
      stepsEl.style.display = "none";
      progressBarWrap.style.display = "none";
      progressMsg.innerHTML = '<span style="color:#f87171">⚠ ' + esc(p.error || "Failed") + "</span>";
      retryPageBtn.style.display = "";
    } else {
      pageResult.style.display = "none";
      pageProcessing.style.display = "";
      stepsEl.style.display = scanOnly ? "none" : "";
      progressBarWrap.style.display = "";
      retryPageBtn.style.display = "none";
      updateSteps(p);
      progressFill.style.width = (p.progress || 0) + "%";
      progressMsg.textContent = p.message || "Starting...";
    }
  }

  function updateSteps(p) {
    const cur = p.step || 0;
    stepsEl.querySelectorAll(".step").forEach(el => {
      const s = parseInt(el.dataset.step);
      el.classList.toggle("done", s < cur);
      el.classList.toggle("active", s === cur);
    });
    stepsEl.querySelectorAll(".step-line").forEach((l, i) => l.classList.toggle("done", i + 1 < cur));
  }

  /* ══ DETAILS: edit / reject ══ */
  function buildTranslationsList(page) {
    const el = document.getElementById("translationsList");
    el.innerHTML = "";
    let items = page.items || [];
    if (items.length === 0 && page.result && page.result.result && page.result.result.translations) {
      items = Object.entries(page.result.result.translations).map(([id, t]) => ({
        id: Number(id), original: t.original, translation: t.translation, type: t.type, placed: true,
      }));
      page.items = items;
    }
    if (items.length === 0) {
      el.innerHTML = '<p style="color:var(--text-dim)">No text regions found</p>';
      return;
    }

    for (const it of items) {
      const isExcluded = page.excluded.has(String(it.id));
      const skipped = !it.placed && !isExcluded;
      const isBubble = it.in_bubble !== false;
      const div = document.createElement("div");
      div.className = "tl-item" + (isExcluded ? " excluded" : "") + (isBubble ? "" : " free-text");
      let badge;
      if (isExcluded) {
        badge = '<span class="tl-badge skip">skipped</span>';
      } else if (isBubble) {
        badge = '<span class="tl-badge ok">bubble</span>';
      } else {
        const label = it.type || "free text";
        badge = `<span class="tl-badge free">${esc(label)}</span>`;
      }
      const skipTitle = isBubble ? "Skip this bubble" : "Skip this text";
      div.innerHTML = `
        <div class="tl-header">
          <span class="tl-id">#${it.id}</span>
          <span class="tl-type">${esc(it.type || "dialogue")}</span>
          ${badge}
          <button class="tl-x" title="${skipTitle}" data-id="${it.id}">✕</button>
        </div>
        <div class="tl-original">${esc(it.original || "")}</div>
        <textarea class="tl-edit" data-id="${it.id}" rows="2" ${isExcluded ? "disabled" : ""}>${esc(it.translation || "")}</textarea>`;
      el.appendChild(div);
    }
    el.querySelectorAll(".tl-x").forEach(btn => {
      btn.addEventListener("click", () => {
        collectEdits(page);
        const id = btn.dataset.id;
        if (page.excluded.has(id)) page.excluded.delete(id); else page.excluded.add(id);
        buildTranslationsList(page);
      });
    });
  }

  function collectEdits(page) {
    document.querySelectorAll(".tl-edit").forEach(t => {
      const it = page.items.find(i => String(i.id) === t.dataset.id);
      if (it) it.translation = t.value;
    });
  }

  applyBtn.addEventListener("click", () => applyChanges());
  async function applyChanges(btn) {
    const page = getActive();
    if (!page || !page.taskId) return;
    collectEdits(page);
    const edits = {};
    page.items.forEach(it => { edits[it.id] = it.translation; });

    const useBtn = btn || applyBtn;
    const label = useBtn.textContent;
    useBtn.disabled = true; useBtn.textContent = "Re-rendering...";
    try {
      const res = await fetch(`/api/rerender/${page.taskId}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          excluded: [...page.excluded], edits,
          font_scale: parseFloat(fontScale.value),
          offsets: page.offsets || {},
        }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const data = await res.json();
      page.items = data.items; page.rev++;
      renderStrip();
      renderActivePage();
    } catch (e) {
      showError(e.message);
    } finally {
      useBtn.disabled = false; useBtn.textContent = label;
    }
  }

  /* ══ MOVE TEXT (drag to reposition) ══ */
  let moveMode = false;
  moveToggle.addEventListener("click", () => {
    moveMode = !moveMode;
    moveLayer.classList.toggle("on", moveMode);
    moveToggle.classList.toggle("btn-primary", moveMode);
    moveToggle.classList.toggle("btn-ghost", !moveMode);
    moveToggle.textContent = moveMode ? "Done moving" : "Move text";
    moveApply.style.display = moveMode ? "" : "none";
    moveHint.textContent = moveMode
      ? "Drag any box, then Apply & Re-render."
      : "Turn on to drag any translation into place.";
    if (moveMode) buildMoveBoxes();
  });
  moveApply.addEventListener("click", () => applyChanges(moveApply));

  function buildMoveBoxes() {
    const page = getActive();
    moveLayer.innerHTML = "";
    if (!page || !page.items) return;
    const W = transFull.naturalWidth, H = transFull.naturalHeight;
    if (!W || !H) return;
    page.offsets = page.offsets || {};

    for (const it of page.items) {
      if (!it.placed || !it.bbox) continue;
      if (page.excluded.has(String(it.id))) continue;
      const [bx, by, bw, bh] = it.bbox;
      const off = page.offsets[it.id] || [0, 0];
      const box = document.createElement("div");
      box.className = "move-box";
      box.style.left   = ((bx + off[0]) / W * 100) + "%";
      box.style.top    = ((by + off[1]) / H * 100) + "%";
      box.style.width  = (bw / W * 100) + "%";
      box.style.height = (bh / H * 100) + "%";
      box.innerHTML = `<span class="move-tag">#${it.id}</span>`;
      bindDrag(box, it, page, W, H);
      moveLayer.appendChild(box);
    }
  }

  function bindDrag(box, it, page, W, H) {
    let startX, startY, baseX, baseY, dragging = false;
    box.addEventListener("pointerdown", e => {
      e.preventDefault();
      dragging = true;
      box.classList.add("dragging");
      box.setPointerCapture(e.pointerId);
      startX = e.clientX; startY = e.clientY;
      const off = page.offsets[it.id] || [0, 0];
      baseX = off[0]; baseY = off[1];
    });
    box.addEventListener("pointermove", e => {
      if (!dragging) return;
      const rect = transFull.getBoundingClientRect();
      const sx = W / rect.width, sy = H / rect.height;   // screen→image px
      const dx = Math.round(baseX + (e.clientX - startX) * sx);
      const dy = Math.round(baseY + (e.clientY - startY) * sy);
      page.offsets[it.id] = [dx, dy];
      const [bx, by] = it.bbox;
      box.style.left = ((bx + dx) / W * 100) + "%";
      box.style.top  = ((by + dy) / H * 100) + "%";
    });
    const end = e => {
      if (!dragging) return;
      dragging = false;
      box.classList.remove("dragging");
      try { box.releasePointerCapture(e.pointerId); } catch (_) {}
    };
    box.addEventListener("pointerup", end);
    box.addEventListener("pointercancel", end);
  }

  /* ══ COMPARISON SLIDER ══ */
  let compBound = false;
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
    if (!compBound) {
      const onStart = e => { dragging = true; setPosition(e.touches ? e.touches[0].clientX : e.clientX); };
      const onMove  = e => { if (!dragging) return; e.preventDefault(); setPosition(e.touches ? e.touches[0].clientX : e.clientX); };
      const onEnd   = () => { dragging = false; };
      container.addEventListener("mousedown", onStart);
      container.addEventListener("touchstart", onStart, { passive: true });
      document.addEventListener("mousemove", onMove);
      document.addEventListener("touchmove", onMove, { passive: false });
      document.addEventListener("mouseup", onEnd);
      document.addEventListener("touchend", onEnd);
      compBound = true;
    }
    const rect = container.getBoundingClientRect();
    setPosition(rect.left + rect.width / 2);
  }

  /* ══ TABS ══ */
  document.querySelectorAll(".tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
      tab.classList.add("active");
      document.getElementById("panel-" + tab.dataset.tab).classList.add("active");
      if (tab.dataset.tab === "translated" && moveMode) buildMoveBoxes();
    });
  });
  // Rebuild drag boxes once the translated image has its real dimensions.
  transFull.addEventListener("load", () => { if (moveMode) buildMoveBoxes(); });

  /* ══ DOWNLOAD / ZIP / NEW ══ */
  downloadBtn.addEventListener("click", () => {
    const p = getActive();
    if (!p || p.status !== "done") return;
    const a = document.createElement("a");
    a.href = `/api/result/${p.taskId}?t=${p.rev}`;
    a.download = "translated_" + (p.name || "page.png").replace(/\.[^.]+$/, "") + ".png";
    a.click();
  });

  zipBtn.addEventListener("click", async () => {
    const ids = pages.filter(p => p.status === "done").map(p => p.taskId);
    if (!ids.length) return;
    zipBtn.disabled = true; zipBtn.textContent = "Zipping...";
    try {
      const res = await fetch("/api/zip", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task_ids: ids }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const blob = await res.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "translated_pages.zip";
      a.click();
      URL.revokeObjectURL(a.href);
    } catch (e) {
      showError(e.message);
    } finally {
      zipBtn.disabled = false; zipBtn.textContent = "Download All (ZIP)";
      updateBatch();
    }
  });

  newBtn.addEventListener("click", resetAll);
  retryBtn.addEventListener("click", () => showSection(pages.length ? "result" : "upload"));

  function resetAll() {
    pages.forEach(p => { try { URL.revokeObjectURL(p.thumb); } catch (_) {} });
    pages = []; activeUid = null; running = 0;
    fileInput.value = "";
    previewRow.style.display = "none";
    dropZone.style.display = "";
    showSection("upload");
  }

  /* ══ SECTIONS / HELPERS ══ */
  function showSection(name) {
    uploadSection.style.display = name === "upload" ? "" : "none";
    resultSection.style.display = name === "result" ? "" : "none";
    errorSection.style.display  = name === "error"  ? "" : "none";
  }
  function showError(msg) { errorMsg.textContent = msg; showSection("error"); }
  function formatBytes(b) {
    if (b < 1024) return b + " B";
    if (b < 1048576) return (b / 1024).toFixed(1) + " KB";
    return (b / 1048576).toFixed(1) + " MB";
  }
  function esc(s) { const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }

});
