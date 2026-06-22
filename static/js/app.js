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
  const sourceLang     = document.getElementById("sourceLang");
  const modelSelect    = document.getElementById("model");
  const smartMode      = document.getElementById("smartMode");
  const translateSfx   = document.getElementById("translateSfx");
  const maxQuality     = document.getElementById("maxQuality");
  const compressOut    = document.getElementById("compressOut");
  const transStyle     = document.getElementById("transStyle");
  const removeWatermark= document.getElementById("removeWatermark");
  const replaceWatermark=document.getElementById("replaceWatermark");
  const hdUpscale      = document.getElementById("hdUpscale");
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
  const editHint      = document.getElementById("editHint");
  const editApply     = document.getElementById("editApply");
  const moveLayer     = document.getElementById("moveLayer");
  const toolBtns      = document.querySelectorAll(".tool-btn");
  const watermarkInput = document.getElementById("watermark");
  const creditInput = document.getElementById("credit");

  if (watermarkInput) {
    watermarkInput.value = localStorage.getItem("manga_watermark") || "";
    watermarkInput.addEventListener("input", () =>
      localStorage.setItem("manga_watermark", watermarkInput.value));
  }
  if (creditInput) {
    creditInput.value = localStorage.getItem("manga_credit") || "";
    creditInput.addEventListener("input", () =>
      localStorage.setItem("manga_credit", creditInput.value));
  }

  const stylePrompt = document.getElementById("stylePrompt");
  if (stylePrompt) {
    stylePrompt.value = localStorage.getItem("manga_style_prompt") || "";
    stylePrompt.addEventListener("input", () =>
      localStorage.setItem("manga_style_prompt", stylePrompt.value));
  }

  /* ══ SETTINGS PERSISTENCE: every control restores on reload ══ */
  const textCase = document.getElementById("textCase");
  const resetSettings = document.getElementById("resetSettings");

  targetLang.value = localStorage.getItem("manga_lang") || "English";
  targetLang.addEventListener("change", () =>
    localStorage.setItem("manga_lang", targetLang.value));

  if (sourceLang) {
    sourceLang.value = localStorage.getItem("manga_source_lang") || "Japanese";
    sourceLang.addEventListener("change", () =>
      localStorage.setItem("manga_source_lang", sourceLang.value));
  }

  smartMode.checked = localStorage.getItem("manga_smart") === "1";
  smartMode.addEventListener("change", () =>
    localStorage.setItem("manga_smart", smartMode.checked ? "1" : "0"));

  if (translateSfx) {
    translateSfx.checked = localStorage.getItem("manga_translate_sfx") === "1";
    translateSfx.addEventListener("change", () =>
      localStorage.setItem("manga_translate_sfx", translateSfx.checked ? "1" : "0"));
  }

  if (maxQuality) {
    maxQuality.checked = localStorage.getItem("manga_max_quality") === "1";
    maxQuality.addEventListener("change", () =>
      localStorage.setItem("manga_max_quality", maxQuality.checked ? "1" : "0"));
  }


  if (compressOut) {
    compressOut.checked = localStorage.getItem("manga_compress_out") === "1";
    compressOut.addEventListener("change", () =>
      localStorage.setItem("manga_compress_out", compressOut.checked ? "1" : "0"));
  }

  if (removeWatermark) {
    const saved = localStorage.getItem("manga_remove_watermark");
    removeWatermark.checked = saved === null ? true : saved === "1";  // default ON
    removeWatermark.addEventListener("change", () =>
      localStorage.setItem("manga_remove_watermark", removeWatermark.checked ? "1" : "0"));
  }

  if (replaceWatermark) {
    replaceWatermark.checked = localStorage.getItem("manga_replace_watermark") === "1";
    replaceWatermark.addEventListener("change", () =>
      localStorage.setItem("manga_replace_watermark", replaceWatermark.checked ? "1" : "0"));
  }

  const checkSystem = document.getElementById("checkSystem");
  const systemStatus = document.getElementById("systemStatus");
  if (checkSystem && systemStatus) {
    checkSystem.addEventListener("click", async () => {
      systemStatus.textContent = "checking…";
      try {
        const h = await fetch("/api/health?refresh=true").then(r => r.json());
        const on = v => v ? "✓" : "✗";
        systemStatus.innerHTML =
          `<b>build ${h.server_commit || "?"}</b> · ` +
          `GPU:${on(h.cuda)}${h.gpu ? " (" + h.gpu + ")" : ""} · ` +
          `seg:${on(h.balloon_seg_yolo)} · ocr:${on(h.manga_ocr)} · ` +
          `lama:${on(h.lama_inpaint)} · text-seg:${on(h.weights && h.weights.comic_text_detector)} · ` +
          `RTL:${on(h.raqm_rtl_shaping || h.arabic_reshaper_fallback)} · ` +
          `arabicFont:${on(h.arabic_font)} · ` +
          `<b>full stack:${on(h.ready_full_stack)}</b>`;
      } catch (e) {
        systemStatus.textContent = "status check failed: " + e;
      }
    });
  }

  if (hdUpscale) {
    hdUpscale.checked = localStorage.getItem("manga_hd_upscale") === "1";
    hdUpscale.addEventListener("change", () =>
      localStorage.setItem("manga_hd_upscale", hdUpscale.checked ? "1" : "0"));
  }

  if (textCase) {
    textCase.value = localStorage.getItem("manga_case") || "upper";
    textCase.addEventListener("change", () =>
      localStorage.setItem("manga_case", textCase.value));
  }

  const pageFinish = document.getElementById("pageFinish");
  if (pageFinish) {
    // Default to the local clean scan — it keeps the original art exactly as
    // drawn. The old "api" finish repainted the whole page with a generative
    // model (changed art); it's gone, so map any saved "api" back to "clean".
    let savedFinish = localStorage.getItem("manga_finish") || "clean";
    if (savedFinish === "api") savedFinish = "clean";
    pageFinish.value = savedFinish;
    pageFinish.addEventListener("change", () => {
      localStorage.setItem("manga_finish", pageFinish.value);
      enhancePanel.style.display = needsEnhancePanel() ? "" : "none";
    });
  }

  fontSelect.addEventListener("change", () =>
    localStorage.setItem("manga_font", fontSelect.value));

  modelSelect.addEventListener("change", () =>
    localStorage.setItem("manga_model_" + engineSelect.value, modelSelect.value));

  if (resetSettings) resetSettings.addEventListener("click", () => {
    Object.keys(localStorage)
      .filter(k => k.startsWith("manga_")
        && !k.startsWith("manga_key_") && !k.startsWith("manga_enh_key_"))
      .forEach(k => localStorage.removeItem(k));
    location.reload();
  });

  fontScale.value = localStorage.getItem("manga_font_scale") || fontScale.value;
  fontScaleVal.textContent = parseFloat(fontScale.value).toFixed(1) + "x";
  fontScale.addEventListener("input", () => {
    fontScaleVal.textContent = parseFloat(fontScale.value).toFixed(1) + "x";
    localStorage.setItem("manga_font_scale", fontScale.value);
  });

  const downloadBtn = document.getElementById("downloadBtn");
  const translateScanBtn = document.getElementById("translateScanBtn");
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
  const needsScan = wf => wf === "raw-scan-translate" || wf === "raw-scan" || wf === "scan-upscale";
  const isClean = wf => wf === "clean";
  const needsTranslate = wf => wf !== "raw-scan" && wf !== "upscale-only" && wf !== "scan-upscale" && wf !== "clean";
  const isUpscaleOnly = wf => wf === "upscale-only";

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
  function needsEnhancePanel() {
    return needsScan(workflow);
  }
  function setWorkflow(wf) {
    workflow = wf;
    wfCards.forEach(c => c.classList.toggle("active", c.dataset.wf === wf));
    enhancePanel.style.display = needsEnhancePanel() ? "" : "none";
    // Show the settings panel for translate AND clean (Clean uses HD Upscale,
    // Page Finish, Isolate Page, Compress Output) — only the pure enhance/
    // upscale workflows hide it.
    document.querySelector(".settings-bar").style.display =
      (needsTranslate(wf) || isClean(wf)) ? "" : "none";
    goBtn.textContent = {
      "scan-translate": "Translate", "raw-scan-translate": "Enhance & Translate",
      "raw-translate": "Translate Raw", "raw-scan": "Enhance to Scan",
      "upscale-only": "Upscale to HD", "scan-upscale": "Enhance + Upscale",
    }[wf] || "Go";
    localStorage.setItem("manga_workflow", wf);
  }
  wfCards.forEach(card => card.addEventListener("click", () => setWorkflow(card.dataset.wf)));
  setWorkflow(localStorage.getItem("manga_workflow") || "scan-translate");

  /* ══ ENGINE SWITCHING ══ */
  let engineInited = false;
  function setEngine(eng) {
    const cfg = ENGINE_CONFIG[eng];
    if (!cfg) return;
    if (engineInited) {
      const prev = ENGINE_CONFIG[engineSelect.value];
      if (prev) localStorage.setItem(prev.storageKey, apiKeyInput.value);
    }
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
    const savedModel = localStorage.getItem("manga_model_" + eng);
    if (savedModel && cfg.models.some(m => m.value === savedModel)) {
      modelSelect.value = savedModel;
    }
    localStorage.setItem("manga_engine", eng);
    engineInited = true;
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
      const savedFont = localStorage.getItem("manga_font") || "";
      if (savedFont && [...fontSelect.options].some(o => o.value === savedFont)) {
        fontSelect.value = savedFont;
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
    if (last) {
      fontSelect.value = last;
      localStorage.setItem("manga_font", last);
    }
    fontUpload.value = "";
  });

  /* ══ ENHANCEMENT SETTINGS ══ */
  function enhKeyName() { return "manga_enh_key_" + enhanceProvider.value; }
  async function initEnhance() {
    let defaultPrompt = "Restore this photographed manga page into a clean flatbed-scanned black-and-white page: flatten all curvature/warp, remove creases, folds, shadows and uneven lighting, de-skew and straighten, square up the borders and crop out the surface around the page; pure white paper, solid black ink, crisp sharp lines. Preserve every panel, character, screentone and all Japanese text exactly — only fix geometry, lighting and tone, never the artwork.";
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
    enhanceKeyLabel.textContent =
      p === "openai" ? "OpenAI API Key" :
      p === "xai"    ? "xAI (Grok) API Key" : "Gemini API Key";
  }
  enhanceProvider.addEventListener("change", () => {
    localStorage.setItem(enhKeyName(), enhanceKey.value);
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
        excluded: new Set(), erased: new Set(), glows: new Set(), offsets: {}, colors: {}, fontScales: {}, boxes: {}, error: "", rev: 0,
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

  /* ══ CROP ══ */
  const cropModal  = document.getElementById("cropModal");
  const cropStage  = document.getElementById("cropStage");
  const cropImg    = document.getElementById("cropImg");
  const cropSel    = document.getElementById("cropSel");
  const cropApply  = document.getElementById("cropApply");
  const cropCancel = document.getElementById("cropCancel");
  const cropBtn    = document.getElementById("cropBtn");

  let cropRect = null;  // {x,y,w,h} in natural image pixels

  if (cropBtn) cropBtn.addEventListener("click", () => {
    if (!pages.length) return;
    cropImg.src = pages[0].thumb;
    cropSel.style.display = "none";
    cropApply.disabled = true;
    cropRect = null;
    cropModal.style.display = "flex";
  });

  if (cropCancel) cropCancel.addEventListener("click", () => {
    cropModal.style.display = "none";
  });

  // drag-to-select on the crop image
  (function initCropDraw() {
    let drawing = false, sx, sy;
    cropStage.addEventListener("pointerdown", e => {
      if (e.target === cropApply || e.target === cropCancel) return;
      drawing = true;
      const r = cropImg.getBoundingClientRect();
      sx = e.clientX; sy = e.clientY;
      cropSel.style.display = "block";
      cropSel.style.left = (sx - r.left) + "px";
      cropSel.style.top = (sy - r.top) + "px";
      cropSel.style.width = "0"; cropSel.style.height = "0";
      // position relative to stage, but we need it relative to the image
      try { cropStage.setPointerCapture(e.pointerId); } catch (_) {}
    });
    cropStage.addEventListener("pointermove", e => {
      if (!drawing) return;
      const r = cropImg.getBoundingClientRect();
      const cx = e.clientX, cy = e.clientY;
      const left = Math.max(0, Math.min(sx, cx) - r.left);
      const top = Math.max(0, Math.min(sy, cy) - r.top);
      const right = Math.min(r.width, Math.max(sx, cx) - r.left);
      const bottom = Math.min(r.height, Math.max(sy, cy) - r.top);
      cropSel.style.left = (r.left - cropStage.getBoundingClientRect().left + left) + "px";
      cropSel.style.top = (r.top - cropStage.getBoundingClientRect().top + top) + "px";
      cropSel.style.width = (right - left) + "px";
      cropSel.style.height = (bottom - top) + "px";
      // store in natural pixels
      const scaleX = cropImg.naturalWidth / r.width;
      const scaleY = cropImg.naturalHeight / r.height;
      cropRect = {
        x: Math.round(left * scaleX), y: Math.round(top * scaleY),
        w: Math.round((right - left) * scaleX), h: Math.round((bottom - top) * scaleY),
      };
      cropApply.disabled = (cropRect.w < 20 || cropRect.h < 20);
    });
    const endCrop = e => {
      if (!drawing) return;
      drawing = false;
      try { cropStage.releasePointerCapture(e.pointerId); } catch (_) {}
    };
    cropStage.addEventListener("pointerup", endCrop);
    cropStage.addEventListener("pointercancel", endCrop);
  })();

  if (cropApply) cropApply.addEventListener("click", () => {
    if (!cropRect || cropRect.w < 20 || cropRect.h < 20) return;
    const img = new window.Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = cropRect.w; canvas.height = cropRect.h;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, cropRect.x, cropRect.y, cropRect.w, cropRect.h,
                    0, 0, cropRect.w, cropRect.h);
      canvas.toBlob(blob => {
        if (!blob) return;
        const page = pages[0];
        if (!page) return;
        const ext = page.name.match(/\.[^.]+$/) || [".png"];
        const cropped = new File([blob], "cropped_" + page.name, { type: blob.type || "image/png" });
        try { URL.revokeObjectURL(page.thumb); } catch (_) {}
        page.file = cropped;
        page.size = blob.size;
        page.thumb = URL.createObjectURL(blob);
        previewImg.src = page.thumb;
        fileName.textContent = page.name + " (cropped)";
        fileSize.textContent = formatBytes(page.size);
        cropModal.style.display = "none";
      }, "image/png");
    };
    img.src = pages[0].thumb;
  });

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
    if (isClean(workflow)) {
      // Clean only: erase all text, no translation (no API key needed).
      f.append("clean_only", "true");
      f.append("font", fontSelect.value);
      f.append("finish", pageFinish ? pageFinish.value : "clean");
      f.append("upscale", hdUpscale && hdUpscale.checked ? "true" : "false");
      f.append("compress", compressOut && compressOut.checked ? "true" : "false");
      if (watermarkInput && watermarkInput.value.trim()) {
        f.append("watermark", watermarkInput.value.trim());
      }
      if (creditInput && creditInput.value.trim()) {
        f.append("credit", creditInput.value.trim());
      }
      return { url: "/api/translate", form: f };
    }
    if (needsTranslate(workflow)) {
      f.append("api_key", apiKeyInput.value.trim());
      f.append("target_lang", targetLang.value);
      f.append("source_lang", sourceLang ? sourceLang.value : "Japanese");
      f.append("provider", engineSelect.value);
      f.append("model", modelSelect.value);
      f.append("smart_mode", smartMode.checked ? "true" : "false");
      f.append("translate_sfx", translateSfx && translateSfx.checked ? "true" : "false");
      f.append("max_quality", maxQuality && maxQuality.checked ? "true" : "false");
      f.append("compress", compressOut && compressOut.checked ? "true" : "false");
      f.append("remove_watermark", removeWatermark && removeWatermark.checked ? "true" : "false");
      f.append("replace_watermark", replaceWatermark && replaceWatermark.checked ? "true" : "false");
      f.append("upscale", hdUpscale && hdUpscale.checked ? "true" : "false");
      f.append("font", fontSelect.value);
      f.append("text_case", textCase ? textCase.value : "upper");
      f.append("finish", pageFinish ? pageFinish.value : "clean");
      const st = styleText();
      if (st) f.append("style_prompt", st);
      f.append("enhance", needsScan(workflow) ? "true" : "false");
      if (needsScan(workflow)) {
        f.append("enhance_provider", enhanceProvider.value);
        f.append("enhance_key", enhanceKey.value.trim());
        f.append("enhance_prompt", enhancePrompt.value);
        f.append("enhance_model", enhanceModel.value);
      }
      if (watermarkInput && watermarkInput.value.trim()) {
        f.append("watermark", watermarkInput.value.trim());
      }
      if (creditInput && creditInput.value.trim()) {
        f.append("credit", creditInput.value.trim());
      }
      if (profileSelect && profileSelect.value) {
        f.append("profile", profileSelect.value);
      }
      return { url: "/api/translate", form: f };
    }
    if (isUpscaleOnly(workflow)) {
      // Faithful HD upscale, no translation and no API keys needed.
      return { url: "/api/upscale", form: f };
    }
    f.append("provider", enhanceProvider.value);
    f.append("api_key", enhanceKey.value.trim());
    f.append("prompt", enhancePrompt.value);
    f.append("model", enhanceModel.value);
    // "AI Scan → HD" chains MangaJaNai after the generative scan; the HD toggle
    // also forces the upscale stage on a plain "Raw → Scan" run.
    if (workflow === "scan-upscale" || (hdUpscale && hdUpscale.checked)) {
      f.append("upscale", "true");
    }
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
            page.erased = new Set();
            page.fontScales = {};
            page.glows = new Set();
            page.boxes = {};
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
    // Result is image-only (no translation) for the scan / upscale workflows —
    // label the panes accordingly so it never says "Translated" when it didn't.
    const noTranslate = !needsTranslate(workflow);
    const leftLabel = (workflow === "raw-scan" || workflow === "scan-upscale") ? "Rough" : "Original";
    const rightLabel = workflow === "upscale-only" ? "Upscaled HD"
                     : workflow === "scan-upscale" ? "Scan + HD"
                     : workflow === "raw-scan" ? "Manga Scan"
                     : "Translated";
    const tabLabel = workflow === "raw-scan" ? "Scan"
                   : (workflow === "upscale-only" || workflow === "scan-upscale") ? "HD"
                   : "Translated";

    if (p.status === "done") {
      pageProcessing.style.display = "none";
      pageResult.style.display = "";
      const bust = `?t=${p.rev}`;
      origImg.src = `/api/original/${p.taskId}`;
      transImg.src = `/api/result/${p.taskId}${bust}`;
      origFull.src = origImg.src;
      transFull.src = transImg.src;
      compLabelLeft.textContent  = leftLabel;
      compLabelRight.textContent = rightLabel;
      tabTranslated.textContent  = tabLabel;
      detailsTab.style.display   = noTranslate ? "none" : "";
      translateScanBtn.style.display = noTranslate ? "" : "none";
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
      stepsEl.style.display = noTranslate ? "none" : "";
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
    const added = page.added || [];
    if (items.length === 0 && added.length === 0) {
      el.innerHTML = '<p style="color:var(--text-dim)">No text regions found. Use the <strong>Translated</strong> tab → <strong>＋ Add</strong> to place text by hand.</p>';
      return;
    }

    for (const it of items) {
      const isErased = page.erased.has(String(it.id));
      const isExcluded = page.excluded.has(String(it.id));
      const isBubble = it.in_bubble !== false;
      const div = document.createElement("div");
      div.className = "tl-item" + (isErased ? " erased" : "")
        + (isExcluded ? " excluded" : "") + (isBubble ? "" : " free-text");
      let badge;
      if (isErased) {
        badge = '<span class="tl-badge skip">erased from art</span>';
      } else if (isExcluded) {
        badge = '<span class="tl-badge skip">shows original</span>';
      } else if (isBubble) {
        badge = '<span class="tl-badge ok">bubble</span>';
      } else {
        const label = it.type || "free text";
        badge = `<span class="tl-badge free">${esc(label)}</span>`;
      }
      const skipTitle = isBubble ? "Skip this bubble" : "Skip this text";
      const c = (page.colors || {})[it.id] || "auto";
      div.innerHTML = `
        <div class="tl-header">
          <span class="tl-id">#${it.id}</span>
          <span class="tl-type">${esc(it.type || "dialogue")}</span>
          ${badge}
          <span class="tl-color" data-id="${it.id}">
            <button class="tl-clr${c === "auto" ? " on" : ""}" data-c="auto" title="Auto color">A</button>
            <button class="tl-clr${c === "black" ? " on" : ""}" data-c="black" title="Black text">B</button>
            <button class="tl-clr${c === "white" ? " on" : ""}" data-c="white" title="White text">W</button>
          </span>
          <span class="tl-fs" title="Font size for this bubble">
            <button class="tl-fsb" data-id="${it.id}" data-d="-1" title="Smaller">A−</button>
            <span class="tl-fsv">${Math.round(((page.fontScales || {})[it.id] || 1) * 100)}%</span>
            <button class="tl-fsb" data-id="${it.id}" data-d="1" title="Bigger">A+</button>
          </span>
          <button class="tl-glow${page.glows && page.glows.has(String(it.id)) ? " on" : ""}" title="Add a soft outer glow (match stylized/glowing original text)" data-id="${it.id}">✨</button>
          <button class="tl-erase${isErased ? " on" : ""}" title="Erase this region from the art (e.g. a watermark the AI typeset by mistake)" data-id="${it.id}">⌫</button>
          <button class="tl-x" title="${skipTitle}" data-id="${it.id}">✕</button>
        </div>
        <div class="tl-original">${esc(it.original || "")}</div>
        <textarea class="tl-edit" data-id="${it.id}" rows="2" ${isExcluded || isErased ? "disabled" : ""}>${esc(it.translation || "")}</textarea>`;
      el.appendChild(div);
    }

    // Manually added regions (drawn with the Add tool).
    for (const it of added) {
      const div = document.createElement("div");
      div.className = "tl-item added-item";
      const ac = (page.colors || {})[it.id] || "auto";
      div.innerHTML = `
        <div class="tl-header">
          <span class="tl-id">＋</span>
          <span class="tl-type">added</span>
          <span class="tl-badge added">manual</span>
          <span class="tl-color" data-id="${it.id}">
            <button class="tl-clr${ac === "auto" ? " on" : ""}" data-c="auto" title="Auto color">A</button>
            <button class="tl-clr${ac === "black" ? " on" : ""}" data-c="black" title="Black text">B</button>
            <button class="tl-clr${ac === "white" ? " on" : ""}" data-c="white" title="White text">W</button>
          </span>
          <span class="tl-fs" title="Font size for this text">
            <button class="tl-fsb" data-id="${it.id}" data-d="-1" title="Smaller">A−</button>
            <span class="tl-fsv">${Math.round(((page.fontScales || {})[it.id] || 1) * 100)}%</span>
            <button class="tl-fsb" data-id="${it.id}" data-d="1" title="Bigger">A+</button>
          </span>
          <button class="tl-del" title="Delete this added text" data-id="${it.id}">✕</button>
        </div>
        <textarea class="tl-edit add-edit" data-id="${it.id}" rows="2">${esc(it.translation || "")}</textarea>`;
      el.appendChild(div);
    }

    // Live sync: typing updates the page model immediately so on-image edits
    // and Details edits never clobber each other.
    el.querySelectorAll(".tl-edit").forEach(t => {
      t.addEventListener("input", () => {
        if (t.classList.contains("add-edit")) {
          const a = (page.added || []).find(i => String(i.id) === t.dataset.id);
          if (a) a.translation = t.value;
        } else {
          const it = (page.items || []).find(i => String(i.id) === t.dataset.id);
          if (it) it.translation = t.value;
        }
      });
    });
    el.querySelectorAll(".tl-x").forEach(btn => {
      btn.addEventListener("click", () => {
        pushUndo(page);
        collectEdits(page);
        const id = btn.dataset.id;
        if (page.excluded.has(id)) page.excluded.delete(id); else page.excluded.add(id);
        buildTranslationsList(page);
      });
    });
    el.querySelectorAll(".tl-erase").forEach(btn => {
      btn.addEventListener("click", () => {
        pushUndo(page);
        collectEdits(page);
        const id = btn.dataset.id;
        page.erased = page.erased || new Set();
        if (page.erased.has(id)) page.erased.delete(id);
        else { page.erased.add(id); page.excluded.delete(id); }
        buildTranslationsList(page);
      });
    });
    el.querySelectorAll(".tl-glow").forEach(btn => {
      btn.addEventListener("click", () => {
        pushUndo(page);
        collectEdits(page);
        const id = btn.dataset.id;
        page.glows = page.glows || new Set();
        if (page.glows.has(id)) page.glows.delete(id); else page.glows.add(id);
        btn.classList.toggle("on");
      });
    });
    el.querySelectorAll(".tl-fsb").forEach(btn => {
      btn.addEventListener("click", () => {
        pushUndo(page);
        collectEdits(page);
        const id = btn.dataset.id;
        page.fontScales = page.fontScales || {};
        const cur = page.fontScales[id] || 1.0;
        const next = Math.max(0.5, Math.min(2.5,
          Math.round((cur + 0.1 * Number(btn.dataset.d)) * 10) / 10));
        if (next === 1.0) delete page.fontScales[id]; else page.fontScales[id] = next;
        // Update just this row's % readout without rebuilding the whole list.
        const span = btn.parentElement.querySelector(".tl-fsv");
        if (span) span.textContent = Math.round((page.fontScales[id] || 1) * 100) + "%";
      });
    });
    el.querySelectorAll(".tl-del").forEach(btn => {
      btn.addEventListener("click", () => {
        page.added = (page.added || []).filter(a => String(a.id) !== btn.dataset.id);
        buildTranslationsList(page);
      });
    });
    el.querySelectorAll(".tl-color").forEach(grp => {
      grp.querySelectorAll(".tl-clr").forEach(btn => {
        btn.addEventListener("click", () => {
          page.colors = page.colors || {};
          page.colors[grp.dataset.id] = btn.dataset.c;
          grp.querySelectorAll(".tl-clr").forEach(b => b.classList.toggle("on", b === btn));
        });
      });
    });
  }

  function collectEdits(page) {
    document.querySelectorAll(".tl-edit").forEach(t => {
      if (t.classList.contains("add-edit")) {
        const a = (page.added || []).find(i => String(i.id) === t.dataset.id);
        if (a) a.translation = t.value;
      } else {
        const it = (page.items || []).find(i => String(i.id) === t.dataset.id);
        if (it) it.translation = t.value;
      }
    });
  }

  applyBtn.addEventListener("click", () => applyChanges());
  async function applyChanges(btn) {
    const page = getActive();
    if (!page || !page.taskId) return;
    collectEdits(page);
    const edits = {};
    (page.items || []).forEach(it => { edits[it.id] = it.translation; });

    const useBtn = btn || applyBtn;
    const label = useBtn.textContent;
    useBtn.disabled = true; useBtn.textContent = "Re-rendering...";
    try {
      const res = await fetch(`/api/rerender/${page.taskId}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          excluded: [...page.excluded], erased: [...(page.erased || [])],
          glows: [...(page.glows || [])], edits,
          font_scale: parseFloat(fontScale.value),
          font_scales: page.fontScales || {},
          boxes: page.boxes || {},
          offsets: page.offsets || {},
          covers: page.covers || [],
          colors: page.colors || {},
          added: (page.added || []).map(a => ({ id: a.id, bbox: a.bbox, translation: a.translation })),
        }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const data = await res.json();
      page.items = data.items;
      if (data.added) {
        page.added = (page.added || []).map(a => {
          const m = data.added.find(d => String(d.id) === String(a.id));
          return m ? Object.assign({}, a, { placed: m.placed }) : a;
        });
      }
      page.rev++;
      renderStrip();
      renderActivePage();
    } catch (e) {
      showError(e.message);
    } finally {
      useBtn.disabled = false; useBtn.textContent = label;
    }
  }

  /* ══ ON-IMAGE TOOLS: move · edit · cover · add ══ */
  let tool = null;
  const HINTS = {
    move: "Drag any translation to move it, then Apply & Re-render.",
    resize: "Drag a box's handles to resize/reshape it (or drag its middle to move it) so the text fits, then Apply & Re-render.",
    edit: "Click any translation to fix its text.",
    cover: "Drag a box over a watermark or leftover text to erase it, then Apply & Re-render.",
    lasso: "Draw a free-form outline around anything (weird shapes) — it's content-aware erased. Then Apply & Re-render.",
    "lasso-add": "Draw a free-form shape over weird-shaped or missed text — it's read & translated. Edit, then Apply & Re-render.",
    add: "Drag a box over missed text — it's OCR'd and auto-translated; edit, then Apply & Re-render.",
    keep: "Draw an outline around the page (like tracing its edge); everything OUTSIDE becomes white — removes carpet/floor/background. Then Apply & Re-render.",
  };

  toolBtns.forEach(b => b.addEventListener("click", () => setTool(b.dataset.tool)));
  editApply.addEventListener("click", () => applyChanges(editApply));

  const STYLE_PRESETS = {
    natural: "",
    literal: "Translate faithfully and accurately — stay close to the original "
      + "wording and meaning; prefer fidelity over flourish.",
    liberal: "Localize liberally — rephrase freely so it reads like it was "
      + "originally written in the target language; prioritize natural flow and impact.",
  };
  const glossary = document.getElementById("glossary");
  function styleText() {
    const preset = STYLE_PRESETS[transStyle ? transStyle.value : "natural"] || "";
    const user = stylePrompt && stylePrompt.value.trim() ? stylePrompt.value.trim() : "";
    const gl = (betaOn() && glossary && glossary.value.trim())
      ? "GLOSSARY — translate these names/terms EXACTLY and consistently on "
        + "every page:\n" + glossary.value.trim()
      : "";
    return [preset, gl, user].filter(Boolean).join("\n");
  }
  if (transStyle) {
    transStyle.value = localStorage.getItem("manga_trans_style") || "natural";
    transStyle.addEventListener("change", () =>
      localStorage.setItem("manga_trans_style", transStyle.value));
  }

  // Formerly-"beta" tools (glossary, free-form add, undo/redo) are now always on.
  function betaOn() { return true; }
  document.body.classList.add("beta");

  // Collapse / expand the settings panel (state remembered across reloads).
  const settingsBar = document.getElementById("settingsBar");
  const settingsToggle = document.getElementById("settingsToggle");
  if (settingsBar && settingsToggle) {
    // Open by default; only collapsed if the user explicitly collapsed it before.
    if (localStorage.getItem("manga_settings_collapsed") === "1")
      settingsBar.classList.add("collapsed");
    settingsToggle.addEventListener("click", () => {
      const collapsed = settingsBar.classList.toggle("collapsed");
      localStorage.setItem("manga_settings_collapsed", collapsed ? "1" : "0");
    });
  }
  if (glossary) {
    glossary.value = localStorage.getItem("manga_glossary") || "";
    glossary.addEventListener("input", () =>
      localStorage.setItem("manga_glossary", glossary.value));
  }

  // ── Undo / Redo (beta): snapshot the page's edit state before each change ──
  function _snapshot(page) {
    return JSON.stringify({
      excluded: [...(page.excluded || [])], erased: [...(page.erased || [])],
      glows: [...(page.glows || [])], offsets: page.offsets || {},
      colors: page.colors || {}, fontScales: page.fontScales || {},
      boxes: page.boxes || {}, covers: page.covers || [],
      added: (page.added || []).map(a => ({ ...a })),
      trans: (page.items || []).map(it => it.translation || ""),
    });
  }
  function _restore(page, str) {
    const s = JSON.parse(str);
    page.excluded = new Set(s.excluded); page.erased = new Set(s.erased);
    page.glows = new Set(s.glows); page.offsets = s.offsets; page.colors = s.colors;
    page.fontScales = s.fontScales; page.boxes = s.boxes; page.covers = s.covers;
    page.added = s.added;
    (page.items || []).forEach((it, i) => { if (i < s.trans.length) it.translation = s.trans[i]; });
  }
  function pushUndo(page) {
    if (!betaOn() || !page) return;
    page._undo = page._undo || []; page._redo = [];
    page._undo.push(_snapshot(page));
    if (page._undo.length > 40) page._undo.shift();
  }
  function doUndo() {
    const page = getActive();
    if (!page || !(page._undo && page._undo.length)) return;
    page._redo = page._redo || []; page._redo.push(_snapshot(page));
    _restore(page, page._undo.pop());
    buildTranslationsList(page); if (tool) buildOverlay();
    if (editHint) editHint.textContent = "Undid last change. Apply & Re-render to see it.";
  }
  function doRedo() {
    const page = getActive();
    if (!page || !(page._redo && page._redo.length)) return;
    page._undo = page._undo || []; page._undo.push(_snapshot(page));
    _restore(page, page._redo.pop());
    buildTranslationsList(page); if (tool) buildOverlay();
  }
  document.addEventListener("keydown", e => {
    if (!betaOn()) return;
    const k = (e.key || "").toLowerCase();
    const mod = e.ctrlKey || e.metaKey;
    if (mod && k === "z" && !e.shiftKey) { e.preventDefault(); doUndo(); }
    else if (mod && (k === "y" || (k === "z" && e.shiftKey))) { e.preventDefault(); doRedo(); }
  });
  const undoBtn = document.getElementById("undoBtn");
  const redoBtn = document.getElementById("redoBtn");
  if (undoBtn) undoBtn.addEventListener("click", doUndo);
  if (redoBtn) redoBtn.addEventListener("click", doRedo);

  const rescanBtn = document.getElementById("rescanBtn");
  if (rescanBtn) rescanBtn.addEventListener("click", async () => {
    const page = getActive();
    if (!page || !page.taskId) return;
    const key = apiKeyInput.value.trim();
    if (!key) { editHint.textContent = "Enter your API key first."; return; }
    collectEdits(page);
    const label = rescanBtn.textContent;
    rescanBtn.disabled = true; rescanBtn.textContent = "Scanning…";
    editHint.textContent = "Re-scanning this page for missed text…";
    try {
      const res = await fetch(`/api/rescan/${page.taskId}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          api_key: key, target_lang: targetLang.value,
          provider: engineSelect.value, model: modelSelect.value,
          style_prompt: styleText(),
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "re-scan failed");
      if (data.items) page.items = data.items;
      buildTranslationsList(page);
      if (data.added_count > 0) {
        editHint.textContent = `Found ${data.added_count} missed region(s) — re-rendering…`;
        await applyChanges(rescanBtn);
      } else {
        editHint.textContent = "No missed text found on this page.";
      }
    } catch (e) {
      editHint.textContent = "Re-scan failed: " + (e.message || e);
    } finally {
      rescanBtn.disabled = false; rescanBtn.textContent = label;
    }
  });

  function setTool(t) {
    tool = (tool === t) ? null : t;
    toolBtns.forEach(b => {
      const on = b.dataset.tool === tool;
      b.classList.toggle("btn-primary", on);
      b.classList.toggle("btn-ghost", !on);
    });
    moveLayer.classList.toggle("on", !!tool);
    moveLayer.dataset.tool = tool || "";
    editApply.style.display = tool ? "" : "none";
    editHint.textContent = HINTS[tool] || "Pick a tool to fix anything by hand.";
    closeEditor();
    buildOverlay();
  }

  function curDims() {
    const W = transFull.naturalWidth, H = transFull.naturalHeight;
    return (W && H) ? [W, H] : null;
  }

  function makeBox(bx, by, bw, bh, W, H, cls) {
    const box = document.createElement("div");
    box.className = cls;
    box.style.left = (bx / W * 100) + "%";
    box.style.top = (by / H * 100) + "%";
    box.style.width = (bw / W * 100) + "%";
    box.style.height = (bh / H * 100) + "%";
    return box;
  }

  function buildOverlay() {
    const page = getActive();
    closeEditor();
    moveLayer.innerHTML = "";
    if (!page || !tool) return;
    const dims = curDims();
    if (!dims) return;
    const [W, H] = dims;
    page.offsets = page.offsets || {};
    page.covers = page.covers || [];
    page.added = page.added || [];

    // Existing cover regions — always visible, click to remove.
    page.covers.forEach((cb, i) => {
      if (cb && cb.keep_poly) {   // Remove BG outline (keep inside, white outside)
        const NS = "http://www.w3.org/2000/svg";
        const svg = document.createElementNS(NS, "svg");
        svg.setAttribute("viewBox", "0 0 100 100");
        svg.setAttribute("preserveAspectRatio", "none");
        svg.style.cssText = "position:absolute;inset:0;width:100%;height:100%;z-index:4;cursor:pointer";
        const pg = document.createElementNS(NS, "polygon");
        pg.setAttribute("points", cb.keep_poly.map(p => `${p[0] / W * 100},${p[1] / H * 100}`).join(" "));
        pg.setAttribute("fill", "none");
        pg.setAttribute("stroke", "#2563eb");
        pg.setAttribute("stroke-width", "0.6");
        pg.setAttribute("stroke-dasharray", "2 1.5");
        svg.appendChild(pg);
        svg.addEventListener("click", () => { page.covers.splice(i, 1); buildOverlay(); });
        moveLayer.appendChild(svg);
        return;
      }
      if (cb && cb.poly) {   // free-form lasso erase
        const NS = "http://www.w3.org/2000/svg";
        const svg = document.createElementNS(NS, "svg");
        svg.setAttribute("viewBox", "0 0 100 100");
        svg.setAttribute("preserveAspectRatio", "none");
        svg.style.cssText = "position:absolute;inset:0;width:100%;height:100%;z-index:4;cursor:pointer";
        const pg = document.createElementNS(NS, "polygon");
        pg.setAttribute("points", cb.poly.map(p => `${p[0] / W * 100},${p[1] / H * 100}`).join(" "));
        pg.setAttribute("fill", "rgba(220,38,38,.18)");
        pg.setAttribute("stroke", "#dc2626");
        pg.setAttribute("stroke-width", "0.5");
        svg.appendChild(pg);
        svg.addEventListener("click", () => { page.covers.splice(i, 1); buildOverlay(); });
        moveLayer.appendChild(svg);
        return;
      }
      const box = makeBox(cb[0], cb[1], cb[2], cb[3], W, H, "cover-box");
      box.innerHTML = `<span class="ov-tag">erase ✕</span>`;
      box.title = "Click to remove this cover";
      box.addEventListener("click", () => { page.covers.splice(i, 1); buildOverlay(); });
      moveLayer.appendChild(box);
    });

    if (tool === "move" || tool === "edit") {
      for (const it of (page.items || [])) {
        if (!it.placed || !it.bbox || page.excluded.has(String(it.id))) continue;
        addItemBox(it, page, W, H, false);
      }
      for (const it of (page.added || [])) {
        if (!it.bbox) continue;
        addItemBox(it, page, W, H, true);
      }
    }

    if (tool === "resize") {
      page.boxes = page.boxes || {};
      for (const it of (page.items || [])) {
        if (!it.bbox || page.excluded.has(String(it.id))) continue;
        addResizeBox(it, page, W, H, false);
      }
      for (const it of (page.added || [])) {
        if (it.bbox) addResizeBox(it, page, W, H, true);
      }
    }
  }

  // A box with 8 Photoshop-style handles: drag the body to move it, drag a
  // handle to reshape it. The new rect is stored as an absolute-pixel override
  // in page.boxes[id] and applied on Apply & Re-render.
  function addResizeBox(it, page, W, H, isAdded) {
    const base = (page.boxes[it.id] || it.bbox).slice();
    const box = makeBox(base[0], base[1], base[2], base[3], W, H,
                        "resize-box" + (isAdded ? " added-box" : ""));
    box.innerHTML = `<span class="move-tag">${isAdded ? "✎" : "#" + it.id}</span>` +
      ["nw", "n", "ne", "e", "se", "s", "sw", "w"]
        .map(d => `<span class="rsz-h rsz-${d}" data-d="${d}"></span>`).join("");
    bindResize(box, it, page, W, H);
    moveLayer.appendChild(box);
  }

  function bindResize(box, it, page, W, H) {
    const cur = () => (page.boxes[it.id] || it.bbox).slice();
    function apply(b) {
      page.boxes[it.id] = b;
      box.style.left = (b[0] / W * 100) + "%";
      box.style.top = (b[1] / H * 100) + "%";
      box.style.width = (b[2] / W * 100) + "%";
      box.style.height = (b[3] / H * 100) + "%";
    }
    let mode = null, sx, sy, base;
    function down(e, m) {
      e.preventDefault(); e.stopPropagation();
      pushUndo(page);
      mode = m; sx = e.clientX; sy = e.clientY; base = cur();
      box.classList.add("dragging");
      try { box.setPointerCapture(e.pointerId); } catch (_) {}
    }
    box.addEventListener("pointerdown", e => {
      if (e.target.classList.contains("rsz-h")) down(e, e.target.dataset.d);
      else down(e, "move");
    });
    box.addEventListener("pointermove", e => {
      if (!mode) return;
      const rect = transFull.getBoundingClientRect();
      const dx = (e.clientX - sx) * (W / rect.width);
      const dy = (e.clientY - sy) * (H / rect.height);
      let [x, y, w, h] = base;
      if (mode === "move") { x += dx; y += dy; }
      else {
        if (mode.includes("w")) { x += dx; w -= dx; }
        if (mode.includes("e")) { w += dx; }
        if (mode.includes("n")) { y += dy; h -= dy; }
        if (mode.includes("s")) { h += dy; }
      }
      w = Math.max(10, w); h = Math.max(10, h);
      apply([Math.round(x), Math.round(y), Math.round(w), Math.round(h)]);
    });
    const end = e => {
      mode = null; box.classList.remove("dragging");
      try { box.releasePointerCapture(e.pointerId); } catch (_) {}
    };
    box.addEventListener("pointerup", end);
    box.addEventListener("pointercancel", end);
  }

  function addItemBox(it, page, W, H, isAdded) {
    const [bx, by, bw, bh] = it.bbox;
    const off = page.offsets[it.id] || [0, 0];
    const box = makeBox(bx + off[0], by + off[1], bw, bh, W, H,
                        "move-box" + (isAdded ? " added-box" : ""));
    box.innerHTML = `<span class="move-tag">${isAdded ? "✎" : "#" + it.id}</span>`;
    if (tool === "move") {
      box.classList.add("draggable");
      bindDrag(box, it, page, W, H);
    } else {
      box.classList.add("editable");
      box.addEventListener("click", () => openEditor(it, page, isAdded));
    }
    moveLayer.appendChild(box);
  }

  function bindDrag(box, it, page, W, H) {
    let startX, startY, baseX, baseY, dragging = false;
    box.addEventListener("pointerdown", e => {
      e.preventDefault();
      pushUndo(page);
      dragging = true;
      box.classList.add("dragging");
      try { box.setPointerCapture(e.pointerId); } catch (_) {}
      startX = e.clientX; startY = e.clientY;
      const off = page.offsets[it.id] || [0, 0];
      baseX = off[0]; baseY = off[1];
    });
    box.addEventListener("pointermove", e => {
      if (!dragging) return;
      const rect = transFull.getBoundingClientRect();
      const sx = W / rect.width, sy = H / rect.height;
      const dx = Math.round(baseX + (e.clientX - startX) * sx);
      const dy = Math.round(baseY + (e.clientY - startY) * sy);
      page.offsets[it.id] = [dx, dy];
      const [bx, by] = it.bbox;
      box.style.left = ((bx + dx) / W * 100) + "%";
      box.style.top = ((by + dy) / H * 100) + "%";
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

  /* inline text editor popup (Edit tool) */
  function closeEditor() {
    const ex = moveLayer.querySelector(".edit-pop");
    if (ex) ex.remove();
  }
  function openEditor(it, page, isAdded) {
    closeEditor();
    const dims = curDims();
    if (!dims) return;
    const [W, H] = dims;
    const off = page.offsets[it.id] || [0, 0];
    const [bx, by, , bh] = it.bbox;
    const pop = document.createElement("div");
    pop.className = "edit-pop";
    pop.style.left = Math.min((bx + off[0]) / W * 100, 62) + "%";
    pop.style.top = Math.min((by + off[1] + bh) / H * 100 + 1, 82) + "%";
    const curClr = (page.colors || {})[it.id] || "auto";
    pop.innerHTML = `
      ${it.original ? `<div class="edit-pop-orig">${esc(it.original)}</div>` : ""}
      <textarea class="edit-pop-text" rows="3">${esc(it.translation || "")}</textarea>
      <div class="edit-pop-color">
        <span class="epop-clabel">Color</span>
        <button class="clr-opt${curClr === "auto" ? " active" : ""}" data-c="auto">Auto</button>
        <button class="clr-opt${curClr === "black" ? " active" : ""}" data-c="black"><span class="clr-dot" style="background:#111"></span> Black</button>
        <button class="clr-opt${curClr === "white" ? " active" : ""}" data-c="white"><span class="clr-dot" style="background:#fff;border-color:#999"></span> White</button>
      </div>
      <div class="edit-pop-row">
        <button class="btn btn-ghost btn-sm epop-remove">${isAdded ? "Delete" : "Skip"}</button>
        <span class="spacer"></span>
        <button class="btn btn-ghost btn-sm epop-cancel">Cancel</button>
        <button class="btn btn-primary btn-sm epop-save">Save</button>
      </div>`;
    pop.addEventListener("pointerdown", e => e.stopPropagation());
    moveLayer.appendChild(pop);
    const ta = pop.querySelector(".edit-pop-text");
    ta.focus(); ta.setSelectionRange(ta.value.length, ta.value.length);
    pop.querySelector(".epop-cancel").addEventListener("click", closeEditor);
    pop.querySelector(".epop-save").addEventListener("click", () => {
      it.translation = ta.value;
      const f = document.querySelector('.tl-edit[data-id="' + it.id + '"]');
      if (f) f.value = ta.value;
      closeEditor();
      applyChanges(editApply);
    });
    pop.querySelectorAll(".clr-opt").forEach(btn => {
      btn.addEventListener("click", () => {
        page.colors = page.colors || {};
        page.colors[it.id] = btn.dataset.c;
        pop.querySelectorAll(".clr-opt").forEach(b => b.classList.toggle("active", b === btn));
      });
    });
    pop.querySelector(".epop-remove").addEventListener("click", () => {
      if (isAdded) {
        page.added = (page.added || []).filter(a => String(a.id) !== String(it.id));
      } else {
        page.excluded.add(String(it.id));
      }
      closeEditor();
      applyChanges(editApply);
    });
  }

  /* draw-a-box surface (Cover / Add tools) — bound once on the overlay */
  (function initDraw() {
    let drawing = false, sx, sy, rectEl = null;
    moveLayer.addEventListener("pointerdown", e => {
      if (tool !== "cover" && tool !== "add") return;
      if (e.target !== moveLayer) return;   // don't start when clicking a box
      const page = getActive(); if (!page || !curDims()) return;
      drawing = true;
      const r = moveLayer.getBoundingClientRect();
      sx = e.clientX - r.left; sy = e.clientY - r.top;
      rectEl = document.createElement("div");
      rectEl.className = (tool === "cover" ? "cover-box" : "add-box") + " drawing";
      rectEl.style.left = (sx / r.width * 100) + "%";
      rectEl.style.top = (sy / r.height * 100) + "%";
      moveLayer.appendChild(rectEl);
      try { moveLayer.setPointerCapture(e.pointerId); } catch (_) {}
    });
    moveLayer.addEventListener("pointermove", e => {
      if (!drawing) return;
      const r = moveLayer.getBoundingClientRect();
      const cx = e.clientX - r.left, cy = e.clientY - r.top;
      rectEl.style.left = (Math.min(sx, cx) / r.width * 100) + "%";
      rectEl.style.top = (Math.min(sy, cy) / r.height * 100) + "%";
      rectEl.style.width = (Math.abs(cx - sx) / r.width * 100) + "%";
      rectEl.style.height = (Math.abs(cy - sy) / r.height * 100) + "%";
    });
    const finish = e => {
      if (!drawing) return;
      drawing = false;
      try { moveLayer.releasePointerCapture(e.pointerId); } catch (_) {}
      const page = getActive(), dims = curDims();
      const r = moveLayer.getBoundingClientRect();
      const cx = e.clientX - r.left, cy = e.clientY - r.top;
      if (rectEl) { rectEl.remove(); rectEl = null; }
      if (!page || !dims) return;
      const [W, H] = dims;
      const x = Math.round(Math.min(sx, cx) / r.width * W);
      const y = Math.round(Math.min(sy, cy) / r.height * H);
      const w = Math.round(Math.abs(cx - sx) / r.width * W);
      const h = Math.round(Math.abs(cy - sy) / r.height * H);
      if (w < 6 || h < 6) return;
      page.covers = page.covers || []; page.added = page.added || [];
      pushUndo(page);
      if (tool === "cover") {
        page.covers.push([x, y, w, h]);
        buildOverlay();
      } else {
        autoTranslate(page, [x, y, w, h]);
      }
    };
    moveLayer.addEventListener("pointerup", finish);
    moveLayer.addEventListener("pointercancel", finish);
  })();

  /* free-form lasso erase surface */
  (function initLasso() {
    const NS = "http://www.w3.org/2000/svg";
    let drawing = false, pts = [], svg = null, poly = null;
    moveLayer.addEventListener("pointerdown", e => {
      if (tool !== "lasso" && tool !== "lasso-add" && tool !== "keep") return;
      if (e.target !== moveLayer) return;
      const page = getActive(); if (!page || !curDims()) return;
      drawing = true; pts = [];
      const add = tool === "lasso-add";
      const keep = tool === "keep";
      svg = document.createElementNS(NS, "svg");
      svg.style.cssText = "position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:6";
      svg.setAttribute("viewBox", "0 0 100 100");
      svg.setAttribute("preserveAspectRatio", "none");
      poly = document.createElementNS(NS, "polyline");
      poly.setAttribute("fill", keep ? "rgba(37,99,235,.12)" : add ? "rgba(22,163,74,.18)" : "rgba(220,38,38,.18)");
      poly.setAttribute("stroke", keep ? "#2563eb" : add ? "#16a34a" : "#dc2626");
      poly.setAttribute("stroke-width", "0.5");
      svg.appendChild(poly); moveLayer.appendChild(svg);
      const r = moveLayer.getBoundingClientRect();
      pts.push([(e.clientX - r.left) / r.width * 100, (e.clientY - r.top) / r.height * 100]);
      try { moveLayer.setPointerCapture(e.pointerId); } catch (_) {}
    });
    moveLayer.addEventListener("pointermove", e => {
      if (!drawing) return;
      const r = moveLayer.getBoundingClientRect();
      pts.push([(e.clientX - r.left) / r.width * 100, (e.clientY - r.top) / r.height * 100]);
      poly.setAttribute("points", pts.map(p => p.join(",")).join(" "));
    });
    const finishLasso = e => {
      if (!drawing) return;
      drawing = false;
      try { moveLayer.releasePointerCapture(e.pointerId); } catch (_) {}
      if (svg) { svg.remove(); svg = null; }
      const page = getActive(), dims = curDims();
      if (!page || !dims || pts.length < 3) { pts = []; return; }
      pushUndo(page);
      const [W, H] = dims;
      const polyImg = pts.map(([px, py]) => [Math.round(px / 100 * W), Math.round(py / 100 * H)]);
      const wasAdd = tool === "lasso-add";
      const wasKeep = tool === "keep";
      pts = [];
      if (wasKeep) {
        // Remove BG: keep inside this outline, white out everything else.
        page.covers = page.covers || [];
        page.covers.push({ keep_poly: polyImg });
        buildOverlay();
        return;
      }
      if (wasAdd) {
        // Free-form add: OCR + translate the shape's bbox, and erase the exact
        // outlined shape (content-aware) so odd-shaped backgrounds stay clean.
        const xs = polyImg.map(p => p[0]), ys = polyImg.map(p => p[1]);
        const x = Math.min(...xs), y = Math.min(...ys);
        const w = Math.max(...xs) - x, h = Math.max(...ys) - y;
        page.covers = page.covers || [];
        page.covers.push({ poly: polyImg });
        if (w >= 6 && h >= 6) autoTranslate(page, [x, y, w, h]);
        else buildOverlay();
      } else {
        page.covers = page.covers || [];
        page.covers.push({ poly: polyImg });
        buildOverlay();
      }
    };
    moveLayer.addEventListener("pointerup", finishLasso);
    moveLayer.addEventListener("pointercancel", finishLasso);
  })();

  async function autoTranslate(page, bbox) {
    editHint.textContent = "Reading & translating…";
    let data = { original: "", translation: "" };
    try {
      const resp = await fetch(`/api/ocr-translate/${page.taskId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          bbox,
          api_key: apiKeyInput.value.trim(),
          provider: engineSelect.value,
          model: modelSelect.value,
          target_lang: targetLang.value,
          style_prompt: stylePrompt ? stylePrompt.value.trim() : "",
        }),
      });
      if (resp.ok) data = await resp.json();
    } catch (_) { /* fall through to manual entry */ }

    // Auto-translated text pre-fills the prompt (editable); if no Japanese was
    // read, fall back to manual entry so the tool still works anywhere.
    const suggested = (data.translation || "").trim();
    const label = suggested
      ? "Edit translation (read: " + data.original + "):"
      : "No text detected here — type the translation:";
    const txt = prompt(label, suggested);
    if (txt && txt.trim()) {
      page.added = page.added || [];
      page.addSeq = (page.addSeq || 0) + 1;
      page.added.push({
        id: "m" + page.addSeq,
        bbox,
        original: data.original || "",
        translation: txt.trim(),
        placed: true,
      });
      buildOverlay();
      editHint.textContent = "Added! Hit Apply & Re-render when ready.";
    } else {
      editHint.textContent = HINTS.add;
    }
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
      if (tab.dataset.tab === "translated" && tool) buildOverlay();
    });
  });
  // Rebuild the tool overlay once the translated image has its real dimensions.
  transFull.addEventListener("load", () => { if (tool) buildOverlay(); });

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

  translateScanBtn.addEventListener("click", async () => {
    const p = getActive();
    if (!p || p.status !== "done") return;
    if (!apiKeyInput.value.trim()) {
      apiKeyInput.focus(); apiKeyInput.style.borderColor = "#f87171"; return;
    }
    apiKeyInput.style.borderColor = "";
    translateScanBtn.disabled = true; translateScanBtn.textContent = "Loading scan...";
    try {
      const res = await fetch(`/api/result/${p.taskId}?t=${p.rev}`);
      const blob = await res.blob();
      const file = new File([blob], "scan_" + (p.name || "page.png"), { type: "image/png" });
      setWorkflow("raw-translate");
      p.file = file;
      try { URL.revokeObjectURL(p.thumb); } catch (_) {}
      p.thumb = URL.createObjectURL(blob);
      p.status = "queued"; p.taskId = null; p.result = null;
      p.items = []; p.excluded = new Set(); p.erased = new Set(); p.glows = new Set(); p.offsets = {}; p.colors = {}; p.fontScales = {}; p.boxes = {};
      p.error = ""; p.rev = 0;
      renderStrip(); updateBatch(); renderActivePage();
      pump();
    } catch (e) {
      showError(e.message);
    } finally {
      translateScanBtn.disabled = false; translateScanBtn.textContent = "Translate This Scan";
    }
  });

  const reuseBtn = document.getElementById("reuseBtn");
  if (reuseBtn) reuseBtn.addEventListener("click", async () => {
    const p = getActive();
    if (!p || p.status !== "done") return;
    reuseBtn.disabled = true; reuseBtn.textContent = "Loading…";
    try {
      const res = await fetch(`/api/result/${p.taskId}?t=${p.rev}`);
      const blob = await res.blob();
      const name = "edited_" + (p.name || "page.png");
      const file = new File([blob], name, { type: blob.type || "image/png" });
      // Start a fresh single-page session from this result, back at the picker.
      pages.forEach(pg => { try { URL.revokeObjectURL(pg.thumb); } catch (_) {} });
      const thumb = URL.createObjectURL(blob);
      pages = [{
        uid: (p.uid || 0), name, file, size: blob.size, thumb,
        taskId: null, status: "pending", progress: 0, step: 0, message: "", result: null,
        items: [], excluded: new Set(), erased: new Set(), glows: new Set(),
        offsets: {}, colors: {}, fontScales: {}, boxes: {}, error: "", rev: 0,
      }];
      activeUid = pages[0].uid;
      previewImg.src = thumb;
      if (fileName) fileName.textContent = name;
      if (fileSize) fileSize.textContent = formatBytes(blob.size);
      previewRow.style.display = "flex";
      dropZone.style.display = "none";
      showSection("upload");
    } catch (e) {
      showError(e.message);
    } finally {
      reuseBtn.disabled = false; reuseBtn.textContent = "↺ Use Result as Input";
    }
  });

  /* ══ TRAINED SERIES STYLE PROFILES ══ */
  const profileSelect = document.getElementById("profileSelect");
  const trainBtn      = document.getElementById("trainBtn");
  const trainModal    = document.getElementById("trainModal");
  const trainProfileSel = document.getElementById("trainProfileSel");
  const trainName     = document.getElementById("trainName");
  const trainFiles    = document.getElementById("trainFiles");
  const trainAll      = document.getElementById("trainAll");
  const trainLearn    = document.getElementById("trainLearn");
  const trainStatus   = document.getElementById("trainStatus");
  const trainResult   = document.getElementById("trainResult");
  const trainStyle    = document.getElementById("trainStyle");
  const trainHon      = document.getElementById("trainHon");
  const trainSfx      = document.getElementById("trainSfx");
  const trainGloss    = document.getElementById("trainGloss");
  const trainSave     = document.getElementById("trainSave");
  const trainDelete   = document.getElementById("trainDelete");
  const trainClose    = document.getElementById("trainClose");
  const trainMeta     = document.getElementById("trainMeta");

  function glossToText(g) {
    return (g || []).map(it => `${it.term ? it.term + " = " : ""}${it.translation}` +
      (it.notes ? `  # ${it.notes}` : "")).join("\n");
  }
  function textToGloss(t) {
    return (t || "").split("\n").map(line => {
      line = line.trim(); if (!line) return null;
      let notes = ""; const h = line.split("#");
      if (h.length > 1) { notes = h.slice(1).join("#").trim(); line = h[0].trim(); }
      const m = line.split(/=|→/);
      if (m.length >= 2) return { term: m[0].trim(), translation: m.slice(1).join("=").trim(), notes };
      return { term: "", translation: line, notes };
    }).filter(x => x && x.translation);
  }

  async function refreshProfiles(selectSlug) {
    if (!profileSelect) return;
    let list = [];
    try { list = (await (await fetch("/api/profiles")).json()).profiles || []; } catch (_) {}
    const cur = selectSlug || profileSelect.value || localStorage.getItem("manga_profile") || "";
    profileSelect.innerHTML = '<option value="">None — generic translation</option>' +
      list.map(p => `<option value="${esc(p.slug)}">${esc(p.name)} · ${p.terms} terms</option>`).join("");
    profileSelect.value = list.some(p => p.slug === cur) ? cur : "";
    localStorage.setItem("manga_profile", profileSelect.value);
    if (trainProfileSel) {
      trainProfileSel.innerHTML = '<option value="">＋ New series…</option>' +
        list.map(p => `<option value="${esc(p.slug)}">${esc(p.name)}</option>`).join("");
    }
  }
  if (profileSelect) {
    profileSelect.addEventListener("change", () => localStorage.setItem("manga_profile", profileSelect.value));
    refreshProfiles();
  }

  function fillProfile(p) {
    trainName.value = p.name || "";
    trainStyle.value = p.style_guide || "";
    trainHon.value = p.honorifics || "";
    trainSfx.value = p.sfx_policy || "";
    trainGloss.value = glossToText(p.glossary);
    trainMeta.textContent = `${(p.glossary || []).length} terms · learned from ${p.sources || 0} pages`;
    trainResult.style.display = "";
    trainDelete.style.display = p.slug ? "" : "none";
  }
  function buildProfileBody() {
    return {
      name: trainName.value.trim(),
      style_guide: trainStyle.value.trim(),
      honorifics: trainHon.value.trim(),
      sfx_policy: trainSfx.value.trim(),
      glossary: textToGloss(trainGloss.value),
    };
  }

  if (trainBtn) trainBtn.addEventListener("click", async () => {
    await refreshProfiles();
    trainProfileSel.value = "";
    trainName.value = ""; trainFiles.value = ""; trainStatus.textContent = "";
    trainResult.style.display = "none"; trainDelete.style.display = "none";
    trainModal.style.display = "flex";
  });
  if (trainClose) trainClose.addEventListener("click", () => { trainModal.style.display = "none"; });
  if (trainModal) trainModal.addEventListener("click", e => { if (e.target === trainModal) trainModal.style.display = "none"; });

  if (trainProfileSel) trainProfileSel.addEventListener("change", async () => {
    const slug = trainProfileSel.value;
    if (!slug) { trainName.value = ""; trainResult.style.display = "none"; trainDelete.style.display = "none"; return; }
    try {
      const p = await (await fetch(`/api/profile/${slug}`)).json();
      fillProfile(p);
    } catch (_) { trainStatus.textContent = "Couldn't load that profile."; }
  });

  if (trainLearn) trainLearn.addEventListener("click", async () => {
    const name = trainName.value.trim();
    if (!name) { trainStatus.textContent = "Enter a series name first."; trainName.focus(); return; }
    if (!trainFiles.files.length) { trainStatus.textContent = "Pick a ZIP or some translated pages."; return; }
    if (!apiKeyInput.value.trim()) { trainStatus.textContent = "Add your translation API key in settings first."; return; }
    const f = new FormData();
    f.append("name", name);
    f.append("provider", engineSelect.value);
    f.append("api_key", apiKeyInput.value.trim());
    f.append("model", modelSelect.value);
    f.append("target_lang", targetLang.value);
    f.append("source_lang", sourceLang ? sourceLang.value : "Japanese");
    f.append("study_all", trainAll && trainAll.checked ? "true" : "false");
    [...trainFiles.files].forEach(file => f.append("files", file));
    trainLearn.disabled = true;
    trainStatus.textContent = (trainAll && trainAll.checked)
      ? "Studying EVERY page with the AI… (this can take a few minutes for many chapters)"
      : "Studying your chapters with the AI… (this can take 30–60s)";
    try {
      const res = await fetch("/api/profile/learn", { method: "POST", body: f });
      if (!res.ok) { let m = res.statusText; try { m = (await res.json()).detail || m; } catch (_) {} throw new Error(m); }
      const data = await res.json();
      fillProfile(data.profile);
      trainStatus.textContent = `Learned from ${data.pages_studied} of ${data.pages_seen} pages. Review & edit below, then Save.`;
      await refreshProfiles(data.profile.slug);
    } catch (e) {
      trainStatus.textContent = "Learning failed: " + e.message;
    } finally {
      trainLearn.disabled = false;
    }
  });

  if (trainSave) trainSave.addEventListener("click", async () => {
    const body = buildProfileBody();
    if (!body.name) { trainStatus.textContent = "A series name is required."; return; }
    trainSave.disabled = true;
    try {
      const slug = body.name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
      const res = await fetch(`/api/profile/${slug}`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const saved = await res.json();
      await refreshProfiles(saved.slug);
      if (profileSelect) { profileSelect.value = saved.slug; localStorage.setItem("manga_profile", saved.slug); }
      trainStatus.textContent = `Saved “${saved.name}”. It's now selected for translation.`;
      trainDelete.style.display = "";
    } catch (e) {
      trainStatus.textContent = "Save failed: " + e.message;
    } finally {
      trainSave.disabled = false;
    }
  });

  if (trainDelete) trainDelete.addEventListener("click", async () => {
    const name = trainName.value.trim();
    const slug = name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
    if (!slug || !confirm(`Delete the “${name}” profile?`)) return;
    try {
      await fetch(`/api/profile/${slug}`, { method: "DELETE" });
      await refreshProfiles("");
      trainName.value = ""; trainResult.style.display = "none"; trainProfileSel.value = "";
      trainStatus.textContent = "Profile deleted.";
    } catch (e) { trainStatus.textContent = "Delete failed: " + e.message; }
  });

  /* ══ END PAGE (one-click "thanks for reading" last page) ══ */
  const endScan    = document.getElementById("endScan");
  const endDiscord = document.getElementById("endDiscord");
  const endShowDiscord = document.getElementById("endShowDiscord");
  const endMessage = document.getElementById("endMessage");
  const endTheme   = document.getElementById("endTheme");
  const endUseColor = document.getElementById("endUseColor");
  const endColor   = document.getElementById("endColor");
  const endCardBtn = document.getElementById("endCardBtn");
  if (endScan) {
    endScan.value    = localStorage.getItem("manga_end_scan")    || "";
    endDiscord.value = localStorage.getItem("manga_end_discord") || "";
    endTheme.value   = localStorage.getItem("manga_end_style")   || "royal";
    if (endMessage) endMessage.value = localStorage.getItem("manga_end_message") || "";
    if (endShowDiscord) endShowDiscord.checked = localStorage.getItem("manga_end_showdiscord") !== "0";
    if (endColor)    endColor.value = localStorage.getItem("manga_end_color") || "#d4af5a";
    if (endUseColor) endUseColor.checked = localStorage.getItem("manga_end_usecolor") === "1";
    endScan.addEventListener("input",    () => localStorage.setItem("manga_end_scan", endScan.value));
    endDiscord.addEventListener("input", () => localStorage.setItem("manga_end_discord", endDiscord.value));
    endTheme.addEventListener("change",  () => localStorage.setItem("manga_end_style", endTheme.value));
    if (endMessage) endMessage.addEventListener("input", () => localStorage.setItem("manga_end_message", endMessage.value));
    if (endShowDiscord) endShowDiscord.addEventListener("change", () => localStorage.setItem("manga_end_showdiscord", endShowDiscord.checked ? "1" : "0"));
    if (endColor)    endColor.addEventListener("input", () => localStorage.setItem("manga_end_color", endColor.value));
    if (endUseColor) endUseColor.addEventListener("change", () => localStorage.setItem("manga_end_usecolor", endUseColor.checked ? "1" : "0"));
  }
  if (endCardBtn) endCardBtn.addEventListener("click", async () => {
    endCardBtn.disabled = true;
    const label = endCardBtn.textContent;
    endCardBtn.textContent = "Making…";
    try {
      const f = new FormData();
      f.append("scanlation", (endScan && endScan.value.trim()) || "BorutoTBV Scanlations");
      // Discord is omitted entirely when the toggle is off (clean page).
      const showDiscord = !endShowDiscord || endShowDiscord.checked;
      f.append("show_discord", showDiscord ? "true" : "false");
      f.append("discord", showDiscord ? ((endDiscord && endDiscord.value.trim()) || "discord.gg/borutotbv") : "");
      if (endMessage && endMessage.value.trim()) f.append("footer", endMessage.value.trim());
      f.append("style", endTheme ? endTheme.value : "royal");
      if (endUseColor && endUseColor.checked && endColor) f.append("accent", endColor.value);
      // Match the chapter's page size so it blends in (fall back to a portrait default).
      const ref = pages.find(p => p.status === "done") || pages[0];
      const dim = await pageDimensions(ref);
      if (dim) { f.append("width", dim.w); f.append("height", dim.h); }
      const res = await fetch("/api/endcard", { method: "POST", body: f });
      if (!res.ok) {
        let msg = res.statusText; try { msg = (await res.json()).detail || msg; } catch (_) {}
        throw new Error(msg);
      }
      const taskId = (await res.json()).task_id;
      const page = {
        uid: ++uidCounter, name: "end-page.png", file: null, size: 0,
        thumb: `/api/result/${taskId}`, taskId, status: "done", progress: 100,
        step: 2, message: "End page ready!", result: { items: [] }, items: [],
        excluded: new Set(), erased: new Set(), glows: new Set(),
        offsets: {}, colors: {}, fontScales: {}, boxes: {}, error: "", rev: 0,
        isEndCard: true,
      };
      pages.push(page);                 // last page of the chapter
      activeUid = page.uid;
      showSection("result");
      renderStrip(); updateBatch(); renderActivePage();
    } catch (e) {
      showError("Couldn't make the end page: " + e.message);
    } finally {
      endCardBtn.disabled = false; endCardBtn.textContent = label;
    }
  });

  // Natural pixel size of a page's image (uploaded file or processed result).
  function pageDimensions(page) {
    return new Promise(resolve => {
      if (!page) return resolve(null);
      const src = (page.status === "done" && page.taskId)
        ? `/api/result/${page.taskId}?t=${page.rev}` : page.thumb;
      if (!src) return resolve(null);
      const im = new Image();
      im.onload  = () => resolve({ w: im.naturalWidth, h: im.naturalHeight });
      im.onerror = () => resolve(null);
      im.src = src;
    });
  }

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
