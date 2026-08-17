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
  const tileMode       = document.getElementById("tileMode");
  if (tileMode) {
    tileMode.value = localStorage.getItem("manga_tile_mode") || "1";
    tileMode.addEventListener("change", () => localStorage.setItem("manga_tile_mode", tileMode.value));
  }
  const protectDark = document.getElementById("protectDark");
  if (protectDark) {
    protectDark.checked = localStorage.getItem("manga_protect_dark") === "1";
    protectDark.addEventListener("change", () =>
      localStorage.setItem("manga_protect_dark", protectDark.checked ? "1" : "0"));
  }
  const gpuCap = document.getElementById("gpuCap");
  if (gpuCap) {
    gpuCap.value = localStorage.getItem("manga_gpu_cap") || "100";
    gpuCap.addEventListener("change", () =>
      localStorage.setItem("manga_gpu_cap", gpuCap.value));
  }
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

  // SBS / text-heavy mode: forces AI-vision detection (best for dense paragraph
  // pages) and surfaces the transcript.
  const oneByOne = document.getElementById("oneByOne");
  const webtoonMode = document.getElementById("webtoonMode");
  if (oneByOne) {
    oneByOne.checked = localStorage.getItem("manga_one_by_one") === "1";
    oneByOne.addEventListener("change", () =>
      localStorage.setItem("manga_one_by_one", oneByOne.checked ? "1" : "0"));
  }
  const sbsMode = document.getElementById("sbsMode");
  if (sbsMode) {
    sbsMode.checked = localStorage.getItem("manga_sbs") === "1";
    sbsMode.addEventListener("change", () => {
      localStorage.setItem("manga_sbs", sbsMode.checked ? "1" : "0");
      if (window.updateCutBtn) window.updateCutBtn();
    });
  }

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
        // Rolling "latest" aliases — Google keeps these pointed at the current
        // models, so they never hit the "no longer available" error. Best
        // default for a fresh install.
        { value: "gemini-flash-latest", text: "Gemini Flash (Latest — recommended)" },
        { value: "gemini-pro-latest", text: "Gemini Pro (Latest — best)" },
        { value: "gemini-flash-lite-latest", text: "Gemini Flash-Lite (Latest — cheapest)" },
        // Newest named models (may depend on account access).
        { value: "gemini-3-pro-preview", text: "Gemini 3 Pro (Newest)" },
        // Current stable 2.5 line.
        { value: "gemini-2.5-flash", text: "Gemini 2.5 Flash (Fast)" },
        { value: "gemini-2.5-flash-lite", text: "Gemini 2.5 Flash-Lite" },
        // Kept for existing accounts — newer Google accounts may 404 on these.
        { value: "gemini-2.5-pro",   text: "Gemini 2.5 Pro (older accounts)" },
        { value: "gemini-2.0-flash", text: "Gemini 2.0 Flash (older accounts)" },
      ],
    },
  };

  const getActive = () => pages.find(p => p.uid === activeUid) || null;
  const needsScan = wf => wf === "raw-scan-translate" || wf === "raw-scan" || wf === "scan-upscale";
  const isClean = wf => wf === "clean";
  const needsTranslate = wf => wf !== "raw-scan" && wf !== "upscale-only" && wf !== "scan-upscale" && wf !== "clean" && wf !== "scan-raw";
  const isUpscaleOnly = wf => wf === "upscale-only";
  const isRawify = wf => wf === "scan-raw";
  const rawStyle = document.getElementById("rawStyle");
  if (rawStyle) {
    rawStyle.value = localStorage.getItem("manga_raw_style") || "photo";
    rawStyle.addEventListener("change", () => localStorage.setItem("manga_raw_style", rawStyle.value));
  }
  const rawStrength = document.getElementById("rawStrength");
  const rawStrengthVal = document.getElementById("rawStrengthVal");
  const rawIntensityPanel = document.getElementById("rawIntensityPanel");
  if (rawStrength) {
    rawStrength.value = localStorage.getItem("manga_raw_strength") || "1";
    if (rawStrengthVal) rawStrengthVal.textContent = parseFloat(rawStrength.value).toFixed(1);
    rawStrength.addEventListener("input", () => {
      if (rawStrengthVal) rawStrengthVal.textContent = parseFloat(rawStrength.value).toFixed(1);
      localStorage.setItem("manga_raw_strength", rawStrength.value);
    });
  }

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
    if (rawIntensityPanel) rawIntensityPanel.style.display = isRawify(wf) ? "" : "none";
    // Keep the Settings panel always visible (it's collapsible) so the top menu
    // never disappears when switching to Raw → Scan / upscale workflows.
    document.querySelector(".settings-bar").style.display = "";
    goBtn.textContent = {
      "scan-translate": "Translate", "raw-scan-translate": "Enhance & Translate",
      "raw-translate": "Translate Raw", "raw-scan": "Enhance to Scan",
      "upscale-only": "Upscale to HD", "scan-upscale": "Enhance + Upscale",
      "scan-raw": "Make it Raw",
    }[wf] || "Go";
    localStorage.setItem("manga_workflow", wf);
    if (window.updateCutBtn) window.updateCutBtn();
  }
  wfCards.forEach(card => card.addEventListener("click", () => setWorkflow(card.dataset.wf)));
  setWorkflow(localStorage.getItem("manga_workflow") || "scan-translate");

  /* ══ ENGINE SWITCHING ══ */
  let engineInited = false;
  function setEngine(eng) {
    const cfg = ENGINE_CONFIG[eng];
    if (!cfg) return;
    // NOTE: don't save the key here on switch — engineSelect.value is already the
    // NEW engine, so this saved the current key onto the wrong engine (collapsing
    // both engines to one key → Claude got the Gemini key → 401). Each engine's
    // key is already persisted per-engine by the apiKeyInput 'input' handler.
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
        // Preview: render each option IN its own font (loaded lazily from /fonts).
        try {
          const fam = "mtfont_" + f.replace(/[^A-Za-z0-9]/g, "_");
          const face = new FontFace(fam, `url("/fonts/${encodeURIComponent(f)}")`);
          face.load().then(loaded => {
            document.fonts.add(loaded);
            opt.style.fontFamily = `"${fam}", inherit`;
            opt.style.fontSize = "1.05em";
          }).catch(() => {});
        } catch (_) { /* older browser — plain names still work */ }
      }
      const savedFont = localStorage.getItem("manga_font") || "";
      if (savedFont && [...fontSelect.options].some(o => o.value === savedFont)) {
        fontSelect.value = savedFont;
      }
      buildFontPicker();
    } catch (_) {}
  }

  // Native <select> popups are OS widgets and ignore web fonts, so options can
  // never preview their typeface. Replace it with a custom dropdown of real DOM
  // rows, each rendered IN its font; the hidden <select> stays the source of
  // truth (value + change events) so nothing else changes.
  function buildFontPicker() {
    const row = fontSelect.closest(".font-row");
    if (!row) return;
    const fam = v => 'mtfont_' + v.replace(/[^A-Za-z0-9]/g, "_");
    let wrap = document.getElementById("fontPicker");
    if (!wrap) {
      fontSelect.style.display = "none";
      wrap = document.createElement("div");
      wrap.id = "fontPicker";
      wrap.className = "font-picker";
      wrap.innerHTML = '<button type="button" class="font-picker-btn"></button>' +
                       '<div class="font-picker-list" style="display:none"></div>';
      row.insertBefore(wrap, fontSelect);
      const btn = wrap.querySelector(".font-picker-btn");
      const list = wrap.querySelector(".font-picker-list");
      btn.addEventListener("click", () => {
        list.style.display = list.style.display === "none" ? "" : "none";
      });
      document.addEventListener("pointerdown", e => {
        if (!wrap.contains(e.target)) list.style.display = "none";
      });
    }
    const btn = wrap.querySelector(".font-picker-btn");
    const list = wrap.querySelector(".font-picker-list");
    const syncBtn = () => {
      const o = fontSelect.options[fontSelect.selectedIndex];
      btn.textContent = o ? o.textContent : "Auto-detect";
      btn.style.fontFamily = (o && o.value) ? '"' + fam(o.value) + '", inherit' : "";
      list.querySelectorAll(".font-picker-item").forEach(x =>
        x.classList.toggle("active", x.dataset.value === fontSelect.value));
    };
    list.innerHTML = "";
    [...fontSelect.options].forEach(o => {
      const it = document.createElement("div");
      it.className = "font-picker-item";
      it.dataset.value = o.value;
      it.textContent = o.textContent;
      if (o.value) it.style.fontFamily = '"' + fam(o.value) + '", inherit';
      it.addEventListener("click", () => {
        fontSelect.value = it.dataset.value;
        fontSelect.dispatchEvent(new Event("change"));
        list.style.display = "none";
        syncBtn();
      });
      list.appendChild(it);
    });
    syncBtn();
    // re-sync the button once each face finishes loading
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(syncBtn);
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
    let defaultPrompt = "Restore this into a clean TCB-style black-and-white manga scan: pure white paper, solid black ink, sharp crisp lines, flattened and straightened, no creases or shadows. Keep all artwork, screentones and Japanese text exactly as drawn.";
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
      p === "local"  ? "API key — not needed (runs on your GPU)" :
      p === "openai" ? "OpenAI API Key" :
      p === "xai"    ? "xAI (Grok) API Key" : "Gemini API Key";
  }
  enhanceProvider.addEventListener("change", () => {
    // Each provider's key is already saved per-provider by the input handler
    // below; just remember the chosen provider and load ITS key/model. (Saving
    // here would overwrite the new provider's key with the old field value.)
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
  dropZone.addEventListener("drop", async e => {
    e.preventDefault(); dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length) addFiles(await expandFiles(e.dataTransfer.files));
  });
  fileInput.addEventListener("change", async () => {
    if (fileInput.files.length) addFiles(await expandFiles(fileInput.files));
    fileInput.value = "";
  });

  // Paste image(s) from the clipboard (Ctrl/Cmd+V) — e.g. a screenshot or a copied
  // page — straight into the queue. Ignored while typing in a text field.
  document.addEventListener("paste", async (e) => {
    const ae = document.activeElement;
    if (ae && (ae.tagName === "INPUT" || ae.tagName === "TEXTAREA" || ae.isContentEditable)) return;
    const items = (e.clipboardData && e.clipboardData.items) || [];
    const files = [];
    let n = 0;
    for (const it of items) {
      if (it.kind === "file" && it.type.startsWith("image/")) {
        const blob = it.getAsFile();
        if (blob) {
          n++;
          const ext = (it.type.split("/")[1] || "png").replace("jpeg", "jpg");
          files.push(new File([blob], blob.name || `pasted-${n}.${ext}`, { type: it.type }));
        }
      }
    }
    if (files.length) { e.preventDefault(); addFiles(files); }
  });

  // Expand any dropped/selected .zip into image Files (via /api/unzip), so a
  // whole chapter can be uploaded as a single zip. Non-zip images pass through.
  async function expandFiles(fileList) {
    const out = [];
    for (const f of [...fileList]) {
      const isZip = /\.zip$/i.test(f.name) || f.type === "application/zip"
                    || f.type === "application/x-zip-compressed";
      if (isZip) {
        try {
          const fd = new FormData(); fd.append("file", f);
          const res = await fetch("/api/unzip", { method: "POST", body: fd });
          if (!res.ok) { let m = res.statusText; try { m = (await res.json()).detail || m; } catch (_) {} throw new Error(m); }
          const data = await res.json();
          for (const im of (data.images || [])) {
            const bin = Uint8Array.from(atob(im.b64), c => c.charCodeAt(0));
            out.push(new File([bin], im.name, { type: im.type || "image/png" }));
          }
        } catch (e) {
          showError(`Couldn't read "${f.name}": ${e.message}`);
        }
      } else if (f.type.startsWith("image/")) {
        out.push(f);
      }
    }
    return out;
  }

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
        excluded: new Set(), erased: new Set(), glows: new Set(), fits: new Set(), offsets: {}, colors: {}, fontScales: {}, boxes: {}, error: "", rev: 0,
        cutRegions: [],
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
    if (window.updateCutBtn) window.updateCutBtn();
  }

  clearBtn.addEventListener("click", resetAll);

  /* ══ CUT INTO PIECES (SBS) ══ */
  (function initCutEditor() {
    const btn = document.getElementById("cutBtn");
    const modal = document.getElementById("cutModal");
    const stage = document.getElementById("cutStage");
    const img = document.getElementById("cutImg");
    const canvas = document.getElementById("cutCanvas");
    if (!btn || !modal || !stage || !img || !canvas) return;
    const ctx = canvas.getContext("2d");
    const countEl = document.getElementById("cutCount");
    const navEl = document.getElementById("cutNav");
    const pageLabel = document.getElementById("cutPageLabel");
    const cutAll = document.getElementById("cutAll");
    const hint = document.getElementById("cutHint");
    const MAX = 6;

    let tool = "box";
    let idx = 0;                 // which staged page is being cut
    let drawing = false, start = null, cur = null;
    let lasso = [];              // in-progress lasso points (normalised)

    // The Cut button shows for any translate workflow when SBS is on.
    function cutApplicable() {
      const sbs = sbsMode && sbsMode.checked;
      return !!sbs && needsTranslate(workflow);
    }
    window.updateCutBtn = function () {
      if (!btn) return;
      const show = cutApplicable() && pages.length > 0;
      btn.style.display = show ? "" : "none";
      if (show) {
        const n = pages.reduce((a, p) => a + ((p.cutRegions || []).length ? 1 : 0), 0);
        btn.textContent = n ? `✂ Cut pieces (${n})` : "✂ Cut pieces";
      }
    };

    function regions() { return pages[idx].cutRegions || (pages[idx].cutRegions = []); }
    function refreshCount() {
      countEl.textContent = `${regions().length} / ${MAX}`;
      const multi = pages.length > 1;
      navEl.style.display = multi ? "inline-flex" : "none";
      if (multi) pageLabel.textContent = `${idx + 1} / ${pages.length}`;
    }

    function loadPage(i) {
      idx = Math.max(0, Math.min(i, pages.length - 1));
      lasso = [];
      img.onload = () => { fitCanvas(); redraw(); refreshCount(); };
      img.src = pages[idx].thumb;
      if (img.complete && img.naturalWidth) { fitCanvas(); redraw(); refreshCount(); }
    }
    function fitCanvas() {
      // The image is centred in the stage (object-fit: contain), so align the
      // overlay canvas to the image's actual on-screen box, not the stage origin.
      canvas.width = img.clientWidth; canvas.height = img.clientHeight;
      canvas.style.width = img.clientWidth + "px";
      canvas.style.height = img.clientHeight + "px";
      canvas.style.left = img.offsetLeft + "px";
      canvas.style.top = img.offsetTop + "px";
    }
    function toNorm(ev) {
      const r = img.getBoundingClientRect();
      let x = (ev.clientX - r.left) / r.width;
      let y = (ev.clientY - r.top) / r.height;
      return [Math.max(0, Math.min(1, x)), Math.max(0, Math.min(1, y))];
    }
    function polyFor(a, b) {
      // Build a normalised polygon for the current tool from two drag points.
      const x0 = Math.min(a[0], b[0]), x1 = Math.max(a[0], b[0]);
      const y0 = Math.min(a[1], b[1]), y1 = Math.max(a[1], b[1]);
      if (tool === "hslice") return [[0, y0], [1, y0], [1, y1], [0, y1]];
      if (tool === "vslice") return [[x0, 0], [x0, 1], [x1, 1], [x1, 0]];
      return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];   // box
    }
    function redraw() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const W = canvas.width, H = canvas.height;
      const drawPoly = (poly, fill, stroke) => {
        if (poly.length < 2) return;
        ctx.beginPath();
        poly.forEach((p, i) => { const X = p[0] * W, Y = p[1] * H; i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
        ctx.closePath();
        ctx.fillStyle = fill; ctx.fill();
        ctx.lineWidth = 2; ctx.strokeStyle = stroke; ctx.stroke();
      };
      regions().forEach((poly, i) => {
        drawPoly(poly, "rgba(80,120,255,0.18)", "rgba(90,130,255,0.95)");
        const cx = poly.reduce((s, p) => s + p[0], 0) / poly.length * W;
        const cy = poly.reduce((s, p) => s + p[1], 0) / poly.length * H;
        ctx.fillStyle = "#fff"; ctx.font = "bold 15px sans-serif";
        ctx.textAlign = "center"; ctx.fillText(String(i + 1), cx, cy);
      });
      if (drawing && start && cur && tool !== "lasso") {
        drawPoly(polyFor(start, cur), "rgba(255,180,60,0.20)", "rgba(255,180,60,0.95)");
      }
      if (tool === "lasso" && lasso.length) {
        ctx.beginPath();
        lasso.forEach((p, i) => { const X = p[0] * W, Y = p[1] * H; i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
        ctx.lineWidth = 2; ctx.strokeStyle = "rgba(255,180,60,0.95)"; ctx.stroke();
        lasso.forEach(p => { ctx.beginPath(); ctx.arc(p[0] * W, p[1] * H, 4, 0, 7); ctx.fillStyle = "#ffb43c"; ctx.fill(); });
      }
    }
    function addRegion(poly) {
      if (regions().length >= MAX) { hint.textContent = `Max ${MAX} pieces per page — Undo to change.`; return; }
      // ignore too-tiny regions
      const xs = poly.map(p => p[0]), ys = poly.map(p => p[1]);
      if ((Math.max(...xs) - Math.min(...xs)) * (Math.max(...ys) - Math.min(...ys)) < 0.004) return;
      regions().push(poly); redraw(); refreshCount(); window.updateCutBtn();
    }

    // Pointer handling on the canvas' parent stage.
    stage.addEventListener("pointerdown", (e) => {
      if (e.target !== img && e.target !== canvas && e.target !== stage) return;
      const pt = toNorm(e);
      if (tool === "lasso") {
        if (lasso.length >= 3) {
          const f = lasso[0];
          if (Math.hypot(pt[0] - f[0], pt[1] - f[1]) < 0.03) { addRegion(lasso.slice()); lasso = []; redraw(); return; }
        }
        lasso.push(pt); redraw(); return;
      }
      drawing = true; start = pt; cur = pt; redraw();
    });
    stage.addEventListener("pointermove", (e) => {
      if (tool === "lasso") { if (lasso.length) { cur = toNorm(e); } return; }
      if (!drawing) return; cur = toNorm(e); redraw();
    });
    window.addEventListener("pointerup", (e) => {
      if (!drawing || tool === "lasso") { drawing = false; return; }
      drawing = false; cur = toNorm(e);
      if (start && cur) addRegion(polyFor(start, cur));
      start = cur = null;
    });

    document.querySelectorAll(".cut-tool").forEach(b => b.addEventListener("click", () => {
      document.querySelectorAll(".cut-tool").forEach(x => x.classList.remove("active"));
      b.classList.add("active"); tool = b.dataset.tool; lasso = []; redraw();
      hint.textContent = tool === "lasso"
        ? "Click points around a region; click near the first point to close."
        : "Drag to draw. Each region is translated on its own, then merged back.";
    }));
    document.getElementById("cutUndo").addEventListener("click", () => { regions().pop(); redraw(); refreshCount(); window.updateCutBtn(); });
    document.getElementById("cutClear").addEventListener("click", () => { pages[idx].cutRegions = []; lasso = []; redraw(); refreshCount(); window.updateCutBtn(); });
    document.getElementById("cutPrev").addEventListener("click", () => loadPage(idx - 1));
    document.getElementById("cutNext").addEventListener("click", () => loadPage(idx + 1));
    document.getElementById("cutDone").addEventListener("click", () => {
      if (cutAll && cutAll.checked) {
        const src = regions().map(p => p.map(q => q.slice()));
        pages.forEach(p => { p.cutRegions = src.map(poly => poly.map(q => q.slice())); });
      }
      modal.style.display = "none"; window.updateCutBtn();
    });
    btn.addEventListener("click", () => {
      if (!pages.length) return;
      modal.style.display = "flex";
      loadPage(0);
    });
    window.addEventListener("resize", () => { if (modal.style.display !== "none") { fitCanvas(); redraw(); } });
  })();

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

  async function startBatch() {
    if (!pages.length) return;
    if (needsTranslate(workflow) && !apiKeyInput.value.trim()) {
      apiKeyInput.focus(); apiKeyInput.style.borderColor = "#f87171"; return;
    }
    apiKeyInput.style.borderColor = "";
    if (needsScan(workflow) && enhanceProvider.value !== "local" && !enhanceKey.value.trim()) {
      enhancePanel.scrollIntoView({ behavior: "smooth" });
      enhanceKey.focus(); enhanceKey.style.borderColor = "#f87171"; return;
    }
    enhanceKey.style.borderColor = "";

    // Webtoon mode: stack every uploaded slice into ONE long strip first, then
    // run the normal flow on that single tall page.
    if (webtoonMode && webtoonMode.checked && pages.length >= 1) {
      const merged = await mergeWebtoonStrip();
      if (!merged) return;              // the error is already on screen
    }

    pages.forEach(p => { if (p.status === "pending") p.status = "queued"; });
    activeUid = pages[0].uid;
    showSection("result");
    renderStrip(); updateBatch(); renderActivePage();
    pump();
  }

  /* Webtoon: send every slice to the server, get one merged strip back, and
     collapse the page list down to that single long page. */
  async function mergeWebtoonStrip() {
    const slices = pages.filter(p => p.file);
    if (!slices.length) return false;
    goBtn.disabled = true;
    const label = goBtn.textContent;
    goBtn.textContent = `Merging ${slices.length} slice(s)...`;
    try {
      // Natural sort so 2 comes before 10 (plain sort puts 10 first).
      const coll = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });
      slices.sort((a, b) => coll.compare(a.name || "", b.name || ""));

      const fd = new FormData();
      slices.forEach(p => fd.append("files", p.file, p.name || "slice.png"));
      // Faithful stack: webtoon slices are exact cuts, so merging them
      // untouched reproduces the episode exactly.
      fd.append("trim_seams", "false");
      const res = await fetch("/api/merge-strip", { method: "POST", body: fd });
      if (!res.ok) {
        let msg = res.statusText;
        try { msg = (await res.json()).detail || msg; } catch (_) {}
        throw new Error(msg);
      }
      const info = await res.json();

      // Pull the merged strip back as a File so the rest of the flow (which
      // uploads a file per page) needs no special-casing at all.
      const blobRes = await fetch(info.url);
      if (!blobRes.ok) throw new Error("Could not read the merged strip");
      const blob = await blobRes.blob();
      const chapter = (chapterName && chapterName.value.trim()) || "webtoon";
      const file = new File([blob], `${chapter}.png`, { type: "image/png" });

      pages.forEach(p => { try { URL.revokeObjectURL(p.thumb); } catch (_) {} });
      pages.length = 0;
      pages.push({
        uid: ++uidCounter, file, name: file.name, size: blob.size,
        thumb: URL.createObjectURL(blob),
        status: "pending", progress: 0, step: 0, message: "", taskId: null,
        result: null, items: [], added: [], covers: [], rotations: {},
        excluded: new Set(), erased: new Set(), glows: new Set(), offsets: {},
        colors: {}, fontScales: {}, boxes: {}, error: "", rev: 0,
        cutRegions: [],
      });
      activeUid = pages[0].uid;
      console.log(`[webtoon] merged ${info.slices} slice(s) -> ${info.width}x${info.height}`);
      return true;
    } catch (e) {
      showError("Merge failed: " + e.message);
      return false;
    } finally {
      goBtn.disabled = false; goBtn.textContent = label;
    }
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
    const gpuCapEl = document.getElementById("gpuCap");
    if (gpuCapEl) f.append("gpu_cap", gpuCapEl.value);
    if (isClean(workflow)) {
      // Clean only: erase all text, no translation (no API key needed).
      f.append("clean_only", "true");
      f.append("font", fontSelect.value);
      f.append("finish", pageFinish ? pageFinish.value : "clean");
      f.append("upscale", hdUpscale && hdUpscale.checked ? "true" : "false");
      f.append("compress", compressOut && compressOut.checked ? "true" : "false");
      if (watermarkInput && watermarkInput.value.trim()) {
        f.append("watermark", watermarkInput.value.trim());
        if (wmPlace) f.append("wm_place", wmPlace.value);
        if (wmOpacity) f.append("wm_opacity", wmOpacity.value);
        if (wmSize) f.append("wm_size", wmSize.value);
        if (wmStyle) f.append("wm_style", wmStyle.value);
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
      // SBS / text-heavy pages need the AI-vision detector to catch paragraph text.
      f.append("smart_mode", (smartMode.checked || (sbsMode && sbsMode.checked)) ? "true" : "false");
      f.append("translate_sfx", translateSfx && translateSfx.checked ? "true" : "false");
      f.append("one_by_one", oneByOne && oneByOne.checked ? "true" : "false");
      f.append("webtoon", webtoonMode && webtoonMode.checked ? "true" : "false");
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
        if (wmPlace) f.append("wm_place", wmPlace.value);
        if (wmOpacity) f.append("wm_opacity", wmOpacity.value);
        if (wmSize) f.append("wm_size", wmSize.value);
        if (wmStyle) f.append("wm_style", wmStyle.value);
      }
      if (creditInput && creditInput.value.trim()) {
        f.append("credit", creditInput.value.trim());
      }
      if (profileSelect && profileSelect.value) {
        f.append("profile", profileSelect.value);
      }
      // SBS "cut into pieces": send this page's drawn regions, if any.
      const _pg = pages.find(p => p.file === file);
      if (_pg && _pg.cutRegions && _pg.cutRegions.length) {
        f.append("cut_regions", JSON.stringify(_pg.cutRegions));
      }
      return { url: "/api/translate", form: f };
    }
    if (isUpscaleOnly(workflow)) {
      // Faithful HD upscale, no translation and no API keys needed.
      appendWm(f);
      return { url: "/api/upscale", form: f };
    }
    if (isRawify(workflow)) {
      // Deterministic rough-raw effect: no models, no API keys.
      f.append("strength", rawStrength ? rawStrength.value : "1");
      f.append("style", rawStyle ? rawStyle.value : "photo");
      appendWm(f);
      return { url: "/api/rawify", form: f };
    }
    f.append("provider", enhanceProvider.value);
    f.append("api_key", enhanceKey.value.trim());
    f.append("prompt", enhancePrompt.value);
    f.append("model", enhanceModel.value);
    if (tileMode) f.append("tiles", tileMode.value);   // beta HD tile mode
    if (protectDark && protectDark.checked) f.append("protect_dark", "true");
    // "AI Scan → HD" chains MangaJaNai after the generative scan; the HD toggle
    // also forces the upscale stage on a plain "Raw → Scan" run.
    if (workflow === "scan-upscale" || (hdUpscale && hdUpscale.checked)) {
      f.append("upscale", "true");
    }
    appendWm(f);
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
            page.fits = new Set();
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
    const rawFx = document.getElementById("rawFxBtn");
    if (rawFx) {
      rawFx.classList.toggle("btn-primary", !!p.rawEffect);
      rawFx.classList.toggle("btn-ghost", !p.rawEffect);
    }
    // Result is image-only (no translation) for the scan / upscale workflows —
    // label the panes accordingly so it never says "Translated" when it didn't.
    const noTranslate = !needsTranslate(workflow);
    const leftLabel = (workflow === "raw-scan" || workflow === "scan-upscale") ? "Rough"
                    : workflow === "scan-raw" ? "Clean" : "Original";
    const rightLabel = workflow === "upscale-only" ? "Upscaled HD"
                     : workflow === "scan-upscale" ? "Scan + HD"
                     : workflow === "raw-scan" ? "Manga Scan"
                     : workflow === "scan-raw" ? "Raw feel"
                     : "Translated";
    const tabLabel = workflow === "raw-scan" ? "Scan"
                   : (workflow === "upscale-only" || workflow === "scan-upscale") ? "HD"
                   : workflow === "scan-raw" ? "Raw"
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
          <button class="tl-vert${(page.rotations || {})[it.id] === -90 ? " on" : ""}" title="VERTICAL: set this text sideways, reading bottom-to-top (for tall single-column text). Click again for normal horizontal." data-id="${it.id}">↕</button>
          <button class="tl-style" title="Copy this bubble's LOOK (size, colour, glow, tilt, fit) — then paste it onto the others" data-id="${it.id}">🎨</button>
          <button class="tl-fit${page.fits && page.fits.has(String(it.id)) ? " on" : ""}" title="FIT THE BOX: grow this text until it properly fills its box (long words are hyphenated so one word can't keep everything tiny). Click again to go back to the normal fit." data-id="${it.id}">⤢</button>
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
    el.querySelectorAll(".tl-vert").forEach(btn => {
      btn.addEventListener("click", () => {
        pushUndo(page);
        collectEdits(page);
        const id = btn.dataset.id;
        page.rotations = page.rotations || {};
        if (page.rotations[id] === -90) delete page.rotations[id];
        else page.rotations[id] = -90;
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
    el.querySelectorAll(".tl-style").forEach(btn => {
      btn.addEventListener("click", () => {
        const id = btn.dataset.id;
        copiedStyle = {
          scale: (page.fontScales || {})[id] || 1,
          color: (page.colors || {})[id] || "auto",
          glow: !!(page.glows && page.glows.has(id)),
          fit: !!(page.fits && page.fits.has(id)),
          rot: (page.rotations || {})[id],
        };
        showStyleBar(page);
      });
    });
    el.querySelectorAll(".tl-fit").forEach(btn => {
      btn.addEventListener("click", () => {
        pushUndo(page);
        collectEdits(page);
        const id = btn.dataset.id;
        page.fits = page.fits || new Set();
        if (page.fits.has(id)) page.fits.delete(id); else page.fits.add(id);
        btn.classList.toggle("on");
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

  // ── Transcript: a readable original → translation dump (great for SBS pages) ──
  function buildTranscript(page) {
    const items = (page && page.items || []).filter(it => (it.translation || "").trim());
    if (!items.length) return "";
    const head = `${page.name || "page"} — ${items.length} lines\n${"=".repeat(40)}\n\n`;
    return head + items.map((it, i) => {
      const o = (it.original || "").replace(/\s*\n\s*/g, " ").trim();
      const t = (it.translation || "").replace(/\s*\n\s*/g, " ").trim();
      return o ? `${i + 1}. ${o}\n   → ${t}` : `${i + 1}. ${t}`;
    }).join("\n\n");
  }
  const copyTranscriptBtn = document.getElementById("copyTranscript");
  const downloadTranscriptBtn = document.getElementById("downloadTranscript");
  if (copyTranscriptBtn) copyTranscriptBtn.addEventListener("click", async () => {
    const txt = buildTranscript(getActive());
    if (!txt) { showError("Nothing to copy yet — translate a page first."); return; }
    try {
      await navigator.clipboard.writeText(txt);
      const o = copyTranscriptBtn.textContent;
      copyTranscriptBtn.textContent = "✓ Copied";
      setTimeout(() => { copyTranscriptBtn.textContent = o; }, 1500);
    } catch (_) { showError("Couldn't copy — your browser blocked clipboard access."); }
  });
  if (downloadTranscriptBtn) downloadTranscriptBtn.addEventListener("click", () => {
    const page = getActive();
    const txt = buildTranscript(page);
    if (!txt) { showError("Nothing to download yet — translate a page first."); return; }
    const blob = new Blob([txt], { type: "text/plain;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = ((page.name || "page").replace(/\.[^.]+$/, "")) + "_translation.txt";
    a.click();
    URL.revokeObjectURL(a.href);
  });

  const orderBtn = document.getElementById("orderBtn");
  if (orderBtn) orderBtn.addEventListener("click", openReadingOrder);
  const findReplaceBtn = document.getElementById("findReplaceBtn");
  if (findReplaceBtn) findReplaceBtn.addEventListener("click", openFindReplace);

  const rawFxBtn = document.getElementById("rawFxBtn");
  if (rawFxBtn) rawFxBtn.addEventListener("click", () => {
    const page = getActive();
    if (!page || !page.taskId) return;
    page.rawEffect = !page.rawEffect;
    rawFxBtn.classList.toggle("btn-primary", !!page.rawEffect);
    rawFxBtn.classList.toggle("btn-ghost", !page.rawEffect);
    applyChanges(rawFxBtn);
  });

  applyBtn.addEventListener("click", () => applyChanges());
  async function applyChanges(btn, pageArg) {
    const page = pageArg || getActive();
    if (!page || !page.taskId) return;
    // Only harvest the on-screen textareas when they actually belong to this
    // page — a batch re-render (find & replace) walks pages that aren't
    // rendered, and reading another page's boxes would clobber the new text.
    if (!pageArg || pageArg.uid === activeUid) collectEdits(page);
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
          raw_effect: !!page.rawEffect,
          raw_strength: rawStrength ? parseFloat(rawStrength.value) : 1.0,
          raw_style: rawStyle ? rawStyle.value : "photo",
          glows: [...(page.glows || [])], fits: [...(page.fits || [])], edits,
          font_scale: parseFloat(fontScale.value),
          font_scales: page.fontScales || {},
          boxes: page.boxes || {},
          offsets: page.offsets || {},
          covers: page.covers || [],
          rotations: page.rotations || {},
          colors: page.colors || {},
          added: (page.added || []).map(a => ({ id: a.id, bbox: a.bbox, poly: a.poly || null, translation: a.translation })),
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
    "type-add": "Drag a box over the text, then TYPE the original yourself (type it how it sounds, or click the characters) — for text the reader can't make out.",
    "vert-add": "Drag a box over a TALL vertical text run — it's read & translated, and the English is set VERTICALLY (bottom-to-top). Edit, then Apply & Re-render.",
    keep: "Draw an outline around the page (like tracing its edge); everything OUTSIDE becomes white — removes carpet/floor/background. Then Apply & Re-render.",
    "pen-add": "CLICK points around the text to outline it (click the first point again or press Enter to close, Esc cancels). Only what's inside is read & translated, and the text stays inside your shape.",
    "tone-poly": "CLICK points around a hole in a toned area (click the first point again or Enter to close) — it's filled with matching screentone.",
    clone: "Click a CLEAN patch of art to copy from, then click or drag over the damage to paint it in. Bracket keys [ ] change the brush size.",
    line: "Click the START then the END of the line — it's redrawn straight. Shift keeps it horizontal/vertical; [ and ] change thickness.",
    "fill-poly": "CLICK points around the damaged area (click the first point again or press Enter to close, Esc cancels) — then pick the colour and it's flooded in.",
    restore: "Draw around a damaged spot — the ORIGINAL art comes back exactly as drawn (undoes content-aware cleaning there; original text returns too). Then Apply & Re-render.",
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
    // Series context: knowing the manga stops invented names/terms.
    const title = (mangaTitle && mangaTitle.value.trim())
      ? `SERIES: this page is from "${mangaTitle.value.trim()}". Use that manga's `
        + "canonical character names, place names, techniques and tone exactly "
        + "as known from the series — never invent or re-romanize them."
      : "";
    return [title, preset, gl, user].filter(Boolean).join("\n");
  }
  const mangaTitle = document.getElementById("mangaTitle");
  if (mangaTitle) {
    mangaTitle.value = localStorage.getItem("manga_series_title") || "";
    mangaTitle.addEventListener("input", () =>
      localStorage.setItem("manga_series_title", mangaTitle.value));
  }
  const wmPlace = document.getElementById("wmPlace");
  const wmOpacity = document.getElementById("wmOpacity");
  if (wmPlace) {
    wmPlace.value = localStorage.getItem("manga_wm_place") || "br";
    wmPlace.addEventListener("change", () => localStorage.setItem("manga_wm_place", wmPlace.value));
  }
  if (wmOpacity) {
    wmOpacity.value = localStorage.getItem("manga_wm_opacity") || "50";
    wmOpacity.addEventListener("change", () => localStorage.setItem("manga_wm_opacity", wmOpacity.value));
  }
  const wmSize = document.getElementById("wmSize");
  if (wmSize) {
    wmSize.value = localStorage.getItem("manga_wm_size") || "m";
    wmSize.addEventListener("change", () => localStorage.setItem("manga_wm_size", wmSize.value));
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
      fits: [...(page.fits || [])],
      glows: [...(page.glows || [])], offsets: page.offsets || {},
      colors: page.colors || {}, fontScales: page.fontScales || {},
      boxes: page.boxes || {}, covers: page.covers || [],
      rotations: JSON.parse(JSON.stringify(page.rotations || {})),
      added: (page.added || []).map(a => ({ ...a })),
      trans: (page.items || []).map(it => it.translation || ""),
    });
  }
  function _restore(page, str) {
    const s = JSON.parse(str);
    page.excluded = new Set(s.excluded); page.erased = new Set(s.erased);
    page.fits = new Set(s.fits || []);
    page.glows = new Set(s.glows); page.offsets = s.offsets; page.colors = s.colors;
    page.fontScales = s.fontScales; page.boxes = s.boxes; page.covers = s.covers;
    page.rotations = s.rotations || {};
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
    // Let the stamp/line tools drop any half-finished state on a tool switch.
    window.dispatchEvent(new CustomEvent("kaisuki:tool", { detail: tool }));
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
        svg.style.cssText = "position:absolute;inset:0;width:100%;height:100%;z-index:4;cursor:pointer;pointer-events:none";   // see setHit()
        const pg = document.createElementNS(NS, "polygon");
        pg.setAttribute("points", cb.keep_poly.map(p => `${p[0] / W * 100},${p[1] / H * 100}`).join(" "));
        pg.setAttribute("fill", "none");
        pg.setAttribute("stroke", "#2563eb");
        pg.setAttribute("stroke-width", "0.6");
        pg.setAttribute("stroke-dasharray", "2 1.5");
        pg.style.pointerEvents = "auto";     // only the shape itself is clickable
        svg.appendChild(pg);
        svg.addEventListener("click", () => { page.covers.splice(i, 1); buildOverlay(); });
        moveLayer.appendChild(svg);
        return;
      }
      if (cb && cb.restore_poly) {   // restore-original outline
        const NS = "http://www.w3.org/2000/svg";
        const svg = document.createElementNS(NS, "svg");
        svg.setAttribute("viewBox", "0 0 100 100");
        svg.setAttribute("preserveAspectRatio", "none");
        svg.style.cssText = "position:absolute;inset:0;width:100%;height:100%;z-index:4;cursor:pointer;pointer-events:none";   // see setHit()
        const pg = document.createElementNS(NS, "polygon");
        pg.setAttribute("points", cb.restore_poly.map(p => `${p[0] / W * 100},${p[1] / H * 100}`).join(" "));
        pg.setAttribute("fill", "rgba(245,158,11,.14)");
        pg.setAttribute("stroke", "#f59e0b");
        pg.setAttribute("stroke-width", "0.5");
        pg.setAttribute("stroke-dasharray", "2 1.5");
        pg.style.pointerEvents = "auto";     // only the shape itself is clickable
        svg.appendChild(pg);
        svg.addEventListener("click", () => { page.covers.splice(i, 1); buildOverlay(); });
        moveLayer.appendChild(svg);
        return;
      }
      if (cb && cb.clone) {   // clone-stamp dab
        const c = cb.clone;
        const d = document.createElement("div");
        d.className = "clone-dab";
        d.style.left = ((c.dst[0] - c.r) / W * 100) + "%";
        d.style.top = ((c.dst[1] - c.r) / H * 100) + "%";
        d.style.width = (2 * c.r / W * 100) + "%";
        d.style.height = (2 * c.r / H * 100) + "%";
        d.title = "Clone dab — click to remove";
        d.addEventListener("click", () => { page.covers.splice(i, 1); buildOverlay(); });
        moveLayer.appendChild(d);
        return;
      }
      if (cb && cb.line) {   // redrawn straight line
        const NS = "http://www.w3.org/2000/svg";
        const svg = document.createElementNS(NS, "svg");
        svg.setAttribute("viewBox", "0 0 100 100");
        svg.setAttribute("preserveAspectRatio", "none");
        svg.style.cssText = "position:absolute;inset:0;width:100%;height:100%;z-index:4;cursor:pointer;pointer-events:none";   // see setHit()
        const ln = document.createElementNS(NS, "line");
        ln.setAttribute("x1", cb.line[0][0] / W * 100);
        ln.setAttribute("y1", cb.line[0][1] / H * 100);
        ln.setAttribute("x2", cb.line[1][0] / W * 100);
        ln.setAttribute("y2", cb.line[1][1] / H * 100);
        ln.setAttribute("stroke", "#06b6d4");
        ln.setAttribute("stroke-width", "0.7");
        // A hairline is almost impossible to hit, so widen the CLICK target
        // without changing how the line looks.
        ln.style.pointerEvents = "stroke";
        ln.setAttribute("stroke-linecap", "round");
        const hit = ln.cloneNode();
        hit.setAttribute("stroke", "transparent");
        hit.setAttribute("stroke-width", "2.5");
        hit.style.pointerEvents = "stroke";
        svg.appendChild(ln);
        svg.appendChild(hit);
        svg.addEventListener("click", () => { page.covers.splice(i, 1); buildOverlay(); });
        moveLayer.appendChild(svg);
        return;
      }
      if (cb && cb.fill_poly) {   // redraw / bucket fill
        const NS = "http://www.w3.org/2000/svg";
        const svg = document.createElementNS(NS, "svg");
        svg.setAttribute("viewBox", "0 0 100 100");
        svg.setAttribute("preserveAspectRatio", "none");
        svg.style.cssText = "position:absolute;inset:0;width:100%;height:100%;z-index:4;cursor:pointer;pointer-events:none";   // see setHit()
        const pg = document.createElementNS(NS, "polygon");
        pg.setAttribute("points", cb.fill_poly.map(p => `${p[0] / W * 100},${p[1] / H * 100}`).join(" "));
        pg.setAttribute("fill", cb.tone ? "#64748b" : (cb.color || "#8b5cf6"));
        pg.setAttribute("fill-opacity", cb.tone ? ".55" : ".85");
        pg.setAttribute("stroke", "#8b5cf6");
        pg.setAttribute("stroke-width", "0.5");
        pg.style.pointerEvents = "auto";     // only the shape itself is clickable
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
        svg.style.cssText = "position:absolute;inset:0;width:100%;height:100%;z-index:4;cursor:pointer;pointer-events:none";   // see setHit()
        const pg = document.createElementNS(NS, "polygon");
        pg.setAttribute("points", cb.poly.map(p => `${p[0] / W * 100},${p[1] / H * 100}`).join(" "));
        pg.setAttribute("fill", "rgba(220,38,38,.18)");
        pg.setAttribute("stroke", "#dc2626");
        pg.setAttribute("stroke-width", "0.5");
        pg.style.pointerEvents = "auto";     // only the shape itself is clickable
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
    moveLayer.querySelectorAll(".tilt-ghost").forEach(g => g.remove());
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
      <div class="edit-pop-color">
        <span class="epop-clabel">Tilt&deg;</span>
        <input type="range" class="epop-rot" min="-80" max="80" step="1"
               value="${(page.rotations || {})[it.id] != null ? (page.rotations || {})[it.id] : (it.rotation || 0)}"
               style="flex:1" title="Drag to tilt — live preview on the page">
        <span class="epop-rot-val" style="min-width:34px;text-align:right"></span>
      </div>
      <div class="edit-pop-color">
        <span class="epop-clabel">Size</span>
        <input type="range" class="epop-fs" min="40" max="300" step="5"
               value="${Math.round(((page.fontScales || {})[it.id] || 1) * 100)}"
               style="flex:1" title="Drag to grow/shrink this text — live preview on the page">
        <span class="epop-fs-val" style="min-width:34px;text-align:right"></span>
      </div>
      <div class="edit-pop-row">
        <button class="btn btn-ghost btn-sm epop-remove">${isAdded ? "Delete" : "Skip"}</button>
        <button class="btn btn-ghost btn-sm epop-ime" title="Retype the ORIGINAL text with the built-in Japanese/Korean keyboard and re-translate it — for a bubble the reader got wrong">あ Retype</button>
        ${isAdded ? "" : `<button class="btn btn-ghost btn-sm epop-erase" title="Delete this translation AND wipe the bubble clean — empty bubble, no text">Empty</button>`}
        <span class="spacer"></span>
        <button class="btn btn-ghost btn-sm epop-cancel">Cancel</button>
        <button class="btn btn-primary btn-sm epop-save">Save</button>
      </div>`;
    pop.addEventListener("pointerdown", e => e.stopPropagation());
    moveLayer.appendChild(pop);
    // LIVE tilt: dragging the slider rotates a text ghost in place over the
    // artwork instantly — the server re-render happens once, on Save.
    const rotEl = pop.querySelector(".epop-rot");
    const rotVal = pop.querySelector(".epop-rot-val");
    const fsEl = pop.querySelector(".epop-fs");
    const fsVal = pop.querySelector(".epop-fs-val");
    if (rotEl) {
      const ghost = document.createElement("div");
      ghost.className = "tilt-ghost";
      ghost.style.left = (bx + off[0]) / W * 100 + "%";
      ghost.style.top = (by + off[1]) / H * 100 + "%";
      ghost.style.width = it.bbox[2] / W * 100 + "%";
      ghost.style.height = it.bbox[3] / H * 100 + "%";
      ghost.textContent = ((it.translation || "TILT") + "").replace(/\s+/g, " ");
      moveLayer.appendChild(ghost);
      const sync = () => {
        const v = parseFloat(rotEl.value) || 0;
        const fscale = fsEl ? (parseFloat(fsEl.value) || 100) / 100 : 1;
        ghost.style.transform = "rotate(" + v + "deg)";
        ghost.style.fontSize =
          Math.max(6, (it.bbox[3] / H) * moveLayer.clientHeight * 0.45 * fscale) + "px";
        if (rotVal) rotVal.textContent = v + "\u00B0";
        if (fsVal) fsVal.textContent = Math.round(fscale * 100) + "%";
      };
      rotEl.addEventListener("input", sync);
      if (fsEl) fsEl.addEventListener("input", sync);
      sync();
    }
    const ta = pop.querySelector(".edit-pop-text");
    ta.focus(); ta.setSelectionRange(ta.value.length, ta.value.length);
    pop.querySelector(".epop-cancel").addEventListener("click", closeEditor);
    pop.querySelector(".epop-save").addEventListener("click", () => {
      it.translation = ta.value;
      const rotSave = pop.querySelector(".epop-rot");
      if (rotSave) {
        const rv = parseFloat(rotSave.value);
        page.rotations = page.rotations || {};
        if (!isNaN(rv)) page.rotations[it.id] = rv;
      }
      const fsSave = pop.querySelector(".epop-fs");
      if (fsSave) {
        const fv = (parseFloat(fsSave.value) || 100) / 100;
        page.fontScales = page.fontScales || {};
        if (fv === 1.0) delete page.fontScales[it.id];
        else page.fontScales[it.id] = fv;
        const span = document.querySelector('.tl-fsv[data-id="' + it.id + '"]');
        if (span) span.textContent = Math.round(fv * 100) + "%";
      }
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
    const imeBtn = pop.querySelector(".epop-ime");
    if (imeBtn) imeBtn.addEventListener("click", () => {
      if (!window.KaisukiIME) return;
      window.KaisukiIME.open({
        title: "Retype the original text",
        text: (it.original || "").trim(),
        translation: (it.translation || "").trim(),
        sourceLang: sourceLang ? sourceLang.value : "Japanese",
        translate: translateTyped,
        onUse: ({ original, translation }) => {
          it.original = original;
          it.translation = translation;
          const box = pop.querySelector(".edit-pop-text");
          if (box) box.value = translation;
          const row = document.querySelector('.tl-edit[data-id="' + it.id + '"]');
          if (row) row.value = translation;
        },
      });
    });

    const eraseBtn = pop.querySelector(".epop-erase");
    if (eraseBtn) eraseBtn.addEventListener("click", () => {
      pushUndo(page);
      page.erased = page.erased || new Set();
      page.erased.add(String(it.id));
      page.excluded.delete(String(it.id));
      closeEditor();
      applyChanges(editApply);
    });
  }

  /* draw-a-box surface (Cover / Add tools) — bound once on the overlay */
  (function initDraw() {
    let drawing = false, sx, sy, rectEl = null;
    moveLayer.addEventListener("pointerdown", e => {
      if (tool !== "cover" && tool !== "add" && tool !== "vert-add"
          && tool !== "type-add") return;
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
      } else if (tool === "type-add") {
        typeTranslate(page, [x, y, w, h]);
      } else {
        autoTranslate(page, [x, y, w, h], null, tool === "vert-add");
      }
    };
    moveLayer.addEventListener("pointerup", finish);
    moveLayer.addEventListener("pointercancel", finish);
  })();

  /* ══ CLONE STAMP + LINE ══
     Clone: first click sets the SOURCE, then every click/drag paints a patch
     copied from there (the offset is locked on the first paint, so dragging
     tracks the source the way a real clone brush does).
     Line: click start, click end. */
  (function initStampLine() {
    let cloneSrc = null;        // [x,y] source point, image coords
    let offset = null;          // [dx,dy] locked on the first paint
    let painting = false;
    let lineStart = null;
    let brush = 34;             // radius, image px

    function imgPt(e) {
      const dims = curDims(); if (!dims) return null;
      const [W, H] = dims;
      const r = moveLayer.getBoundingClientRect();
      return [Math.round((e.clientX - r.left) / r.width * W),
              Math.round((e.clientY - r.top) / r.height * H)];
    }

    function stamp(page, at) {
      if (!cloneSrc) return;
      if (!offset) offset = [at[0] - cloneSrc[0], at[1] - cloneSrc[1]];
      page.covers = page.covers || [];
      page.covers.push({ clone: { src: [at[0] - offset[0], at[1] - offset[1]],
                                  dst: at, r: brush } });
    }

    moveLayer.addEventListener("pointerdown", e => {
      if (tool !== "clone" && tool !== "line") return;
      if (e.target !== moveLayer) return;
      const page = getActive(); const p = imgPt(e);
      if (!page || !p) return;

      if (tool === "line") {
        if (!lineStart) {
          lineStart = p;
          editHint.textContent = "Now click where the line should END.";
          return;
        }
        let end = p.slice();
        if (e.shiftKey) {           // snap to horizontal / vertical
          if (Math.abs(end[0] - lineStart[0]) > Math.abs(end[1] - lineStart[1]))
            end[1] = lineStart[1];
          else end[0] = lineStart[0];
        }
        pushUndo(page);
        page.covers = page.covers || [];
        page.covers.push({ line: [lineStart, end], width: lineWidth(),
                           color: lineColour(page, lineStart) });
        lineStart = null;
        buildOverlay();
        editHint.textContent = "Line added — hit Apply & Re-render.";
        return;
      }

      // CLONE
      if (!cloneSrc || e.altKey) {
        cloneSrc = p; offset = null;
        editHint.textContent = "Source set. Now click/drag over the damage to paint it in.";
        buildOverlay();
        return;
      }
      pushUndo(page);
      painting = true;
      stamp(page, p);
      try { moveLayer.setPointerCapture(e.pointerId); } catch (_) {}
      buildOverlay();
    });

    moveLayer.addEventListener("pointermove", e => {
      if (!painting || tool !== "clone") return;
      const page = getActive(); const p = imgPt(e);
      if (!page || !p) return;
      // Space the dabs out so a drag doesn't queue hundreds of covers.
      const last = (page.covers || []).filter(c => c && c.clone).pop();
      if (last && Math.hypot(last.clone.dst[0] - p[0],
                             last.clone.dst[1] - p[1]) < brush * 0.5) return;
      stamp(page, p);
      buildOverlay();
    });

    const endPaint = e => {
      if (!painting) return;
      painting = false;
      try { moveLayer.releasePointerCapture(e.pointerId); } catch (_) {}
      editHint.textContent = "Painted — hit Apply & Re-render (Alt-click to pick a new source).";
    };
    moveLayer.addEventListener("pointerup", endPaint);
    moveLayer.addEventListener("pointercancel", endPaint);

    document.addEventListener("keydown", e => {
      if (tool === "line") {
        if (e.key === "[") { lastWidth = Math.max(1, lastWidth - 1); editHint.textContent = `Line thickness ${lastWidth}px`; }
        if (e.key === "]") { lastWidth = Math.min(80, lastWidth + 1); editHint.textContent = `Line thickness ${lastWidth}px`; }
        if (e.key === "Escape") { lineStart = null; editHint.textContent = HINTS.line; }
        return;
      }
      if (tool !== "clone") return;
      if (e.key === "[") { brush = Math.max(6, brush - 6); editHint.textContent = `Brush ${brush}px`; }
      if (e.key === "]") { brush = Math.min(200, brush + 6); editHint.textContent = `Brush ${brush}px`; }
      if (e.key === "Escape") { cloneSrc = null; offset = null; buildOverlay(); }
    });

    // Reset the tools' state whenever the active tool changes.
    window.addEventListener("kaisuki:tool", () => {
      cloneSrc = null; offset = null; painting = false; lineStart = null;
    });

    let lastWidth = 5;
    function lineWidth() { return lastWidth; }

    function lineColour(page, at) {
      // The user clicks where the border is MISSING, so sampling that exact
      // pixel returns the blank they want to cover. Judge the SURROUNDINGS
      // instead: ink on a light page is black, on a dark panel it's white.
      let sum = 0, n = 0;
      for (let dx = -18; dx <= 18; dx += 6) {
        for (let dy = -18; dy <= 18; dy += 6) {
          const c = pixelAt(at[0] + dx, at[1] + dy);
          if (!c) continue;
          sum += parseInt(c.slice(1, 3), 16) + parseInt(c.slice(3, 5), 16)
               + parseInt(c.slice(5, 7), 16);
          n++;
        }
      }
      const mean = n ? sum / (3 * n) : 255;
      return mean < 110 ? "#ffffff" : "#000000";
    }

    window.__cloneSrc = () => cloneSrc;
  })();

  /* free-form lasso erase surface */
  (function initLasso() {
    const NS = "http://www.w3.org/2000/svg";
    let drawing = false, pts = [], svg = null, poly = null;
    moveLayer.addEventListener("pointerdown", e => {
      if (tool !== "lasso" && tool !== "lasso-add" && tool !== "keep" && tool !== "restore") return;
      if (e.target !== moveLayer) return;
      const page = getActive(); if (!page || !curDims()) return;
      drawing = true; pts = [];
      const add = tool === "lasso-add";
      const keep = tool === "keep";
      const restore = tool === "restore";
      svg = document.createElementNS(NS, "svg");
      svg.style.cssText = "position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:6";
      svg.setAttribute("viewBox", "0 0 100 100");
      svg.setAttribute("preserveAspectRatio", "none");
      poly = document.createElementNS(NS, "polyline");
      poly.setAttribute("fill", restore ? "rgba(245,158,11,.18)" : keep ? "rgba(37,99,235,.12)" : add ? "rgba(22,163,74,.18)" : "rgba(220,38,38,.18)");
      poly.setAttribute("stroke", restore ? "#f59e0b" : keep ? "#2563eb" : add ? "#16a34a" : "#dc2626");
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
      if (!page || !dims) { pts = []; return; }
      // Restore tool: a simple CLICK (no drag) un-deletes the damaged area
      // under the cursor — the server finds the changed blob automatically.
      if (tool === "restore" && pts.length >= 1) {
        const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
        if (Math.max(...xs) - Math.min(...xs) < 1.2 &&
            Math.max(...ys) - Math.min(...ys) < 1.2) {
          pushUndo(page);
          const [W0, H0] = dims;
          page.covers = page.covers || [];
          page.covers.push({ restore_click: [Math.round(pts[0][0] / 100 * W0),
                                             Math.round(pts[0][1] / 100 * H0)] });
          pts = [];
          buildOverlay();
          return;
        }
      }
      if (pts.length < 3) { pts = []; return; }
      pushUndo(page);
      const [W, H] = dims;
      const polyImg = pts.map(([px, py]) => [Math.round(px / 100 * W), Math.round(py / 100 * H)]);
      const wasAdd = tool === "lasso-add";
      const wasKeep = tool === "keep";
      const wasRestore = tool === "restore";
      pts = [];
      if (wasRestore) {
        // Restore eraser: original pixels come back inside this outline.
        page.covers = page.covers || [];
        page.covers.push({ restore_poly: polyImg });
        buildOverlay();
        return;
      }
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

  /* pen / point selection: click points, connect them into a custom shape */
  (function initPen() {
    const NS = "http://www.w3.org/2000/svg";
    let pts = [], svg = null, line = null;
    function reset() {
      if (svg) svg.remove();
      svg = null; line = null; pts = [];
    }
    function redraw() {
      if (!svg || !svg.isConnected) {
        if (svg) svg.remove();
        svg = document.createElementNS(NS, "svg");
        svg.style.cssText = "position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:6";
        svg.setAttribute("viewBox", "0 0 100 100");
        svg.setAttribute("preserveAspectRatio", "none");
        line = document.createElementNS(NS, "polyline");
        const fm = tool === "fill-poly" || tool === "tone-poly";
        line.setAttribute("fill", fm ? "rgba(139,92,246,.16)" : "rgba(22,163,74,.12)");
        line.setAttribute("stroke", fm ? "#8b5cf6" : "#16a34a");
        line.setAttribute("stroke-width", "0.15");
        svg.appendChild(line);
        moveLayer.appendChild(svg);
      }
      line.setAttribute("points", pts.map(p => p.join(",")).join(" "));
      svg.querySelectorAll("circle").forEach(c => c.remove());
      pts.forEach(([px, py], i) => {
        const c = document.createElementNS(NS, "circle");
        c.setAttribute("cx", px); c.setAttribute("cy", py);
        c.setAttribute("r", i === 0 ? "0.45" : "0.25");   // first point bigger = click it to close
        c.setAttribute("fill", i === 0 ? "#dc2626" : "#16a34a");
        svg.appendChild(c);
      });
    }
    function finish() {
      const page = getActive(), dims = curDims();
      const fillMode = tool === "fill-poly";
      const toneMode = tool === "tone-poly";
      const myPts = pts;
      reset();
      if (!page || !dims || myPts.length < 3) return;
      pushUndo(page);
      const [W, H] = dims;
      const polyImg = myPts.map(([px, py]) => [Math.round(px / 100 * W), Math.round(py / 100 * H)]);
      const xs = polyImg.map(p => p[0]), ys = polyImg.map(p => p[1]);
      const x = Math.min(...xs), y = Math.min(...ys);
      const w = Math.max(...xs) - x, h = Math.max(...ys) - y;
      page.covers = page.covers || [];
      if (toneMode) {
        // Screentone patch: density is measured from the art around it.
        page.covers.push({ fill_poly: polyImg, tone: true });
        buildOverlay();
        editHint.textContent = "Tone patch outlined — hit Apply & Re-render.";
        return;
      }
      if (fillMode) {
        // Redraw: ask which colour to flood the shape with (pre-sampled from
        // the art just outside it), then store it as a fill cover.
        openFillPicker(page, polyImg);
        return;
      }
      page.covers.push({ poly: polyImg });   // erase the original inside the shape
      if (w >= 6 && h >= 6) autoTranslate(page, [x, y, w, h], polyImg);
      else buildOverlay();
    }
    moveLayer.addEventListener("pointerdown", e => {
      // Eyedropper armed: this click samples a colour instead of adding a point.
      if (_eyedrop) {
        e.preventDefault(); e.stopPropagation();
        const r = moveLayer.getBoundingClientRect();
        const dims = curDims();
        if (dims) {
          const [W, H] = dims;
          _eyedrop(pixelAt((e.clientX - r.left) / r.width * W,
                           (e.clientY - r.top) / r.height * H));
        } else { _eyedrop(null); }
        return;
      }
      if ((tool !== "pen-add" && tool !== "fill-poly" && tool !== "tone-poly")
          || e.target !== moveLayer) return;
      const page = getActive(); if (!page || !curDims()) return;
      if (svg && !svg.isConnected) reset();   // overlay was rebuilt — stale points
      const r = moveLayer.getBoundingClientRect();
      const px = (e.clientX - r.left) / r.width * 100;
      const py = (e.clientY - r.top) / r.height * 100;
      if (pts.length >= 3 && Math.hypot(px - pts[0][0], py - pts[0][1]) < 2.2) {
        finish();   // clicked the first point — close the shape
        return;
      }
      pts.push([px, py]);
      redraw();
    });
    moveLayer.addEventListener("dblclick", () => {
      if ((tool === "pen-add" || tool === "fill-poly" || tool === "tone-poly")
          && pts.length >= 3) finish();
    });
    document.addEventListener("keydown", e => {
      if ((tool !== "pen-add" && tool !== "fill-poly" && tool !== "tone-poly")
          || !pts.length) return;
      if (e.key === "Enter") { e.preventDefault(); finish(); }
      if (e.key === "Escape") reset();
    });
  })();

  /* ══ MANUAL READING ORDER ══
     When the detector guesses the panel order wrong, the translations answer
     the wrong lines. Drag the bubbles into the order they should be READ, then
     re-translate: the model receives them in that order as one conversation,
     which is the context it needs to get the back-and-forth right. */
  function openReadingOrder() {
    const page = getActive();
    if (!page || !page.items || !page.items.length) {
      showError("Nothing to reorder on this page yet.");
      return;
    }
    let order = page.items.map(it => String(it.id));

    const back = document.createElement("div");
    back.className = "ro-back";
    back.innerHTML = `
      <div class="ro-box">
        <div class="ro-head"><strong>Reading order</strong>
          <button class="ro-x" title="Close">✕</button></div>
        <p class="ro-hint">Drag the lines into the order they should be read.
          Re-translating sends them in this order as one conversation, so each
          line answers the one before it.</p>
        <div class="ro-list"></div>
        <div class="ro-foot">
          <span class="ro-msg"></span><span style="flex:1"></span>
          <button class="btn btn-ghost btn-sm" id="roCancel">Cancel</button>
          <button class="btn btn-ghost btn-sm" id="roSave">Save order only</button>
          <button class="btn btn-primary btn-sm" id="roGo">Re-translate in this order</button>
        </div>
      </div>`;
    document.body.appendChild(back);
    const list = back.querySelector(".ro-list");

    function draw() {
      list.innerHTML = "";
      order.forEach((id, idx) => {
        const it = page.items.find(i => String(i.id) === id);
        if (!it) return;
        const row = document.createElement("div");
        row.className = "ro-row";
        row.draggable = true;
        row.dataset.id = id;
        row.innerHTML = `<span class="ro-n">${idx + 1}</span>
          <span class="ro-jp">${esc((it.original || "").slice(0, 22))}</span>
          <span class="ro-en">${esc((it.translation || "").slice(0, 46))}</span>
          <span class="ro-grip">⠿</span>`;
        row.addEventListener("dragstart", e => {
          e.dataTransfer.setData("text/plain", id);
          row.classList.add("dragging");
        });
        row.addEventListener("dragend", () => row.classList.remove("dragging"));
        row.addEventListener("dragover", e => e.preventDefault());
        row.addEventListener("drop", e => {
          e.preventDefault();
          const from = e.dataTransfer.getData("text/plain");
          if (!from || from === id) return;
          order = order.filter(x => x !== from);
          order.splice(order.indexOf(id), 0, from);
          draw();
        });
        // clicking a row flashes its bubble on the page
        row.addEventListener("click", () => {
          const b = document.querySelector('.ov-box[data-id="' + id + '"]');
          if (b) { b.classList.add("flash"); setTimeout(() => b.classList.remove("flash"), 700); }
        });
        list.appendChild(row);
      });
    }
    draw();

    const close = () => back.remove();
    back.querySelector(".ro-x").onclick = close;
    back.querySelector("#roCancel").onclick = close;

    const persist = () => {
      // Reorder page.items so every later step (rerender, transcript, export)
      // sees the order the user chose.
      const map = new Map(page.items.map(i => [String(i.id), i]));
      page.items = order.map(id => map.get(id)).filter(Boolean);
      page.readingOrder = order.slice();
    };

    back.querySelector("#roSave").onclick = () => {
      pushUndo(page); persist(); close();
      buildTranslationsList(page);
      editHint.textContent = "Order saved.";
    };

    back.querySelector("#roGo").onclick = async () => {
      const msg = back.querySelector(".ro-msg");
      if (!apiKeyInput.value.trim()) { msg.textContent = "Add your API key first."; return; }
      pushUndo(page); persist();
      const btn = back.querySelector("#roGo");
      btn.disabled = true; btn.textContent = "Re-translating…";
      try {
        const payload = page.items
          .filter(it => (it.original || "").trim())
          .map(it => ({ id: String(it.id), original: it.original }));
        if (!payload.length) throw new Error("No source text on this page to re-translate.");
        const res = await fetch(`/api/retranslate-ordered/${page.taskId}`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            items: payload,
            api_key: apiKeyInput.value.trim(),
            provider: engineSelect.value,
            model: modelSelect.value,
            target_lang: targetLang.value,
            source_lang: sourceLang ? sourceLang.value : "Japanese",
            style_prompt: styleText(),
          }),
        });
        if (!res.ok) {
          let m = res.statusText;
          try { m = (await res.json()).detail || m; } catch (_) {}
          throw new Error(m);
        }
        const data = await res.json();
        (data.translations || []).forEach(t => {
          const it = page.items.find(i => String(i.id) === String(t.id));
          if (it && (t.translation || "").trim()) it.translation = t.translation.trim();
        });
        close();
        buildTranslationsList(page);
        await applyChanges();
        editHint.textContent = "Re-translated in your reading order.";
      } catch (e) {
        msg.textContent = e.message;
        btn.disabled = false; btn.textContent = "Re-translate in this order";
      }
    };
  }

  /* ══ FIND & REPLACE ACROSS THE BATCH ══
     A name spelled two ways across twenty pages is tedious to chase by hand.
     Shows every match with its page and context BEFORE anything changes, then
     rewrites only what you confirm and re-renders just the pages that changed. */
  function openFindReplace() {
    const done = pages.filter(p => p.status === "done");
    if (!done.length) { showError("Translate some pages first."); return; }

    const back = document.createElement("div");
    back.className = "fr-back";
    back.innerHTML = `
      <div class="fr-box">
        <div class="fr-head"><strong>Find &amp; replace across ${done.length} page(s)</strong>
          <button class="fr-x" title="Close">✕</button></div>
        <div class="fr-row">
          <input type="text" id="frFind" placeholder="Find…" autocomplete="off">
          <input type="text" id="frRepl" placeholder="Replace with…" autocomplete="off">
        </div>
        <div class="fr-row fr-opts">
          <label><input type="checkbox" id="frCase"> Match case</label>
          <label><input type="checkbox" id="frWhole"> Whole words only</label>
          <span style="flex:1"></span>
          <span class="fr-count"></span>
        </div>
        <div class="fr-list"></div>
        <div class="fr-foot">
          <span class="fr-msg"></span><span style="flex:1"></span>
          <button class="btn btn-ghost btn-sm" id="frCancel">Cancel</button>
          <button class="btn btn-primary btn-sm" id="frGo" disabled>Replace &amp; re-render</button>
        </div>
      </div>`;
    document.body.appendChild(back);

    const find = back.querySelector("#frFind");
    const repl = back.querySelector("#frRepl");
    const list = back.querySelector(".fr-list");
    const count = back.querySelector(".fr-count");
    const go = back.querySelector("#frGo");
    let hits = [];

    const rx = () => {
      const term = find.value;
      if (!term) return null;
      const esc = term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      const body = back.querySelector("#frWhole").checked ? `\\b${esc}\\b` : esc;
      return new RegExp(body, back.querySelector("#frCase").checked ? "g" : "gi");
    };

    function scan() {
      hits = [];
      const r = rx();
      list.innerHTML = "";
      if (!r) { count.textContent = ""; go.disabled = true; return; }
      done.forEach(p => {
        const scanOne = (obj, kind) => {
          const t = obj.translation || "";
          r.lastIndex = 0;
          if (!r.test(t)) return;
          hits.push({ page: p, obj, kind });
        };
        (p.items || []).forEach(it => scanOne(it, "item"));
        (p.added || []).forEach(a => scanOne(a, "added"));
      });
      count.textContent = hits.length
        ? `${hits.length} bubble(s) match` : "no matches";
      go.disabled = !hits.length;
      hits.slice(0, 60).forEach(h => {
        const row = document.createElement("div");
        row.className = "fr-hit";
        const r2 = rx();
        const before = h.obj.translation;
        const after = before.replace(r2, repl.value);
        row.innerHTML = `<span class="fr-pg">${esc(h.page.name || "page")}</span>
          <span class="fr-b">${esc(before)}</span>
          <span class="fr-arrow">→</span>
          <span class="fr-a">${esc(after)}</span>`;
        list.appendChild(row);
      });
      if (hits.length > 60) {
        const more = document.createElement("div");
        more.className = "fr-more";
        more.textContent = `…and ${hits.length - 60} more`;
        list.appendChild(more);
      }
    }

    [find, repl].forEach(el => el.addEventListener("input", scan));
    back.querySelector("#frCase").addEventListener("change", scan);
    back.querySelector("#frWhole").addEventListener("change", scan);

    const close = () => back.remove();
    back.querySelector(".fr-x").onclick = close;
    back.querySelector("#frCancel").onclick = close;

    go.onclick = async () => {
      const touched = new Set();
      hits.forEach(h => {
        const r2 = rx();
        h.obj.translation = h.obj.translation.replace(r2, repl.value);
        touched.add(h.page);
      });
      close();
      // Re-render only the pages that actually changed.
      const list2 = [...touched];
      editHint.textContent = `Replaced on ${list2.length} page(s) — re-rendering…`;
      for (const p of list2) {
        try { await applyChanges(null, p); } catch (_) {}
      }
      renderStrip();
      renderActivePage();
      editHint.textContent = `Done — updated ${list2.length} page(s).`;
    };

    setTimeout(() => find.focus(), 30);
  }

  /* ══ COPY TYPESETTING ══
     Copy one bubble's look (size, colour, glow, fit, tilt) and stamp it onto
     the others — the fiddly part of matching a chapter's lettering by hand. */
  let copiedStyle = null;

  function applyStyle(page, id, st) {
    page.fontScales = page.fontScales || {};
    page.colors = page.colors || {};
    page.glows = page.glows || new Set();
    page.fits = page.fits || new Set();
    page.rotations = page.rotations || {};
    if (st.scale && st.scale !== 1) page.fontScales[id] = st.scale;
    else delete page.fontScales[id];
    if (st.color && st.color !== "auto") page.colors[id] = st.color;
    else delete page.colors[id];
    if (st.glow) page.glows.add(id); else page.glows.delete(id);
    if (st.fit) page.fits.add(id); else page.fits.delete(id);
    if (st.rot !== undefined && st.rot !== null) page.rotations[id] = st.rot;
    else delete page.rotations[id];
  }

  function showStyleBar(page) {
    let bar = document.getElementById("styleBar");
    if (!bar) {
      bar = document.createElement("div");
      bar.id = "styleBar";
      bar.className = "style-bar";
      document.body.appendChild(bar);
    }
    const st = copiedStyle;
    bar.innerHTML = `
      <span class="sb-txt">Copied look: <b>${Math.round((st.scale || 1) * 100)}%</b>
        · ${st.color || "auto"}${st.glow ? " · glow" : ""}${st.fit ? " · fit" : ""}${st.rot !== undefined && st.rot !== null ? " · " + st.rot + "&deg;" : ""}</span>
      <button class="btn btn-ghost btn-sm" id="sbAll">Paste to all on this page</button>
      <button class="btn btn-ghost btn-sm" id="sbBubbles">Only speech bubbles</button>
      <button class="btn btn-ghost btn-sm" id="sbChapter">Every page in the batch</button>
      <button class="btn btn-ghost btn-sm" id="sbClose">✕</button>`;
    bar.style.display = "flex";

    const paste = (pgs, onlyBubbles) => {
      pgs.forEach(pg => {
        pushUndo(pg);
        (pg.items || []).forEach(it => {
          if (onlyBubbles && it.in_bubble === false) return;
          applyStyle(pg, String(it.id), st);
        });
        (pg.added || []).forEach(a => {
          if (onlyBubbles) return;
          applyStyle(pg, String(a.id), st);
        });
      });
      buildTranslationsList(page);
      bar.style.display = "none";
      editHint.textContent = "Look pasted — hit Apply & Re-render.";
    };
    bar.querySelector("#sbAll").onclick = () => paste([page], false);
    bar.querySelector("#sbBubbles").onclick = () => paste([page], true);
    bar.querySelector("#sbChapter").onclick = () =>
      paste(pages.filter(p => p.status === "done"), false);
    bar.querySelector("#sbClose").onclick = () => { bar.style.display = "none"; };
  }

  /* ══ REDRAW FILL: sample a colour, flood the outlined shape ══
     Reads pixels straight off the rendered page (same origin, so the canvas
     isn't tainted), so both the auto-sample and the eyedropper work without a
     round trip to the server. */
  let _fillCanvas = null;

  function pageCanvas() {
    // transFull is the full-resolution page already loaded for the editor.
    if (!transFull || !transFull.naturalWidth) return null;
    if (_fillCanvas && _fillCanvas._src === transFull.src
        && _fillCanvas.width === transFull.naturalWidth) return _fillCanvas;
    const c = document.createElement("canvas");
    c.width = transFull.naturalWidth;
    c.height = transFull.naturalHeight;
    try {
      c.getContext("2d").drawImage(transFull, 0, 0);
    } catch (_) { return null; }
    c._src = transFull.src;
    _fillCanvas = c;
    return c;
  }

  const hex2 = v => Math.max(0, Math.min(255, v | 0)).toString(16).padStart(2, "0");
  const toHex = (r, g, b) => "#" + hex2(r) + hex2(g) + hex2(b);

  function pixelAt(x, y) {
    const c = pageCanvas(); if (!c) return null;
    x = Math.max(0, Math.min(c.width - 1, Math.round(x)));
    y = Math.max(0, Math.min(c.height - 1, Math.round(y)));
    try {
      const d = c.getContext("2d").getImageData(x, y, 1, 1).data;
      return toHex(d[0], d[1], d[2]);
    } catch (_) { return null; }
  }

  /* Median colour of a ring just OUTSIDE the outline = the background the
     shape sits in. Median (not mean) so a stray dark line in the ring can't
     drag the sample grey. */
  function sampleAround(poly) {
    const c = pageCanvas(); if (!c) return "#ffffff";
    const xs = poly.map(p => p[0]), ys = poly.map(p => p[1]);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
    const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    const ctx = c.getContext("2d");
    const R = [], G = [], B = [];
    for (const [px, py] of poly) {
      // step ~10px outward from each vertex, away from the centre
      const dx = px - cx, dy = py - cy;
      const len = Math.hypot(dx, dy) || 1;
      const sx = Math.round(px + (dx / len) * 10);
      const sy = Math.round(py + (dy / len) * 10);
      if (sx < 0 || sy < 0 || sx >= c.width || sy >= c.height) continue;
      try {
        const d = ctx.getImageData(sx, sy, 1, 1).data;
        R.push(d[0]); G.push(d[1]); B.push(d[2]);
      } catch (_) { return "#ffffff"; }
    }
    if (!R.length) return "#ffffff";
    const med = a => a.sort((x, y) => x - y)[Math.floor(a.length / 2)];
    return toHex(med(R), med(G), med(B));
  }

  let _eyedrop = null;   // active picker callback while sampling from the page

  function openFillPicker(page, poly) {
    const suggested = sampleAround(poly);
    let colour = suggested;

    const back = document.createElement("div");
    back.className = "fill-back";
    back.innerHTML = `
      <div class="fill-box">
        <div class="fill-head"><strong>Redraw this area</strong></div>
        <p class="fill-hint">The colour was sampled from the art around your
          outline. Use the dropper to take it from anywhere on the page, or set
          it by hand.</p>
        <div class="fill-row">
          <span class="fill-sw" id="fillSw"></span>
          <input type="color" id="fillPick" value="${suggested}">
          <input type="text" id="fillHex" value="${suggested}" spellcheck="false">
          <button class="btn btn-ghost btn-sm" id="fillDrop" title="Click the page to sample a colour">Dropper</button>
        </div>
        <div class="fill-row">
          <button class="btn btn-ghost btn-sm fill-preset" data-c="${suggested}">Sampled</button>
          <button class="btn btn-ghost btn-sm fill-preset" data-c="#ffffff">White</button>
          <button class="btn btn-ghost btn-sm fill-preset" data-c="#000000">Black</button>
        </div>
        <div class="fill-foot">
          <span style="flex:1"></span>
          <button class="btn btn-ghost btn-sm" id="fillCancel">Cancel</button>
          <button class="btn btn-primary btn-sm" id="fillOk">Fill it</button>
        </div>
      </div>`;
    document.body.appendChild(back);

    const sw = back.querySelector("#fillSw");
    const pick = back.querySelector("#fillPick");
    const hex = back.querySelector("#fillHex");
    const setColour = v => {
      if (!/^#[0-9a-fA-F]{6}$/.test(v)) return;
      colour = v.toLowerCase();
      sw.style.background = colour;
      pick.value = colour; hex.value = colour;
    };
    setColour(suggested);

    pick.addEventListener("input", () => setColour(pick.value));
    hex.addEventListener("change", () => setColour(hex.value.trim()));
    back.querySelectorAll(".fill-preset").forEach(
      b => b.addEventListener("click", () => setColour(b.dataset.c)));

    const close = () => { _eyedrop = null; back.remove(); };
    back.querySelector("#fillCancel").addEventListener("click", () => {
      close(); buildOverlay();
    });
    back.querySelector("#fillOk").addEventListener("click", () => {
      page.covers = page.covers || [];
      page.covers.push({ fill_poly: poly, color: colour });
      close();
      buildOverlay();
      editHint.textContent = "Area outlined — hit Apply & Re-render to paint it.";
    });
    back.querySelector("#fillDrop").addEventListener("click", () => {
      // Hide the dialog, let the next click on the page pick the colour.
      back.style.display = "none";
      editHint.textContent = "Click anywhere on the page to take that colour…";
      _eyedrop = c => {
        back.style.display = "";
        if (c) setColour(c);
        _eyedrop = null;
        editHint.textContent = HINTS["fill-poly"];
      };
    });
  }

  /* Translate SOURCE text the user typed on the built-in keyboard. */
  async function translateTyped(text) {
    const resp = await fetch("/api/translate-text", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        api_key: apiKeyInput.value.trim(),
        provider: engineSelect.value,
        model: modelSelect.value,
        target_lang: targetLang.value,
        source_lang: sourceLang ? sourceLang.value : "Japanese",
        style_prompt: styleText(),
      }),
    });
    if (!resp.ok) {
      let m = resp.statusText;
      try { m = (await resp.json()).detail || m; } catch (_) {}
      throw new Error(m);
    }
    return (await resp.json()).translation || "";
  }

  /* Place a region whose ORIGINAL text the user keys in by hand. */
  function typeTranslate(page, bbox, poly, vertical) {
    if (!window.KaisukiIME) { autoTranslate(page, bbox, poly, vertical); return; }
    editHint.textContent = "Type the original text, then place it.";
    window.KaisukiIME.open({
      title: "Type the original text",
      sourceLang: sourceLang ? sourceLang.value : "Japanese",
      translate: translateTyped,
      onCancel: () => { editHint.textContent = HINTS[tool] || HINTS.add; },
      onUse: ({ original, translation }) => {
        placeAdded(page, bbox, poly, original, translation, vertical);
      },
    });
  }

  /* Shared placement used by every add path. */
  function placeAdded(page, bbox, poly, original, translation, vertical) {
    page.added = page.added || [];
    page.addSeq = (page.addSeq || 0) + 1;
    const nid = "m" + page.addSeq;
    page.added.push({
      id: nid, bbox, poly: poly || null,
      original: original || "", translation: translation.trim(), placed: true,
    });
    if (vertical) {
      page.rotations = page.rotations || {};
      page.rotations[nid] = -90;
    }
    buildOverlay();
    editHint.textContent = "Added! Hit Apply & Re-render when ready.";
  }

  async function autoTranslate(page, bbox, poly, vertical) {
    editHint.textContent = "Reading & translating…";
    let data = { original: "", translation: "" };
    try {
      const resp = await fetch(`/api/ocr-translate/${page.taskId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          bbox,
          poly: poly || null,   // pen outline: OCR reads ONLY inside the shape
          api_key: apiKeyInput.value.trim(),
          provider: engineSelect.value,
          model: modelSelect.value,
          target_lang: targetLang.value,
          style_prompt: styleText(),
        }),
      });
      if (resp.ok) data = await resp.json();
    } catch (_) { /* fall through to manual entry */ }

    // Nothing readable here: hand over to the built-in keyboard so the user
    // can key the ORIGINAL in themselves (no system IME needed) and get a
    // proper translation — far better than guessing at the English.
    const suggested = (data.translation || "").trim();
    if (!suggested && window.KaisukiIME) {
      window.KaisukiIME.open({
        title: "Couldn't read this — type the original text",
        text: (data.original || "").trim(),
        sourceLang: sourceLang ? sourceLang.value : "Japanese",
        translate: translateTyped,
        onCancel: () => { editHint.textContent = HINTS[tool] || HINTS.add; },
        onUse: ({ original, translation }) => {
          placeAdded(page, bbox, poly, original, translation, vertical);
        },
      });
      return;
    }
    const label = "Edit translation (read: " + data.original + "):";
    const txt = prompt(label, suggested);
    if (txt && txt.trim()) {
      placeAdded(page, bbox, poly, data.original || "", txt, vertical);
    } else {
      editHint.textContent = HINTS[tool] || HINTS.add;
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
    const chN = (chapterName && chapterName.value.trim()) || "";
    const stem = (p.name || "page.png").replace(/\.[^.]+$/, "");
    a.download = (chN ? chN + " - " + stem : "translated_" + stem) + ".png";
    a.click();
  });

  const wmStyle = document.getElementById("wmStyle");
  const wmPreview = document.getElementById("wmPreview");
  if (wmStyle) {
    wmStyle.value = localStorage.getItem("manga_wm_style") || "clean";
    wmStyle.addEventListener("change", () => localStorage.setItem("manga_wm_style", wmStyle.value));
  }
  let _wmPrevTimer = null;
  function refreshWmPreview() {
    if (!wmPreview) return;
    const txt = (watermarkInput && watermarkInput.value.trim()) || "";
    if (!txt) { wmPreview.style.display = "none"; return; }
    clearTimeout(_wmPrevTimer);
    _wmPrevTimer = setTimeout(() => {
      const q = new URLSearchParams({
        text: txt,
        style: wmStyle ? wmStyle.value : "clean",
        place: wmPlace ? wmPlace.value : "br",
        opacity: wmOpacity ? wmOpacity.value : 70,
        size: wmSize ? wmSize.value : "m",
        credit: (creditInput && creditInput.value.trim()) || "",
      });
      wmPreview.src = "/api/wm-preview?" + q.toString() + "&t=" + Date.now();
      wmPreview.style.display = "";
    }, 350);
  }
  [watermarkInput, wmStyle, document.getElementById("wmPlace"),
   document.getElementById("wmSize"), document.getElementById("wmOpacity"),
   creditInput].forEach(el => {
    if (!el) return;
    el.addEventListener("input", refreshWmPreview);
    el.addEventListener("change", refreshWmPreview);
  });
  setTimeout(refreshWmPreview, 400);
  const wmAllOut = document.getElementById("wmAllOut");
  if (wmAllOut) {
    wmAllOut.checked = localStorage.getItem("manga_wm_all") === "1";
    wmAllOut.addEventListener("change", () =>
      localStorage.setItem("manga_wm_all", wmAllOut.checked ? "1" : "0"));
  }
  function appendWm(f) {
    // watermark + credit for non-translate outputs, when the master toggle is on
    if (!(wmAllOut && wmAllOut.checked)) return;
    if (watermarkInput && watermarkInput.value.trim()) {
      f.append("watermark", watermarkInput.value.trim());
      if (wmPlace) f.append("wm_place", wmPlace.value);
      if (wmOpacity) f.append("wm_opacity", wmOpacity.value);
      if (wmSize) f.append("wm_size", wmSize.value);
      if (wmStyle) f.append("wm_style", wmStyle.value);
    }
    if (creditInput && creditInput.value.trim()) f.append("credit", creditInput.value.trim());
  }
  const chapterName = document.getElementById("chapterName");
  if (chapterName) {
    chapterName.value = localStorage.getItem("manga_chapter_name") || "";
    chapterName.addEventListener("input", () =>
      localStorage.setItem("manga_chapter_name", chapterName.value));
  }
  const stampBtn = document.getElementById("stampBtn");
  if (stampBtn) stampBtn.addEventListener("click", async () => {
    const p = getActive();
    if (!p || p.status !== "done") return;
    const wmk = (watermarkInput && watermarkInput.value.trim()) || "";
    const cr = (creditInput && creditInput.value.trim()) || "";
    if (!wmk && !cr) { showError("Type a watermark (or credit) in Settings first."); return; }
    stampBtn.disabled = true;
    try {
      const res = await fetch(`/api/stamp/${p.taskId}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          watermark: wmk, credit: cr,
          wm_place: wmPlace ? wmPlace.value : "br",
          wm_opacity: wmOpacity ? wmOpacity.value : 50,
          wm_size: wmSize ? wmSize.value : "m",
          wm_style: wmStyle ? wmStyle.value : "clean",
        }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      p.rev = (p.rev || 0) + 1;
      renderActivePage();
    } catch (e) { showError(e.message); }
    finally { stampBtn.disabled = false; }
  });

  zipBtn.addEventListener("click", async () => {
    const ids = pages.filter(p => p.status === "done").map(p => p.taskId);
    if (!ids.length) return;
    const chName = (chapterName && chapterName.value.trim()) || "";
    zipBtn.disabled = true; zipBtn.textContent = "Zipping...";
    try {
      const res = await fetch("/api/zip", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task_ids: ids, name: chName }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const blob = await res.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = (chName ? chName : "translated_pages") + ".zip";
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
      p.items = []; p.excluded = new Set(); p.erased = new Set(); p.glows = new Set(); p.fits = new Set(); p.offsets = {}; p.colors = {}; p.fontScales = {}; p.boxes = {};
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
    if (endColor)    endColor.value = localStorage.getItem("manga_end_color") || "#d4af5a";
    if (endUseColor) endUseColor.checked = localStorage.getItem("manga_end_usecolor") === "1";
    endScan.addEventListener("input",    () => localStorage.setItem("manga_end_scan", endScan.value));
    endDiscord.addEventListener("input", () => localStorage.setItem("manga_end_discord", endDiscord.value));
    endTheme.addEventListener("change",  () => localStorage.setItem("manga_end_style", endTheme.value));
    if (endMessage) endMessage.addEventListener("input", () => localStorage.setItem("manga_end_message", endMessage.value));
    if (endColor)    endColor.addEventListener("input", () => localStorage.setItem("manga_end_color", endColor.value));
    if (endUseColor) endUseColor.addEventListener("change", () => localStorage.setItem("manga_end_usecolor", endUseColor.checked ? "1" : "0"));
  }
  if (endCardBtn) endCardBtn.addEventListener("click", async () => {
    endCardBtn.disabled = true;
    const label = endCardBtn.textContent;
    endCardBtn.textContent = "Making…";
    try {
      const f = new FormData();
      f.append("scanlation", (endScan && endScan.value.trim()) || "Kaisuki");
      // The link shows when you type one, and is omitted when the field is
      // blank — no toggle, no default. Whatever you enter is exactly what
      // appears, and the field persists between sessions so you set it once.
      const link = (endDiscord && endDiscord.value.trim()) || "";
      f.append("discord", link);
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
