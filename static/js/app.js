document.addEventListener("DOMContentLoaded", () => {

  /* ── DOM refs ── */
  const dropZone       = document.getElementById("dropZone");
  const fileInput      = document.getElementById("fileInput");
  const previewRow     = document.getElementById("previewRow");
  const previewImg     = document.getElementById("previewImg");
  const fileName       = document.getElementById("fileName");
  const fileSize       = document.getElementById("fileSize");
  const translateBtn   = document.getElementById("translateBtn");
  const clearBtn       = document.getElementById("clearBtn");
  const apiKeyInput    = document.getElementById("apiKey");
  const targetLang     = document.getElementById("targetLang");
  const modelSelect    = document.getElementById("model");
  const smartMode      = document.getElementById("smartMode");
  const fontSelect     = document.getElementById("fontSelect");
  const fontUpload     = document.getElementById("fontUpload");

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

  let selectedFile = null;
  let currentTaskId = null;
  let pollTimer = null;

  /* ── Persist API key ── */
  const saved = localStorage.getItem("manga_api_key");
  if (saved) apiKeyInput.value = saved;
  apiKeyInput.addEventListener("change", () =>
    localStorage.setItem("manga_api_key", apiKeyInput.value)
  );

  /* ── Fonts ── */
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

  /* ── Drag & drop ── */
  dropZone.addEventListener("click", () => fileInput.click());

  dropZone.addEventListener("dragover", e => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });
  dropZone.addEventListener("dragleave", () =>
    dropZone.classList.remove("drag-over")
  );
  dropZone.addEventListener("drop", e => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
  });

  function handleFile(file) {
    if (!file.type.startsWith("image/")) return;
    selectedFile = file;
    const url = URL.createObjectURL(file);
    previewImg.src = url;
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

  /* ── Translate ── */
  translateBtn.addEventListener("click", startTranslation);

  async function startTranslation() {
    const key = apiKeyInput.value.trim();
    if (!key) { apiKeyInput.focus(); apiKeyInput.style.borderColor = "#f87171"; return; }
    apiKeyInput.style.borderColor = "";
    if (!selectedFile) return;

    showSection("processing");
    resetProgress();

    const form = new FormData();
    form.append("file", selectedFile);
    form.append("api_key", key);
    form.append("target_lang", targetLang.value);
    form.append("model", modelSelect.value);
    form.append("smart_mode", smartMode.checked ? "true" : "false");
    form.append("font", fontSelect.value);

    try {
      const res = await fetch("/api/translate", { method: "POST", body: form });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
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

  /* ── Progress ── */
  function resetProgress() {
    progressFill.style.width = "0%";
    progressMsg.textContent = "Starting...";
    document.querySelectorAll(".step").forEach(s => {
      s.classList.remove("active", "done");
    });
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

  /* ── Result ── */
  function showResult(data) {
    const origUrl  = data.original_url;
    const transUrl = data.output_url;

    document.getElementById("origImg").src = origUrl;
    document.getElementById("transImg").src = transUrl;
    document.getElementById("origFull").src = origUrl;
    document.getElementById("transFull").src = transUrl;

    buildTranslationsList(data.result);
    showSection("result");
    initComparison();
  }

  function buildTranslationsList(result) {
    const el = document.getElementById("translationsList");
    el.innerHTML = "";
    if (!result || !result.translations) return;

    const entries = Object.entries(result.translations);
    if (entries.length === 0) {
      el.innerHTML = '<p style="color:var(--text-dim)">No translations available</p>';
      return;
    }

    for (const [id, t] of entries) {
      const div = document.createElement("div");
      div.className = "tl-item";
      div.innerHTML = `
        <div class="tl-header">
          <span class="tl-id">Region ${id}</span>
          <span class="tl-type">${t.type || "dialogue"}</span>
        </div>
        <div class="tl-original">${esc(t.original || "")}</div>
        <div class="tl-translation">${esc(t.translation || "")}</div>
      `;
      el.appendChild(div);
    }
  }

  /* ── Comparison slider ── */
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
      const x = e.touches ? e.touches[0].clientX : e.clientX;
      setPosition(x);
    }
    function onMove(e) {
      if (!dragging) return;
      e.preventDefault();
      const x = e.touches ? e.touches[0].clientX : e.clientX;
      setPosition(x);
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

  /* ── Tabs ── */
  document.querySelectorAll(".tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
      tab.classList.add("active");
      document.getElementById("panel-" + tab.dataset.tab).classList.add("active");
    });
  });

  /* ── Download ── */
  downloadBtn.addEventListener("click", () => {
    if (!currentTaskId) return;
    const a = document.createElement("a");
    a.href = `/api/result/${currentTaskId}`;
    a.download = "translated_page.png";
    a.click();
  });

  /* ── New / Retry ── */
  newBtn.addEventListener("click", resetAll);
  retryBtn.addEventListener("click", () => {
    showSection("upload");
    if (selectedFile) {
      dropZone.style.display = "none";
      previewRow.style.display = "flex";
    }
  });

  function resetAll() {
    selectedFile = null;
    currentTaskId = null;
    fileInput.value = "";
    previewRow.style.display = "none";
    dropZone.style.display = "";
    showSection("upload");
  }

  /* ── Section toggling ── */
  function showSection(name) {
    uploadSection.style.display     = name === "upload"     ? "" : "none";
    processingSection.style.display = name === "processing" ? "" : "none";
    resultSection.style.display     = name === "result"     ? "" : "none";
    errorSection.style.display      = name === "error"      ? "" : "none";
  }

  function showError(msg) {
    errorMsg.textContent = msg;
    showSection("error");
  }

  /* ── Helpers ── */
  function formatBytes(b) {
    if (b < 1024) return b + " B";
    if (b < 1048576) return (b / 1024).toFixed(1) + " KB";
    return (b / 1048576).toFixed(1) + " MB";
  }
  function esc(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

});
