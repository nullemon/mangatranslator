# MangaTranslator — Fresh Install Guide

Everything needed to set this up on a brand-new computer, from zero.
The app runs entirely on **your** machine — pages never leave it except the
text/image calls to the AI translator you configure.

---

## 1. What you need

| Thing | Notes |
|---|---|
| **Windows 10/11 or Linux** | Both covered below. On Windows, WSL2 is the smoothest route for GPU. |
| **Python 3.10 – 3.12** | 3.11 or 3.12 recommended. 3.13 is not supported yet by some of the AI libraries. |
| **Disk space** | ~1 GB for the CPU-only app. ~10 GB total with the full GPU stack + models. |
| **Internet (first run)** | AI models auto-download on first use (~700 MB–1 GB total). After that it works offline except translation calls. |
| **NVIDIA GPU (optional)** | 6 GB+ VRAM recommended. Unlocks the precise balloon model, manga-OCR, LaMa erasure, upscaler. Without it everything still works on CPU fallbacks (lower quality). |
| **Gemini API key** | For translation (aistudio.google.com → "Get API key"). A Claude key works too. |
| **xAI (Grok) key (optional)** | Only for the AI *Scan / enhance* workflows. Skip if you don't use those. |

---

## 2. Quick start — CPU only, any OS (5 minutes)

Works everywhere, no GPU needed. Good for trying it out; detection/erasure use
the built-in CV fallbacks.

1. **Install Python** from `python.org` (Windows: tick **"Add python.exe to PATH"** in the installer).
2. **Extract the ZIP** somewhere permanent, e.g. `C:\MangaTranslator` or `~/MangaTranslator`.
3. Open a terminal **in that folder** (Windows: type `cmd` in the Explorer address bar) and run:

   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate

   pip install -r requirements.txt
   python setup_models.py            # recommended: the accuracy models (works without GPU)
   python app.py
   ```

4. Open **http://localhost:8000** in your browser.
5. Paste your **Gemini API key** in Settings, upload a page, hit **Translate**.

> Start it again later: open a terminal in the folder, activate the venv
> (step 3's activate line), `python app.py`.

---

## 3. Full install with NVIDIA GPU (recommended)

This is what makes the output good: exact balloon masks, per-bubble OCR
(no mixed-up translations), clean LaMa text erasure, HD upscaling.

### 3a. Windows — via WSL2 (recommended)

1. Install a current NVIDIA driver on Windows (that's all — no CUDA toolkit needed).
2. In **PowerShell (Admin)**: `wsl --install` → reboot → create your Ubuntu user.
3. In the **Ubuntu** terminal:

   ```bash
   sudo apt update && sudo apt install -y python3-venv python3-pip unzip
   # copy/extract the app zip into your home, e.g. ~/MangaTranslator, then:
   cd ~/MangaTranslator
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ./setup_gpu.sh                    # torch + all GPU models, guided
   ./setup_gpu.sh --mangajanai       # optional: best-quality manga upscaler
   python app.py
   ```

4. Open **http://localhost:8000** in your Windows browser (WSL forwards the port).

### 3b. Windows — native (no WSL)

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python setup_models.py --gpu
python app.py
```

Notes for native Windows:
- The balloon model, manga-OCR, LaMa and the text-stroke model **auto-download
  on first run** — the first translated page takes a few extra minutes.
- The upscaler auto-downloads a fallback anime model on first use. For the
  top-tier **MangaJaNai** model set, run `./setup_gpu.sh --mangajanai` once in
  WSL, or copy a ready `models/mangajanai/` folder from another install.

### 3c. Linux (native)

```bash
sudo apt install -y python3-venv    # or your distro's equivalent
cd ~/MangaTranslator
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
./setup_gpu.sh
./setup_gpu.sh --mangajanai         # optional
python app.py
```

### 3d. macOS (Intel & Apple Silicon)

Everything runs, but there's no NVIDIA/CUDA on a Mac — the AI models run on
the CPU (Apple-GPU/MPS acceleration is partial). Translation quality is the
same; heavy steps (erasure, OCR, upscale) are slower than on an NVIDIA PC.

```bash
# Terminal, in the extracted folder:
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # the app itself
python setup_models.py                   # the full AI stack, Mac (CPU/MPS) builds
python app.py
```

Do **not** run `setup_gpu.sh` on a Mac — it targets Linux/NVIDIA (it would try
to install `onnxruntime-gpu`, which doesn't exist for macOS). Use the pip
lines above instead. Models auto-download on first run exactly as on PC.

Rough expectations: a typical page that takes ~15–30 s on an RTX GPU takes a
few minutes on an M-series Mac in Maximum Quality; turn Maximum Quality off
for day-to-day speed.

### Verify the GPU stack

```bash
python check_setup.py
```

…or just start the app and read the banner — every stage prints its status:

```
[pipeline]   balloon detect : segmentation model (GPU)
[pipeline]   manga-ocr      : GPU/ON
[pipeline]   LaMa inpaint   : GPU/ON
...
```

`CV detector` / `OFF` lines mean that stage is on the CPU fallback.

---

## 4. First run & API keys

- **Translation key** — Settings → paste your **Gemini** (or Claude) key.
  Keys are stored in your browser only.
- **Scan/enhance key (optional)** — the Scan workflows panel takes an
  **xAI (Grok)**, OpenAI or Gemini image-model key.
- **Manga title** — fill it in; it gives the translator context for names.
- The **first translated page is slow** (model downloads + warmup). After
  that, pages take seconds-to-a-minute depending on GPU and settings.

---

## 5. Everyday use

- **Start**: terminal in the folder → activate venv → `python app.py` → `http://localhost:8000`.
- **Stop**: `Ctrl+C` in the terminal.
- **Different port**: create `.env` with `PORT=8080`.
- **Update to a new build**: extract the new ZIP **over** the folder (or into a
  fresh one), keeping your `.env`, `profiles/` and any fonts you added. Restart
  the app and hard-refresh the browser (`Ctrl+Shift+R`).

---

## 6. Optional extras

- **Fonts** — drop any `.ttf`/`.otf` into `fonts/` (or use the in-app upload).
  Anton / Bangers give the big-display pro look; Anime Ace is the classic
  dialogue font.
- **.env** — copy `.env.example` → `.env` for optional settings:
  `HF_TOKEN` (faster model downloads), `PORT`, detection tuning knobs.
- **GPU usage cap** — in-app Settings; lower it if the machine lags while
  translating.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `python` not found (Windows) | Reinstall Python with **Add to PATH** ticked, reopen the terminal. |
| Port 8000 already in use | Another copy is running — close it, or set `PORT=8080` in `.env`. |
| First page takes forever | Normal — models are downloading (~700 MB). Watch the terminal. |
| `No upscale model installed` | Run `./setup_gpu.sh --mangajanai` (WSL/Linux), or leave HD Upscale off. |
| CUDA out of memory | Lower the **GPU usage cap** in Settings; avoid HD Upscale + Maximum Quality together on 8 GB cards. |
| Text quality poor / wrong bubbles | Make sure the banner shows the GPU stages ON — the CV fallbacks are much weaker. |
| Browser shows an old UI after updating | Hard-refresh: `Ctrl+Shift+R`. |
| Firewall prompt on first start | Allow it — it's the local web server (localhost only). |

---

## 8. Folder map

```
MangaTranslator/
├─ app.py              ← the server; `python app.py` runs everything
├─ VERSION             ← build id shown in the startup banner
├─ requirements.txt    ← core install (CPU)
├─ requirements-gpu.txt← GPU extras (installed by setup_gpu.sh)
├─ setup_gpu.sh        ← guided GPU setup (Linux / WSL)
├─ check_setup.py      ← doctor: prints what's installed / missing
├─ core/               ← detection, OCR, translation, typesetting engine
├─ static/, templates/ ← the web UI
├─ fonts/              ← lettering fonts (drop more in here)
├─ models/             ← AI model weights (auto-created / downloaded)
├─ uploads/, output/   ← working files (auto-created, safe to empty)
└─ profiles/           ← your learned series profiles (keep when updating)
```
