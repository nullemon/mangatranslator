#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# MangaTranslator — one-command setup (safe to re-run)
#
# Installs EVERYTHING needed:  base deps, GPU stack, all model weights,
# CUDA library shims, CRAFT free-text detector, and verifies it all works.
#
# Usage:
#     chmod +x setup_gpu.sh && ./setup_gpu.sh
#     ./setup_gpu.sh --mangajanai     # also fetch the MangaJaNai manga upscaler
#
# After it finishes the final check_setup.py tells you what's green.
# Then start the app:   python3 app.py
# ═══════════════════════════════════════════════════════════════════════════
set -e

PIP="pip3 install --user --break-system-packages"

WANT_MANGAJANAI=0
for arg in "$@"; do
    [ "$arg" = "--mangajanai" ] && WANT_MANGAJANAI=1
done

# ── .env (secrets / config) ──────────────────────────────────────────────
if [ -f .env ]; then
    set -a; . ./.env; set +a
    echo "[✓] Loaded .env"
else
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "[!] Created .env from .env.example — edit it to add your HF_TOKEN"
    fi
fi

# ── Base Python deps (FastAPI, OpenCV, Pillow, etc.) ─────────────────────
echo ""
echo "==> [1/8] Base dependencies..."
$PIP -r requirements.txt

# ── GPU stack (torch, YOLO, onnxruntime, spandrel) ───────────────────────
echo ""
echo "==> [2/8] GPU stack (torch, ultralytics, onnxruntime-gpu, spandrel)..."
$PIP -r requirements-gpu.txt
$PIP "huggingface_hub[hf_xet]"
export HF_XET_HIGH_PERFORMANCE=1

# ── CUDA 12 runtime libs for onnxruntime-gpu ─────────────────────────────
# onnxruntime-gpu dlopens CUDA 12 libs by soname; torch CUDA 13 only ships
# .so.13 — these pip wheels provide the .so.12 versions so the text-pixel
# segmentation model actually runs on GPU instead of falling back to CPU.
echo ""
echo "==> [3/8] CUDA 12 runtime libs for onnxruntime..."
$PIP nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 \
     nvidia-cufft-cu12 nvidia-curand-cu12 \
  || echo "    (install failed — text-seg will fall back to CPU, still works)"

# ── CRAFT free-text detector (narration, labels, dramatic text) ──────────
# --no-deps: CRAFT pins a 2021 opencv (<4.5.4.62) that won't build on modern
# Python and breaks everything. Its actual runtime deps (torch, cv2, scipy,
# gdown) are already installed above.
echo ""
echo "==> [4/8] CRAFT free-text detector..."
$PIP gdown scipy || true
$PIP --no-deps craft-text-detector \
  || echo "    (CRAFT install failed — built-in CV fallback will be used)"

# ── Model weights ────────────────────────────────────────────────────────
echo ""
echo "==> [5/8] Downloading model weights..."

mkdir -p models

echo "  → Speech-balloon segmentation (YOLOv8)..."
python3 - <<'PY'
import os
from huggingface_hub import hf_hub_download
repo = os.environ.get("BUBBLE_MODEL_REPO", "kitsumed/yolov8m_seg-speech-bubble")
fname = os.environ.get("BUBBLE_MODEL_FILE", "model.pt")
print("    cached at:", hf_hub_download(repo_id=repo, filename=fname))
PY

echo "  → manga-ocr..."
python3 - <<'PY'
try:
    from manga_ocr import MangaOcr
    MangaOcr()
    print("    manga-ocr ready")
except Exception as e:
    print("    manga-ocr skipped:", e)
PY

echo "  → LaMa inpainting..."
python3 - <<'PY'
try:
    from simple_lama_inpainting import SimpleLama
    SimpleLama()
    print("    LaMa ready")
except Exception as e:
    print("    LaMa skipped:", e)
PY

if [ ! -f models/comictextdetector.pt.onnx ]; then
    echo "  → comic-text-detector (~90MB)..."
    curl -fL -o models/comictextdetector.pt.onnx \
      https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx \
    || curl -fL -o models/comictextdetector.pt.onnx \
      https://github.com/dmMaze/comic-text-detector/releases/download/data/comictextdetector.pt.onnx \
    || echo "    (download failed — will auto-retry on first run)"
else
    echo "  → comic-text-detector: already downloaded"
fi

if [ ! -f models/RealESRGAN_x4plus_anime_6B.pth ]; then
    echo "  → Real-ESRGAN anime (~18MB)..."
    curl -fL -o models/RealESRGAN_x4plus_anime_6B.pth \
      https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth \
    || curl -fL -o models/RealESRGAN_x4plus_anime_6B.pth \
      https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4plus_anime_6B.pth \
    || echo "    (download failed — will auto-retry on first run)"
else
    echo "  → Real-ESRGAN: already downloaded"
fi

# MangaJaNai — manga-trained upscaler (faithful: sharpens & de-artifacts without
# redrawing). Opt-in (the V1 set is ~hundreds of MB). Run with --mangajanai.
if [ "$WANT_MANGAJANAI" = "1" ]; then
    if ls models/mangajanai/*.pth >/dev/null 2>&1; then
        echo "  → MangaJaNai: already installed ($(ls models/mangajanai/*.pth | wc -l) models)"
    else
        echo "  → MangaJaNai V1 manga upscaler (B&W model set)..."
        mkdir -p models/mangajanai
        if curl -fL -o models/mangajanai/_mj.zip \
            https://github.com/the-database/MangaJaNai/releases/download/1.0.0/MangaJaNai_V1_ModelsOnly.zip; then
            python3 - <<'PY'
import zipfile, os, glob
z = "models/mangajanai/_mj.zip"
with zipfile.ZipFile(z) as f:
    for m in f.namelist():
        if m.lower().endswith(".pth"):
            data = f.read(m)
            with open(os.path.join("models/mangajanai", os.path.basename(m)), "wb") as o:
                o.write(data)
os.remove(z)
print("    extracted:", len(glob.glob("models/mangajanai/*.pth")), "MangaJaNai models")
PY
        else
            echo "    (download failed — get MangaJaNai_V1_ModelsOnly.zip from"
            echo "     https://github.com/the-database/MangaJaNai/releases and unzip"
            echo "     the .pth files into models/mangajanai/ )"
        fi
    fi
else
    echo "  → MangaJaNai: skipped (run ./setup_gpu.sh --mangajanai to add the manga upscaler)"
fi

# Arabic font for right-to-left output — bundled in the repo; re-fetch only if
# missing so Arabic renders in a proper Naskh face, not a generic fallback.
if [ ! -f fonts/NotoNaskhArabic-Bold.ttf ]; then
    echo "  → Noto Naskh Arabic (RTL output font)..."
    curl -fsSL -o fonts/NotoNaskhArabic-Bold.ttf \
      https://raw.githubusercontent.com/notofonts/notofonts.github.io/main/fonts/NotoNaskhArabic/hinted/ttf/NotoNaskhArabic-Bold.ttf \
    || echo "    (download failed — Arabic falls back to DejaVu, still works)"
else
    echo "  → Noto Naskh Arabic: already present"
fi

echo "  → CRAFT text detection..."
python3 - <<'PY'
try:
    from craft_text_detector import Craft
    import torch
    Craft(output_dir=None, cuda=torch.cuda.is_available(), crop_type="box")
    print("    CRAFT ready")
except Exception as e:
    print("    CRAFT skipped:", e)
PY

# ── Quick GPU test ───────────────────────────────────────────────────────
echo ""
echo "==> [6/8] GPU check..."
python3 - <<'PY'
import torch
cuda = torch.cuda.is_available()
gpu = torch.cuda.get_device_name(0) if cuda else "CPU only"
print(f"    torch {torch.__version__} | CUDA: {gpu}")
if not cuda:
    print("    ⚠  No GPU detected — models will run on CPU (slower but works)")
PY

python3 - <<'PY'
try:
    import onnxruntime as ort
    provs = ort.get_available_providers()
    has_cuda = "CUDAExecutionProvider" in provs
    print(f"    onnxruntime {ort.__version__} | {'CUDA' if has_cuda else 'CPU only'}")
except ImportError:
    print("    onnxruntime: not installed")
PY

# ── Kill any old server ──────────────────────────────────────────────────
echo ""
echo "==> [7/8] Stopping old server (if any)..."
pkill -f "python3 app.py" 2>/dev/null && echo "    killed old process" \
  || echo "    no old server running"
sleep 1

# ── Full verification ────────────────────────────────────────────────────
echo ""
echo "==> [8/8] Running full verification..."
echo ""
python3 check_setup.py

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Setup complete! Start the app:"
echo ""
echo "      python3 app.py"
echo ""
echo "  Then open http://localhost:8000 in your browser."
echo "════════════════════════════════════════════════════════════════"
