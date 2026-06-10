#!/usr/bin/env bash
# Installs the optional GPU stack and pre-downloads the speech-balloon model.
# Safe to re-run. After it finishes, restart the app: python3 app.py
set -e

echo "==> Installing GPU bubble-segmentation dependencies (torch, ultralytics)..."
pip3 install --user --break-system-packages -r requirements-gpu.txt

echo "==> Installing hf_xet (high-performance HuggingFace downloads)..."
pip3 install --user --break-system-packages "huggingface_hub[hf_xet]"
export HF_XET_HIGH_PERFORMANCE=1

echo "==> Checking CUDA..."
python3 - <<'PY'
import torch
print("torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available(),
      "|", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"))
PY

echo "==> Pre-downloading the speech-balloon segmentation model..."
python3 - <<'PY'
import os
from huggingface_hub import hf_hub_download
repo = os.environ.get("BUBBLE_MODEL_REPO", "kitsumed/yolov8m_seg-speech-bubble")
fname = os.environ.get("BUBBLE_MODEL_FILE", "model.pt")
print("Model cached at:", hf_hub_download(repo_id=repo, filename=fname))
PY

echo "==> Pre-downloading manga-ocr (reads each bubble locally)..."
python3 - <<'PY'
try:
    from manga_ocr import MangaOcr
    MangaOcr()
    print("manga-ocr ready")
except Exception as e:
    print("manga-ocr skipped:", e)
PY

echo "==> Pre-downloading LaMa inpainting (clean text removal)..."
python3 - <<'PY'
try:
    from simple_lama_inpainting import SimpleLama
    SimpleLama()
    print("LaMa ready")
except Exception as e:
    print("LaMa skipped:", e)
PY

echo "==> Installing text-pixel segmentation (comic-text-detector)..."
pip3 install --user --break-system-packages onnxruntime-gpu \
  || pip3 install --user --break-system-packages onnxruntime
mkdir -p models
if [ ! -f models/comictextdetector.pt.onnx ]; then
  echo "==> Downloading comic-text-detector weights (~90MB)..."
  curl -fL -o models/comictextdetector.pt.onnx \
    https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx \
  || curl -fL -o models/comictextdetector.pt.onnx \
    https://github.com/dmMaze/comic-text-detector/releases/download/data/comictextdetector.pt.onnx \
  || echo "    (download failed — the app will retry on first run)"
fi

echo "==> Installing super-resolution (Real-ESRGAN anime via spandrel)..."
pip3 install --user --break-system-packages "spandrel>=0.3.0"
if [ ! -f models/RealESRGAN_x4plus_anime_6B.pth ]; then
  echo "==> Downloading Real-ESRGAN anime weights (~18MB)..."
  curl -fL -o models/RealESRGAN_x4plus_anime_6B.pth \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth \
  || curl -fL -o models/RealESRGAN_x4plus_anime_6B.pth \
    https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4plus_anime_6B.pth \
  || echo "    (download failed — the app will retry on first run)"
fi

echo "==> Installing CRAFT text detector (free text / narration)..."
# --no-deps: craft's own dependency list pins a 2021 opencv that no longer
# builds; everything it really needs (torch, cv2, scipy, gdown) is installed
# above. Failures here are non-fatal — the CV fallback still works.
pip3 install --user --break-system-packages gdown scipy || true
pip3 install --user --break-system-packages --no-deps craft-text-detector \
  || echo "    (CRAFT install failed — built-in CV free-text fallback will be used)"

echo "==> Pre-downloading CRAFT model weights..."
python3 - <<'PY'
try:
    from craft_text_detector import Craft
    import torch
    Craft(output_dir=None, cuda=torch.cuda.is_available(), crop_type="box")
    print("CRAFT ready")
except Exception as e:
    print("CRAFT skipped:", e)
PY

echo "==> Done. Restart the app:  python3 app.py"
echo "    Detect step shows 'segmentation model (GPU)'; bubbles are read by"
echo "    manga-ocr and erased with LaMa when those models are installed."
echo "    Free text (narration/labels) detected by CRAFT when installed."
echo "    Text strokes masked pixel-precisely by comic-text-detector."
