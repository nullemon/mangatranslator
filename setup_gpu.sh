#!/usr/bin/env bash
# Installs the optional GPU stack and pre-downloads the speech-balloon model.
# Safe to re-run. After it finishes, restart the app: python3 app.py
set -e

echo "==> Installing GPU bubble-segmentation dependencies (torch, ultralytics)..."
pip3 install --user --break-system-packages -r requirements-gpu.txt

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

echo "==> Done. Restart the app:  python3 app.py"
echo "    Detect step shows 'segmentation model (GPU)'; bubbles are read by"
echo "    manga-ocr and erased with LaMa when those models are installed."
