#!/usr/bin/env python3
"""Feature check for MangaTranslator — verifies every component is installed
and actually loads, so you know exactly what will run before translating.

Usage:
    python3 check_setup.py            # full check (loads every model)
    python3 check_setup.py --quick    # imports + weight files only, no model loads

The full check may download missing weights on first run (CRAFT ~80MB,
Real-ESRGAN ~18MB, comic-text-detector ~90MB) — same as the app would.
"""

import os
import subprocess
import sys

QUICK = "--quick" in sys.argv


def _load_env(path=".env"):
    """Same .env loading the app does, so HF_TOKEN etc. apply here too."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    except OSError:
        pass


_load_env()

GREEN, RED, YELLOW, DIM, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[0m"
results = []


def check(group, name, fn):
    try:
        detail = fn()
        status, detail = ("SKIP", detail[5:]) if isinstance(detail, str) and detail.startswith("SKIP:") else ("OK", detail or "")
    except ImportError as e:
        status, detail = "MISSING", str(e).split("(")[0].strip()
    except Exception as e:
        status, detail = "FAIL", f"{type(e).__name__}: {e}"
    results.append((group, name, status, str(detail)[:90]))
    icon = {"OK": f"{GREEN}✓{RESET}", "MISSING": f"{RED}✗{RESET}",
            "FAIL": f"{RED}✗{RESET}", "SKIP": f"{YELLOW}-{RESET}"}[status]
    print(f"  {icon} [{group:5s}] {name:<28s} {DIM}{detail}{RESET}")


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


print(f"\n== MangaTranslator feature check ==")
print(f"Python {sys.version.split()[0]} | commit {git_commit()}"
      f" | mode: {'quick' if QUICK else 'full'}\n")

# ── Core libraries ──
def _cv2():
    import cv2; return cv2.__version__
check("core", "OpenCV", _cv2)

def _numpy():
    import numpy; return numpy.__version__
check("core", "NumPy", _numpy)

def _pil():
    import PIL; return PIL.__version__
check("core", "Pillow", _pil)

def _fastapi():
    import fastapi; return fastapi.__version__
check("core", "FastAPI", _fastapi)

def _httpx():
    import httpx; return httpx.__version__
check("core", "httpx", _httpx)

def _anthropic():
    import anthropic; return anthropic.__version__
check("core", "anthropic SDK", _anthropic)

def _hf_token():
    tok = (os.environ.get("HF_TOKEN") or
           os.environ.get("HUGGING_FACE_HUB_TOKEN") or "")
    if tok:
        return f"set ({tok[:7]}…) — HF downloads authenticated"
    cached = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(cached):
        return "logged in via huggingface-cli"
    return ("SKIP:not set — copy .env.example to .env and add HF_TOKEN=... "
            "(silences HF rate-limit warnings)")
check("core", "HuggingFace token", _hf_token)

# ── GPU stack ──
def _torch():
    import torch
    cuda = torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0) if cuda else "CPU only"
    return f"{torch.__version__} | CUDA: {gpu}"
check("gpu", "torch", _torch)

def _onnx():
    import onnxruntime as ort
    provs = ort.get_available_providers()
    has_cuda = "CUDAExecutionProvider" in provs
    return f"{ort.__version__} | {'CUDA' if has_cuda else 'CPU only'} ({len(provs)} providers)"
check("gpu", "onnxruntime", _onnx)

def _ultra():
    import ultralytics; return ultralytics.__version__
check("gpu", "ultralytics (YOLO)", _ultra)

def _spandrel():
    import spandrel; return spandrel.__version__
check("gpu", "spandrel", _spandrel)

# ── Weight files on disk ──
def _file_check(path, hint):
    def fn():
        if os.path.exists(path):
            mb = os.path.getsize(path) / 1e6
            return f"{mb:.0f}MB at {path}"
        return f"SKIP:not downloaded yet — {hint}"
    return fn
check("model", "comic-text-detector weights",
      _file_check("models/comictextdetector.pt.onnx", "auto-downloads on first run"))
check("model", "Real-ESRGAN weights",
      _file_check("models/RealESRGAN_x4plus_anime_6B.pth", "auto-downloads on first run"))

# ── Model loads (skipped in --quick) ──
if QUICK:
    print(f"  {YELLOW}-{RESET} [model] (model loading skipped — drop --quick for the full check)")
else:
    def _bubble():
        from core.bubble_seg import BubbleSegDetector
        d = BubbleSegDetector()
        if not d.ok:
            raise RuntimeError("model failed to load — run ./setup_gpu.sh")
        return "speech-balloon YOLOv8 ready"
    check("model", "Bubble segmentation", _bubble)

    def _ocr():
        from core.ocr import MangaOCR
        o = MangaOCR()
        if not o.ok:
            raise RuntimeError("manga-ocr failed to load")
        return "manga-ocr ready"
    check("model", "manga-ocr", _ocr)

    def _lama():
        from core.lama import LamaInpaint
        l = LamaInpaint()
        if not l.ok:
            raise RuntimeError("LaMa failed to load")
        return "LaMa inpainting ready"
    check("model", "LaMa inpainting", _lama)

    def _seg():
        from core.text_seg import TextSegmenter, _load
        t = TextSegmenter()
        if not t.ok:
            raise RuntimeError("comic-text-detector failed to load")
        sess = _load()
        prov = (sess.get_providers() or ["?"])[0] if sess else "?"
        return f"ready — running on {prov.replace('ExecutionProvider', '')}"
    check("model", "Text-pixel segmentation", _seg)

    def _upscale():
        from core.upscale import Upscaler, _mangajanai_for
        mj = _mangajanai_for(1500)
        u = Upscaler()
        if not u.ok:
            raise RuntimeError("upscaler failed to load")
        if mj:
            d = os.path.dirname(mj)
            n = len([f for f in os.listdir(d) if f.lower().endswith(".pth")])
            return f"MangaJaNai ({n} manga models) — faithful, preferred"
        return "Real-ESRGAN anime (add MangaJaNai: ./setup_gpu.sh --mangajanai)"
    check("model", "Upscaler", _upscale)

    def _craft():
        from core.text_detect import _load_craft
        c = _load_craft()
        if c is None:
            raise RuntimeError("CRAFT failed to load — CV fallback will be used")
        return "CRAFT free-text detector ready"
    check("model", "CRAFT free-text", _craft)

# ── App features (cheap functional tests, no GPU needed) ──
def _rotation():
    from PIL import Image
    from core.renderer import TextRenderer
    r = TextRenderer()
    img = Image.new("RGB", (400, 300), (255, 255, 255))
    r.draw_in_rect(img, (50, 50, 300, 200), "rotated lettering test",
                   (0, 0, 0), rotation=12)
    return "rotated text renders"
check("app", "Rotation-matched lettering", _rotation)

def _balanced():
    from PIL import Image, ImageDraw
    from core.renderer import TextRenderer
    r = TextRenderer()
    img = Image.new("RGB", (10, 10))
    d = ImageDraw.Draw(img)
    font = r._get_font(20)
    lines = r._wrap("the quick brown fox jumps over the lazy dog", font, 220, d)
    if len(lines) < 2:
        raise RuntimeError("expected multi-line wrap")
    return f"balanced wrap -> {len(lines)} lines"
check("app", "Balanced line wrapping", _balanced)

def _arabic():
    import numpy as np
    from PIL import Image, features
    from core.renderer import TextRenderer
    r = TextRenderer()
    eff = r._effective_font_path("الفصل 1185 : دعيهم وشأنهم")
    if not eff or r._coverage(eff) is None or ord("ا") not in r._coverage(eff):
        raise RuntimeError("no Arabic-capable font found — bundle one in fonts/")
    img = Image.new("RGB", (760, 90), (255, 255, 255))
    r.draw_in_rect(img, (20, 10, 720, 70), "الفصل 1185 : دعيهم وشأنهم", (0, 0, 0))
    if int((np.array(img) < 200).any(axis=2).sum()) < 500:
        raise RuntimeError("Arabic rendered no ink")
    raqm = features.check("raqm")
    try:
        import arabic_reshaper, bidi  # noqa: F401
        reshaper = True
    except Exception:
        reshaper = False
    if not raqm and not reshaper:
        return (f"SKIP:renders via {os.path.basename(eff)} but neither libraqm "
                "nor arabic-reshaper present — shaping may be imperfect")
    how = "raqm (native)" if raqm else "arabic-reshaper fallback"
    return f"{os.path.basename(eff)} + {how}"
check("app", "Arabic / RTL typesetting", _arabic)

def _clamp():
    from core.compositor import Compositor
    r = Compositor._clamp_rect((-20, -20, 100, 100), 200, 200)
    if r != (0, 0, 80, 80):
        raise RuntimeError(f"clamp gave {r}")
    aabb = Compositor._rotated_aabb((10, 10, 100, 40), 30)
    if aabb[2] <= 100:
        raise RuntimeError("rotated AABB should widen")
    return "placement clamp + rotated AABB"
check("app", "In-bounds placement", _clamp)

def _finish():
    import numpy as np
    from core.pipeline import scan_finish
    img = np.full((64, 64, 3), 200, np.uint8)
    img[20:40, 20:40] = 40
    out = scan_finish(img)
    if out.shape != img.shape:
        raise RuntimeError("shape changed")
    return "local clean-scan finish works"
check("app", "Clean-scan finish (local)", _finish)

def _enhancer():
    from core.enhancer import ImageEnhancer
    provs = sorted(ImageEnhancer.DEFAULT_MODELS)
    return "providers: " + ", ".join(provs)
check("app", "API page finish (enhancer)", _enhancer)

def _clean_finish_option():
    with open("templates/index.html", encoding="utf-8") as f:
        html = f.read()
    # The translated page is delivered surgically (art untouched). The old
    # generative "api" page finish repainted the art and was removed; the
    # delivered finish is local-only ("clean" keeps art, "off" = original).
    if 'value="clean"' not in html:
        raise RuntimeError("Clean-scan finish option missing from UI — pull latest")
    if 'value="api"' in html:
        raise RuntimeError("Generative 'api' page finish still in UI — it repaints art; pull latest")
    return "UI delivers art-safe finish (clean / off)"
check("app", "Page finish keeps art (no generative repaint)", _clean_finish_option)

def _fonts():
    if not os.path.isdir("fonts"):
        return "SKIP:no fonts/ dir — renderer falls back to system fonts"
    fonts = [f for f in os.listdir("fonts") if f.lower().endswith((".ttf", ".otf"))]
    if not fonts:
        return "SKIP:fonts/ is empty — renderer falls back to system fonts"
    return f"{len(fonts)} font(s): " + ", ".join(fonts[:4])
check("app", "Lettering fonts", _fonts)

# ── Summary ──
ok = sum(1 for r in results if r[2] == "OK")
miss = sum(1 for r in results if r[2] in ("MISSING", "FAIL"))
skip = sum(1 for r in results if r[2] == "SKIP")
print(f"\n  {ok} OK · {miss} missing/failed · {skip} skipped\n")
if miss:
    print("  Fix missing items with:  ./setup_gpu.sh   then re-run this check.")
    sys.exit(1)
print("  Everything ready. Start the app:  python3 app.py")
