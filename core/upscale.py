"""GPU super-resolution for low-resolution raws (loaded via spandrel).

Low-res raws are the hardest quality ceiling: blurry strokes OCR worse, text
edges render soft, and the page looks like a phone photo. This upscales small
pages with a manga/anime-trained model before the pipeline runs, so detection,
OCR, cleanup and lettering all happen at a crisp working resolution.

Two backends, picked automatically:

  • MangaJaNai (preferred when installed) — models purpose-trained on B&W manga,
    one per source height (1200p–2048p). They UPSCALE FAITHFULLY: remove JPEG /
    moiré artifacts and sharpen line art without redrawing content, so unlike a
    generative "scan" they never invert blacks or melt screentone. Grayscale.
    Drop the V1 model set in models/mangajanai/ (./setup_gpu.sh --mangajanai),
    and the right height variant is chosen per page.

  • Real-ESRGAN anime (fallback) — generic anime SR, auto-downloaded (~18MB).

Optional everywhere: needs torch + spandrel. When neither is available `ok` is
False and pages pass through untouched. Disable per-run with MANGA_UPSCALE=0;
force a specific model file with UPSCALE_MODEL; point at a model dir with
MANGAJANAI_DIR (default models/mangajanai).
"""

import os
import re
import cv2
import numpy as np

_CACHE = {}        # weights path -> (descriptor, device)

REALESRGAN_PATH = "models/RealESRGAN_x4plus_anime_6B.pth"
REALESRGAN_URLS = [
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x4plus_anime_6B.pth",
    "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4plus_anime_6B.pth",
]
MANGAJANAI_DIR = "models/mangajanai"


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _download_realesrgan(dest: str) -> bool:
    import httpx
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    for url in REALESRGAN_URLS:
        try:
            print(f"[upscale] downloading Real-ESRGAN weights (~18MB): {url}")
            with httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as r:
                if r.status_code != 200:
                    print(f"[upscale] {r.status_code} from {url}, trying next...")
                    continue
                tmp = dest + ".part"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
            os.replace(tmp, dest)
            print(f"[upscale] weights saved to {dest}")
            return True
        except Exception as e:
            print(f"[upscale] download failed from {url}: {e}")
    return False


def _mangajanai_for(height: int):
    """Pick the MangaJaNai model whose optimized source height is closest to the
    page — the height is encoded in the filename (…1200p…, …2048p…). Returns a
    weights path, or None when the model set isn't installed."""
    d = os.environ.get("MANGAJANAI_DIR", MANGAJANAI_DIR)
    if not os.path.isdir(d):
        return None
    best, best_d = None, 1 << 30
    for f in sorted(os.listdir(d)):
        if not f.lower().endswith(".pth"):
            continue
        m = re.search(r"(\d{3,4})\s*p", f) or re.search(r"(\d{3,4})", f)
        h = int(m.group(1)) if m else 1500
        if abs(h - height) < best_d:
            best, best_d = os.path.join(d, f), abs(h - height)
    return best


def _pick_path(height: int):
    """Choose which weights to use for a page of this height. Explicit override
    wins; then MangaJaNai if installed; then Real-ESRGAN (auto-download)."""
    override = os.environ.get("UPSCALE_MODEL", "").strip()
    if override:
        return override
    mj = _mangajanai_for(height)
    if mj:
        return mj
    return REALESRGAN_PATH


def _get(path: str):
    """Load + cache the model at `path`. Real-ESRGAN auto-downloads."""
    if path in _CACHE:
        return _CACHE[path]
    try:
        import torch
        from spandrel import ModelLoader
        if os.path.abspath(path) == os.path.abspath(REALESRGAN_PATH) and not os.path.exists(path):
            if not _download_realesrgan(path):
                return None
        if not os.path.exists(path):
            print(f"[upscale] model file not found: {path}")
            return None
        desc = ModelLoader().load_from_file(path)
        device = _device()
        desc = desc.to(device).eval()
        _CACHE[path] = (desc, device)
        print(f"[upscale] loaded {os.path.basename(path)} "
              f"(x{desc.scale}, {desc.input_channels}ch, {device})")
        return _CACHE[path]
    except Exception as e:
        print(f"[upscale] could not load {os.path.basename(path)} ({e})")
        return None


class Upscaler:
    """Lazy SR wrapper. Tiled inference so a full page fits in VRAM. Picks the
    best installed backend per page; grayscale-aware for MangaJaNai."""

    @property
    def ok(self) -> bool:
        # Available if we can load *some* model (probe at a typical page height).
        return _get(_pick_path(1500)) is not None

    def upscale(self, image: np.ndarray, target_long: int = 2400,
                tile: int = 384, overlap: int = 16) -> np.ndarray:
        """Upscale `image` (BGR) and return it with its long edge at
        `target_long` (never more than the model's native scale allows)."""
        h, w = image.shape[:2]
        loaded = _get(_pick_path(max(h, w)))
        if loaded is None:
            return image
        desc, device = loaded
        import torch

        scale = int(desc.scale)
        gray_model = int(getattr(desc, "input_channels", 3)) == 1

        if gray_model:
            src = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            out = np.zeros((h * scale, w * scale), np.float32)
        else:
            src = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            out = np.zeros((h * scale, w * scale, 3), np.float32)

        with torch.inference_mode():
            for y0 in range(0, h, tile):
                for x0 in range(0, w, tile):
                    y1, x1 = min(h, y0 + tile), min(w, x0 + tile)
                    # Expand the tile by `overlap` so seams blend away, then
                    # crop the expansion back off the model output.
                    ey0, ex0 = max(0, y0 - overlap), max(0, x0 - overlap)
                    ey1, ex1 = min(h, y1 + overlap), min(w, x1 + overlap)
                    patch = src[ey0:ey1, ex0:ex1]
                    if gray_model:
                        t = torch.from_numpy(patch)[None, None].to(device)
                        sr = desc(t)[0, 0].clamp_(0, 1).cpu().numpy()
                    else:
                        t = torch.from_numpy(patch.transpose(2, 0, 1))[None].to(device)
                        sr = desc(t)[0].clamp_(0, 1).cpu().numpy().transpose(1, 2, 0)
                    cy0, cx0 = (y0 - ey0) * scale, (x0 - ex0) * scale
                    out[y0 * scale:y1 * scale, x0 * scale:x1 * scale] = sr[
                        cy0:cy0 + (y1 - y0) * scale, cx0:cx0 + (x1 - x0) * scale
                    ]

        out = (out * 255.0 + 0.5).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR if gray_model else cv2.COLOR_RGB2BGR)

        long_edge = max(out.shape[:2])
        target = min(target_long, long_edge)
        if long_edge > target:
            s = target / long_edge
            out = cv2.resize(out, (max(1, round(out.shape[1] * s)),
                                   max(1, round(out.shape[0] * s))),
                             interpolation=cv2.INTER_AREA)
        return out
