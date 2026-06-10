"""GPU super-resolution (Real-ESRGAN anime model, loaded via spandrel).

Low-resolution raws are the hardest quality ceiling: blurry strokes OCR worse,
text edges render soft, and the final page looks like a phone photo. This
upscales small pages with a model trained on anime/manga line art before the
pipeline runs, so detection, OCR, cleanup and lettering all happen at a crisp
working resolution.

Optional: needs torch + spandrel and the RealESRGAN_x4plus_anime_6B weights
(auto-downloaded to models/ on first use, or pre-fetched by ./setup_gpu.sh).
When missing, `ok` is False and pages pass through untouched.

Disable per-run with env var MANGA_UPSCALE=0; override weights with
UPSCALE_MODEL.
"""

import os
import cv2
import numpy as np

_MODEL = None       # (descriptor, device)
_TRIED = False

DEFAULT_PATH = "models/RealESRGAN_x4plus_anime_6B.pth"
WEIGHT_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
    "RealESRGAN_x4plus_anime_6B.pth"
)


def _weights_path() -> str:
    return os.environ.get("UPSCALE_MODEL", "").strip() or DEFAULT_PATH


def _download_weights(dest: str) -> bool:
    import httpx
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    try:
        print(f"[upscale] downloading weights (~18MB): {WEIGHT_URL}")
        with httpx.stream("GET", WEIGHT_URL, follow_redirects=True, timeout=300.0) as r:
            if r.status_code != 200:
                return False
            tmp = dest + ".part"
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
        os.replace(tmp, dest)
        print(f"[upscale] weights saved to {dest}")
        return True
    except Exception as e:
        print(f"[upscale] download failed: {e}")
        return False


def _load():
    global _MODEL, _TRIED
    if _MODEL is not None or _TRIED:
        return _MODEL
    _TRIED = True
    try:
        import torch
        from spandrel import ModelLoader
        path = _weights_path()
        if not os.path.exists(path) and not _download_weights(path):
            print("[upscale] weights unavailable; pages pass through as-is")
            return None
        desc = ModelLoader().load_from_file(path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        desc = desc.to(device).eval()
        _MODEL = (desc, device)
        print(f"[upscale] Real-ESRGAN anime ready ({device}, x{desc.scale})")
    except Exception as e:
        print(f"[upscale] unavailable ({e}); pages pass through as-is")
        _MODEL = None
    return _MODEL


class Upscaler:
    """Lazy Real-ESRGAN wrapper. Tiled inference so a full page fits in VRAM."""

    @property
    def ok(self) -> bool:
        return _load() is not None

    def upscale(self, image: np.ndarray, target_long: int = 2400,
                tile: int = 384, overlap: int = 16) -> np.ndarray:
        """Upscale `image` (BGR) and return it with its long edge at
        `target_long` (never more than the model's native scale allows)."""
        loaded = _load()
        if loaded is None:
            return image
        desc, device = loaded
        import torch

        h, w = image.shape[:2]
        scale = int(desc.scale)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out = np.zeros((h * scale, w * scale, 3), np.float32)

        with torch.inference_mode():
            for y0 in range(0, h, tile):
                for x0 in range(0, w, tile):
                    y1, x1 = min(h, y0 + tile), min(w, x0 + tile)
                    # Expand the tile by `overlap` so seams blend away, then
                    # crop the expansion back off the model output.
                    ey0, ex0 = max(0, y0 - overlap), max(0, x0 - overlap)
                    ey1, ex1 = min(h, y1 + overlap), min(w, x1 + overlap)
                    t = torch.from_numpy(
                        rgb[ey0:ey1, ex0:ex1].transpose(2, 0, 1)
                    )[None].to(device)
                    sr = desc(t)[0].clamp_(0, 1).cpu().numpy().transpose(1, 2, 0)
                    cy0, cx0 = (y0 - ey0) * scale, (x0 - ex0) * scale
                    out[y0 * scale:y1 * scale, x0 * scale:x1 * scale] = sr[
                        cy0:cy0 + (y1 - y0) * scale, cx0:cx0 + (x1 - x0) * scale
                    ]

        out = (out * 255.0 + 0.5).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

        long_edge = max(out.shape[:2])
        target = min(target_long, long_edge)
        if long_edge > target:
            s = target / long_edge
            out = cv2.resize(out, (max(1, round(out.shape[1] * s)),
                                   max(1, round(out.shape[0] * s))),
                             interpolation=cv2.INTER_AREA)
        return out
