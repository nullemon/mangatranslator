"""GPU speech-balloon segmentation.

Uses a YOLOv8-seg model trained on manga speech bubbles to produce a precise
pixel mask per balloon — far more accurate than heuristic CV, especially on
dense pages. Loads lazily, runs on CUDA when present, and degrades gracefully:
if ultralytics / the weights aren't installed, `available()` is False and the
pipeline falls back to the CV detector. Nothing breaks when it's absent.

Install on the user's machine:  ./setup_gpu.sh   (see requirements-gpu.txt)
Override the model with env vars BUBBLE_MODEL_PATH or BUBBLE_MODEL_REPO/_FILE.
"""

import os
import cv2
import numpy as np
from typing import List, Optional

from .detector import TextRegion, draw_annotations

_MODEL = None
_TRIED = False

DEFAULT_REPO = "kitsumed/yolov8m_seg-speech-bubble"
DEFAULT_FILE = "model.pt"


def available() -> bool:
    try:
        import ultralytics  # noqa: F401
        return True
    except Exception:
        return False


def _device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


def _vram_imgsz_cap() -> int:
    """Largest detection size the GPU can run without OOM-ing (which would drop
    us to crude CV detection). Scaled to total VRAM; CPU gets a modest size."""
    try:
        import torch
        if torch.cuda.is_available():
            gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gb >= 14:
                return 2048
            if gb >= 10:
                return 1792
            if gb >= 7:          # 8GB laptop 4060 etc.
                return 1536
            return 1280
    except Exception:
        pass
    return 1024


def _load():
    """Load (and cache) the YOLO model, downloading weights on first use."""
    global _MODEL, _TRIED
    if _MODEL is not None or _TRIED:
        return _MODEL
    _TRIED = True
    try:
        from ultralytics import YOLO
        path = os.environ.get("BUBBLE_MODEL_PATH", "").strip()
        if not path:
            from huggingface_hub import hf_hub_download
            repo = os.environ.get("BUBBLE_MODEL_REPO", DEFAULT_REPO)
            fname = os.environ.get("BUBBLE_MODEL_FILE", DEFAULT_FILE)
            path = hf_hub_download(repo_id=repo, filename=fname)
        model = YOLO(path)
        try:
            model.to(_device() if _device() == "cpu" else "cuda")
        except Exception:
            pass
        _MODEL = model
    except Exception as e:
        print(f"[bubble_seg] model unavailable, using CV fallback: {e}")
        _MODEL = None
    return _MODEL


class BubbleSegDetector:
    def __init__(self, conf: Optional[float] = None):
        # Lower confidence = higher recall (catch faint, small, half-cut and
        # overlapping balloons). False positives are cheap: every region is
        # OCR-checked downstream and dropped if it isn't really Japanese.
        # Override with BUBBLE_CONF.
        self.conf = conf if conf is not None else float(
            os.environ.get("BUBBLE_CONF", "0.15"))
        self.model = _load()

    @property
    def ok(self) -> bool:
        return self.model is not None

    def detect(self, image: np.ndarray) -> List[TextRegion]:
        if self.model is None:
            return []
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Run detection at high resolution. The model's default 640px downscale
        # loses small / distant / dense balloons on a full manga page — the
        # single biggest cause of "it missed bubbles". This is internal-only; it
        # never changes the output pixels. The cap is sized to available VRAM so
        # detection can't OOM and silently fall back to crude CV detection (an
        # 8GB laptop GPU is also feeding manga-ocr, LaMa, text-seg and CRAFT).
        # Override either dial with BUBBLE_IMGSZ.
        env_imgsz = os.environ.get("BUBBLE_IMGSZ", "").strip()
        if env_imgsz:
            imgsz = int(env_imgsz)
        else:
            imgsz = min(_vram_imgsz_cap(), max(1024, (max(h, w) + 31) // 32 * 32))
        try:
            from .gpu_throttle import limit as _gpu_limit
            with _gpu_limit():
                results = self.model.predict(
                    image, conf=self.conf, device=_device(),
                    retina_masks=True, verbose=False, imgsz=imgsz,
                )
        except Exception as e:
            print(f"[bubble_seg] inference failed, CV fallback: {e}")
            return []
        if not results:
            return []
        r = results[0]
        if getattr(r, "masks", None) is None or r.masks.data is None:
            return []

        regions: List[TextRegion] = []
        for seg in r.masks.data.cpu().numpy():
            m = (seg > 0.5).astype(np.uint8) * 255
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            reg = self._region_from_mask(gray, m)
            if reg is not None:
                regions.append(reg)

        regions = self._dedupe(regions)
        self._sort_reading_order(regions, h)
        for i, reg in enumerate(regions):
            reg.id = i + 1
        return regions

    def _region_from_mask(self, gray, raw_mask) -> Optional[TextRegion]:
        cnts, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        cnt = max(cnts, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(cnt)

        # Page-relative minimum-size gate. A FIXED 12px floor is meaningless
        # across scan sizes (~1.4% of an 850px page but ~0.6% of a 2000px page),
        # so on hi-res scans the segmenter's hallucinated background specks
        # (30-50px blobs over art / screentone) sailed straight through and got
        # a translation stamped on them. Gate on fractions of the ACTUAL page,
        # chosen well below the smallest real bubble (a lone 'え?' / '?!'
        # reaction balloon is ~0.0012 of page area, short side ~3-4% — a 2x+
        # margin) so genuine tiny bubbles always survive. Lettering-free art of
        # any size is caught separately downstream by the text-evidence gate.
        ph, pw = raw_mask.shape[:2]
        page_area = ph * pw
        balloon_area = float(cv2.contourArea(cnt))
        min_area_floor = max(0.0005 * page_area, 250.0)   # ~0.05% of the page
        min_dim_floor = max(0.013 * min(ph, pw), 10)      # ~1.3% of short side
        if balloon_area < min_area_floor or min(bw, bh) < min_dim_floor:
            return None

        # Solid balloon footprint (fill any seg holes).
        solid = np.zeros_like(raw_mask)
        cv2.drawContours(solid, [cnt], -1, 255, -1)

        # Is this a dark balloon (white text on black) or a normal one?
        vals = gray[solid > 0]
        if vals.size == 0:
            return None
        dark = bool(float(vals.mean()) < 110)

        # Carve the interior: erode inward to clear the inked outline so the
        # wipe never paints over the balloon border. Scale erosion to size.
        k = int(np.clip(round(min(bw, bh) * 0.05), 3, 12))
        interior = cv2.erode(solid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        if cv2.countNonZero(interior) < 0.2 * cv2.countNonZero(solid):
            interior = solid

        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] else x + bw // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] else y + bh // 2
        rx, ry, rw, rh = cv2.boundingRect(interior)
        return TextRegion(
            id=0, bbox=(rx, ry, rw, rh), mask=interior, center=(cx, cy),
            area=int(cv2.countNonZero(interior)), dark=dark, region_type="bubble",
        )

    def _dedupe(self, regions: List[TextRegion]) -> List[TextRegion]:
        if len(regions) <= 1:
            return regions
        regions = sorted(regions, key=lambda r: r.area, reverse=True)
        kept: List[TextRegion] = []
        for r in regions:
            if any(_iou(r.bbox, k.bbox) > 0.4 for k in kept):
                continue
            kept.append(r)
        return kept

    def _sort_reading_order(self, regions: List[TextRegion], img_h: int):
        row_h = max(img_h / max(4, len(regions) // 2 + 1), 1)
        regions.sort(key=lambda r: (int(r.center[1] / row_h), -r.center[0]))

    def create_annotated_image(self, image, regions):
        return draw_annotations(image, regions)


def _iou(b1, b2) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xi, yi = max(x1, x2), max(y1, y2)
    xf, yf = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    if xi >= xf or yi >= yf:
        return 0.0
    inter = (xf - xi) * (yf - yi)
    return inter / max(w1 * h1 + w2 * h2 - inter, 1)
