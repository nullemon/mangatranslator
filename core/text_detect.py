"""Free text detection for manga pages.

Uses CRAFT (Character Region Awareness for Text detection) to find text
not inside speech bubbles — narration, dramatic overlaid text, labels,
etc. Loaded lazily; the pipeline falls back gracefully when
craft-text-detector isn't installed.

Install:  pip install craft-text-detector
"""

import cv2
import numpy as np
from typing import List, Tuple

_CRAFT = None
_TRIED = False


def available() -> bool:
    try:
        import craft_text_detector  # noqa: F401
        return True
    except ImportError:
        return False


def _load():
    global _CRAFT, _TRIED
    if _CRAFT is not None or _TRIED:
        return _CRAFT
    _TRIED = True
    try:
        from craft_text_detector import Craft
        import torch
        _CRAFT = Craft(
            output_dir=None,
            cuda=torch.cuda.is_available(),
            crop_type="box",
            text_threshold=0.65,
            link_threshold=0.35,
            low_text=0.35,
        )
        print("[text_detect] CRAFT loaded OK")
    except Exception as e:
        print(f"[text_detect] CRAFT unavailable: {e}")
    return _CRAFT


class FreeTextDetector:
    """Finds text regions not inside any detected bubble."""

    def __init__(self):
        self.craft = _load()

    @property
    def ok(self) -> bool:
        return self.craft is not None

    def detect(
        self,
        image: np.ndarray,
        existing_boxes: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[int, int, int, int]]:
        if not self.ok:
            return []

        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            result = self.craft.detect_text(rgb)
        except Exception as e:
            print(f"[text_detect] CRAFT inference failed: {e}")
            return []

        boxes = result.get("boxes", [])
        if not boxes or len(boxes) == 0:
            return []

        rects: List[Tuple[int, int, int, int]] = []
        for box in boxes:
            pts = np.array(box, dtype=np.float32)
            x = max(0, int(pts[:, 0].min()))
            y = max(0, int(pts[:, 1].min()))
            x2 = min(w, int(pts[:, 0].max()))
            y2 = min(h, int(pts[:, 1].max()))
            bw, bh = x2 - x, y2 - y
            if bw < 8 or bh < 8:
                continue
            if bw * bh < h * w * 0.0002:
                continue
            if any(_overlaps((x, y, bw, bh), eb) for eb in existing_boxes):
                continue
            rects.append((x, y, bw, bh))

        gap_x = max(int(w * 0.025), 8)
        gap_y = max(int(h * 0.018), 6)
        blocks = _group_boxes(rects, gap_x=gap_x, gap_y=gap_y)

        filtered = []
        page_area = h * w
        for bx, by, bw, bh in blocks:
            area = bw * bh
            if area < page_area * 0.0008:
                continue
            if area > page_area * 0.20:
                continue
            if any(_overlaps((bx, by, bw, bh), eb, 0.25) for eb in existing_boxes):
                continue
            pad = max(3, min(bw, bh) // 10)
            bx = max(0, bx - pad)
            by = max(0, by - pad)
            bw = min(w - bx, bw + 2 * pad)
            bh = min(h - by, bh + 2 * pad)
            filtered.append((bx, by, bw, bh))

        return filtered


def _overlaps(a, b, thresh=0.3):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    xi, yi = max(ax, bx), max(ay, by)
    xf, yf = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if xi >= xf or yi >= yf:
        return False
    inter = (xf - xi) * (yf - yi)
    smaller = min(aw * ah, bw * bh)
    if smaller <= 0:
        return False
    return inter / smaller > thresh


def _group_boxes(boxes, gap_x=20, gap_y=15):
    """Merge nearby CRAFT character boxes into larger text blocks."""
    if not boxes:
        return []
    boxes = list(boxes)
    used = [False] * len(boxes)
    merged = []

    for i in range(len(boxes)):
        if used[i]:
            continue
        group = [boxes[i]]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(len(boxes)):
                if used[j]:
                    continue
                bx, by, bw, bh = boxes[j]
                for gx, gy, gw, gh in group:
                    if (bx <= gx + gw + gap_x and bx + bw >= gx - gap_x
                            and by <= gy + gh + gap_y and by + bh >= gy - gap_y):
                        group.append(boxes[j])
                        used[j] = True
                        changed = True
                        break

        xs = [b[0] for b in group]
        ys = [b[1] for b in group]
        x2s = [b[0] + b[2] for b in group]
        y2s = [b[1] + b[3] for b in group]
        merged.append((min(xs), min(ys), max(x2s) - min(xs), max(y2s) - min(ys)))

    return merged
