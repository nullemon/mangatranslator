"""Free text detection for manga pages.

Finds text not inside speech bubbles — narration, dramatic overlaid text,
labels, banner captions, etc.  Two backends:

  1. CRAFT (pip install craft-text-detector) — more accurate, GPU-accelerated.
  2. Built-in CV detector — zero extra dependencies, uses morphological ops +
     MSER to find character clusters.  Always available.

The class picks CRAFT when present and falls back to CV automatically.
"""

import cv2
import numpy as np
from typing import List, Tuple

_CRAFT = None
_TRIED = False


def _patch_torchvision_vgg():
    """Newer torchvision removed model_urls from vgg module. CRAFT still
    imports it, so inject a stub before CRAFT loads."""
    try:
        from torchvision.models import vgg as _vgg
        if not hasattr(_vgg, "model_urls"):
            _vgg.model_urls = {
                "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
            }
    except Exception:
        pass


def _load_craft():
    global _CRAFT, _TRIED
    if _CRAFT is not None or _TRIED:
        return _CRAFT
    _TRIED = True
    try:
        _patch_torchvision_vgg()
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
        print(f"[text_detect] CRAFT unavailable, using CV fallback: {e}")
    return _CRAFT


class FreeTextDetector:
    """Finds text regions not inside any detected bubble."""

    def __init__(self):
        self.craft = _load_craft()
        self._method = "CRAFT" if self.craft else "CV"
        print(f"[text_detect] using {self._method}")

    @property
    def ok(self) -> bool:
        return True

    def detect(
        self,
        image: np.ndarray,
        existing_boxes: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[int, int, int, int]]:
        if self.craft:
            boxes = self._detect_craft(image, existing_boxes)
        else:
            boxes = self._detect_cv(image, existing_boxes)
        return boxes

    # ── CRAFT backend ──
    def _detect_craft(self, image, existing_boxes):
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            from .gpu_throttle import limit as _gpu_limit
            with _gpu_limit():
                result = self.craft.detect_text(rgb)
        except Exception as e:
            print(f"[text_detect] CRAFT failed, trying CV: {e}")
            return self._detect_cv(image, existing_boxes)

        char_boxes = result.get("boxes")
        # CRAFT returns boxes as a numpy array — test it explicitly, never with
        # `if not char_boxes` (ambiguous truth value on a multi-element array).
        if char_boxes is None or len(char_boxes) == 0:
            return self._detect_cv(image, existing_boxes)

        rects = []
        for box in char_boxes:
            pts = np.array(box, dtype=np.float32)
            x = max(0, int(pts[:, 0].min()))
            y = max(0, int(pts[:, 1].min()))
            x2 = min(w, int(pts[:, 0].max()))
            y2 = min(h, int(pts[:, 1].max()))
            bw, bh = x2 - x, y2 - y
            if bw < 8 or bh < 8:
                continue
            if any(_overlaps((x, y, bw, bh), eb) for eb in existing_boxes):
                continue
            rects.append((x, y, bw, bh))

        gap_x = max(int(w * 0.025), 8)
        gap_y = max(int(h * 0.018), 6)
        blocks = _group_boxes(rects, gap_x, gap_y)
        return _filter_blocks(blocks, h, w, existing_boxes)

    # ── CV fallback backend ──
    def _detect_cv(self, image, existing_boxes):
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        page_area = h * w

        candidates: List[Tuple[int, int, int, int]] = []

        # Pass 1: dark text on light background
        candidates += self._cv_pass(gray, h, w, bright_text=False)
        # Pass 2: light text on dark background (banners, dark panels)
        candidates += self._cv_pass(gray, h, w, bright_text=True)

        # Dedupe and filter
        candidates = _dedupe(candidates)
        return _filter_blocks(candidates, h, w, existing_boxes)

    def _cv_pass(self, gray, h, w, bright_text=False):
        """Find text-like regions using morphological operations."""
        page_area = h * w

        if bright_text:
            _, binary = cv2.threshold(gray, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(gray, 0, 255,
                                      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Connect nearby strokes into text-like clusters — small kernels so
        # text doesn't merge with nearby artwork.
        # Horizontal kernel for horizontal text.
        kw = max(8, w // 90)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3))
        h_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, h_kernel)

        # Vertical kernel for vertical Japanese text columns.
        vh = max(8, h // 90)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, vh))
        v_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, v_kernel)

        combined = cv2.bitwise_or(h_closed, v_closed)

        # Clean up noise
        clean_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, clean_k)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        results = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            if area < page_area * 0.0015 or area > page_area * 0.12:
                continue
            if bw < 15 or bh < 15:
                continue
            # Reject oversized blocks — a single text region shouldn't span
            # more than ~30% of the page in either direction.
            if bw > w * 0.45 or bh > h * 0.30:
                continue
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if aspect > 10:
                continue

            # Text has moderate stroke density — art is either very sparse
            # (lines) or very dense (screentone/fill).
            roi = binary[y:y + bh, x:x + bw]
            density = cv2.countNonZero(roi) / max(area, 1)
            if density < 0.10 or density > 0.65:
                continue

            # Text regions have multiple separated blobs (characters).
            roi_clean = cv2.morphologyEx(roi, cv2.MORPH_OPEN, clean_k)
            n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
                roi_clean, 8)
            min_char = max(area * 0.001, 12)
            max_char = area * 0.30
            chars = sum(1 for i in range(1, n_labels)
                        if min_char <= stats[i, cv2.CC_STAT_AREA] <= max_char)
            if chars < 3:
                continue

            results.append((x, y, bw, bh))

        return results


# ── Shared helpers ──

def _overlaps(a, b, thresh=0.3):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    xi, yi = max(ax, bx), max(ay, by)
    xf, yf = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if xi >= xf or yi >= yf:
        return False
    inter = (xf - xi) * (yf - yi)
    smaller = min(aw * ah, bw * bh)
    return smaller > 0 and inter / smaller > thresh


def _group_boxes(boxes, gap_x=20, gap_y=15):
    """Merge nearby character boxes into larger text blocks."""
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


def _filter_blocks(blocks, h, w, existing_boxes):
    """Keep text-sized blocks that don't overlap detected bubbles."""
    page_area = h * w
    out = []
    for bx, by, bw, bh in blocks:
        area = bw * bh
        if area < page_area * 0.0008 or area > page_area * 0.12:
            continue
        if bw > w * 0.45 or bh > h * 0.30:
            continue
        if any(_overlaps((bx, by, bw, bh), eb, 0.25) for eb in existing_boxes):
            continue
        pad = max(3, min(bw, bh) // 12)
        bx = max(0, bx - pad)
        by = max(0, by - pad)
        bw = min(w - bx, bw + 2 * pad)
        bh = min(h - by, bh + 2 * pad)
        out.append((bx, by, bw, bh))
    return out


def _dedupe(boxes, iou_thresh=0.4):
    """Remove near-duplicate boxes."""
    if len(boxes) <= 1:
        return boxes
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for b in boxes:
        if not any(_overlaps(b, k, iou_thresh) for k in kept):
            kept.append(b)
    return kept
