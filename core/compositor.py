import cv2
import numpy as np
from PIL import Image
from typing import List, Optional

from .renderer import TextRenderer

# Region types we never translate — sound effects / onomatopoeia / art text.
SFX_TYPES = {"sfx", "sound", "sound_effect", "soundeffect", "onomatopoeia"}


class Compositor:
    """Bubble-aware text replacement.

    For each detected region we locate the REAL enclosed speech bubble it sits
    in (a white blob bounded by the inked outline), wipe its entire interior so
    the original Japanese disappears completely, then fit the translation inside
    the true bubble shape. Regions that are not inside a real bubble (sound
    effects, free art text) are left untouched.
    """

    def __init__(self, font_path: Optional[str] = None):
        self.renderer = TextRenderer(font_path)

    def compose(self, image: np.ndarray, items: List[dict]) -> np.ndarray:
        h, w = image.shape[:2]
        page_area = h * w
        result = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        placements = []     # (rect, text, dark)
        used_boxes = []     # bubble bboxes already consumed

        for it in items:
            it["placed"] = False
            text = (it.get("translation") or "").strip()
            if not text:
                continue
            kind = (it.get("type") or "").lower().replace(" ", "_")
            if kind in SFX_TYPES:
                continue
            if it.get("in_bubble") is False:
                continue
            bbox = it.get("bbox")
            if not bbox:
                continue

            bubble = self._resolve_bubble(gray, bbox, page_area)
            if bubble is None:
                continue
            mask, bb, dark = bubble

            if any(self._overlaps(bb, ub) for ub in used_boxes):
                continue
            used_boxes.append(bb)

            self._wipe(result, mask, dark)
            rect = self._inner_rect(mask)
            if rect is None:
                continue
            placements.append((rect, text, dark))
            it["placed"] = True

        if placements:
            pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            for rect, text, dark in placements:
                color = (255, 255, 255) if dark else (0, 0, 0)
                self.renderer.draw_in_rect(pil, rect, text, color)
            result = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        return result

    # ── Resolve the actual bubble enclosing a detected region ──
    def _resolve_bubble(self, gray, bbox, page_area):
        H, W = gray.shape[:2]
        x, y, bw, bh = [int(v) for v in bbox]
        if bw <= 0 or bh <= 0:
            return None
        cx, cy = x + bw // 2, y + bh // 2

        mx = int(max(bw * 0.7, 50))
        my = int(max(bh * 0.7, 50))
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(W, x + bw + mx), min(H, y + bh + my)
        roi = gray[y0:y1, x0:x1]
        if roi.size == 0:
            return None

        _, white = cv2.threshold(roi, 195, 255, cv2.THRESH_BINARY)
        # Denoise + bridge tiny outline breaks WITHOUT crossing the inked
        # border (a big close would merge the interior with the page). The
        # text inside is recovered later by filling holes in the contour.
        white = cv2.morphologyEx(
            white, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        white = cv2.morphologyEx(
            white, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )

        num, labels, stats, _ = cv2.connectedComponentsWithStats(white)
        lcx, lcy = cx - x0, cy - y0
        lbl = 0
        if 0 <= lcy < labels.shape[0] and 0 <= lcx < labels.shape[1]:
            lbl = int(labels[lcy, lcx])
        if lbl == 0:
            iy0 = max(0, (y - bh // 4) - y0)
            iy1 = min(roi.shape[0], (y + bh + bh // 4) - y0)
            ix0 = max(0, (x - bw // 4) - x0)
            ix1 = min(roi.shape[1], (x + bw + bw // 4) - x0)
            sub = labels[iy0:iy1, ix0:ix1]
            vals, counts = np.unique(sub[sub > 0], return_counts=True)
            if len(vals) == 0:
                return None
            lbl = int(vals[counts.argmax()])

        comp = (labels == lbl).astype(np.uint8) * 255

        # Reject the page background (a blob touching all four ROI edges).
        if comp[0, :].any() and comp[-1, :].any() and comp[:, 0].any() and comp[:, -1].any():
            return None

        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        cnt = max(cnts, key=cv2.contourArea)
        filled = np.zeros_like(comp)
        cv2.drawContours(filled, [cnt], -1, 255, -1)

        area = int(cv2.countNonZero(filled))
        if area < page_area * 0.0006 or area > page_area * 0.16:
            return None
        rx, ry, rw, rh = cv2.boundingRect(cnt)
        if rw * rh == 0 or area / float(rw * rh) < 0.45:
            return None

        eroded = cv2.erode(
            filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2
        )
        vals = roi[eroded > 0]
        dark = bool(vals.size > 0 and float(vals.mean()) < 110)

        full = np.zeros((H, W), np.uint8)
        full[y0:y1, x0:x1] = filled
        return full, (x0 + rx, y0 + ry, rw, rh), dark

    def _overlaps(self, a, b) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        xi, yi = max(ax, bx), max(ay, by)
        xf, yf = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        if xi >= xf or yi >= yf:
            return False
        inter = (xf - xi) * (yf - yi)
        return inter / max(min(aw * ah, bw * bh), 1) > 0.5

    def _wipe(self, result, mask, dark):
        # Erode a touch so we keep the inked bubble outline intact.
        inner = cv2.erode(
            mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
        )
        result[inner > 0] = (0, 0, 0) if dark else (255, 255, 255)

    def _inner_rect(self, mask):
        """Largest axis-aligned rectangle that fits inside the bubble mask,
        grown greedily from the point furthest from any edge."""
        m = mask > 0
        H, W = m.shape
        dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, maxv, _, loc = cv2.minMaxLoc(dt)
        if maxv < 6:
            return None
        px, py = int(loc[0]), int(loc[1])
        l = r = t = b = 1
        step = 2

        def ok(l, r, t, b):
            X0, X1, Y0, Y1 = px - l, px + r, py - t, py + b
            if X0 < 0 or Y0 < 0 or X1 >= W or Y1 >= H:
                return False
            if not m[Y0, X0:X1 + 1].all():
                return False
            if not m[Y1, X0:X1 + 1].all():
                return False
            if not m[Y0:Y1 + 1, X0].all():
                return False
            if not m[Y0:Y1 + 1, X1].all():
                return False
            return True

        grew = True
        while grew:
            grew = False
            for side in range(4):
                nl, nr, nt, nb = l, r, t, b
                if side == 0:
                    nr += step
                elif side == 1:
                    nl += step
                elif side == 2:
                    nb += step
                else:
                    nt += step
                if ok(nl, nr, nt, nb):
                    l, r, t, b = nl, nr, nt, nb
                    grew = True

        return (px - l, py - t, l + r, t + b)
