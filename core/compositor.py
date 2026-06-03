import cv2
import numpy as np
from PIL import Image
from typing import List, Optional, Dict

from .renderer import TextRenderer

SFX_TYPES = {"sfx", "sound", "sound_effect", "soundeffect", "onomatopoeia"}


class Compositor:
    """Replaces balloon text. Given a precise interior mask per region it wipes
    the whole interior (so the original Japanese vanishes completely) and fits
    the translation inside the true balloon shape. When no mask is supplied it
    recovers one from the bounding box; failing that it wipes an inscribed
    ellipse — never a bare rectangle that would spill past the outline."""

    def __init__(self, font_path: Optional[str] = None):
        self.renderer = TextRenderer(font_path)

    def compose(
        self,
        image: np.ndarray,
        items: List[dict],
        masks: Optional[Dict] = None,
    ) -> np.ndarray:
        masks = masks or {}
        h, w = image.shape[:2]
        page_area = h * w
        result = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        placements = []     # (rect, text, color)
        used_boxes = []

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
            bx, by, bw, bh = [int(v) for v in bbox]

            mask = masks.get(it["id"])
            if mask is None:
                mask = masks.get(str(it["id"]))
            dark = bool(it.get("dark", False))

            if mask is None:
                resolved = self._resolve_bubble(gray, bbox, page_area)
                if resolved is not None:
                    mask, _, dark = resolved

            if mask is not None:
                rr = cv2.boundingRect(mask)
                if rr[2] == 0 or rr[3] == 0:
                    mask = None

            if mask is not None:
                bb = cv2.boundingRect(mask)
                if any(self._overlaps(bb, ub) for ub in used_boxes):
                    continue
                used_boxes.append(bb)
                self._wipe(result, mask, dark)
                rect = self._inner_rect(mask)
                if rect is None:
                    rect = (bb[0] + 4, bb[1] + 4, max(bb[2] - 8, 10), max(bb[3] - 8, 10))
            else:
                # Last resort: no shape found. Wipe an inscribed ellipse in the
                # bbox (rounded, won't bleed past a real balloon outline).
                if bw < 14 or bh < 14:
                    continue
                bb = (bx, by, bw, bh)
                if any(self._overlaps(bb, ub) for ub in used_boxes):
                    continue
                used_boxes.append(bb)
                ell = np.zeros((h, w), np.uint8)
                cv2.ellipse(ell, (bx + bw // 2, by + bh // 2),
                            (max(bw // 2 - 2, 4), max(bh // 2 - 2, 4)), 0, 0, 360, 255, -1)
                self._wipe(result, ell, dark)
                pad = max(3, min(bw, bh) // 12)
                rect = (bx + pad, by + pad, bw - 2 * pad, bh - 2 * pad)

            color = (255, 255, 255) if dark else (0, 0, 0)
            placements.append((rect, text, color))
            it["placed"] = True

        if placements:
            pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            for rect, text, color in placements:
                self.renderer.draw_in_rect(pil, rect, text, color)
            result = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        return result

    def _wipe(self, result, mask, dark):
        inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        result[inner > 0] = (0, 0, 0) if dark else (255, 255, 255)

    # ── Recover a balloon mask from a bbox (used when no mask is supplied) ──
    def _resolve_bubble(self, gray, bbox, page_area):
        H, W = gray.shape[:2]
        x, y, bw, bh = [int(v) for v in bbox]
        if bw <= 0 or bh <= 0:
            return None
        cx, cy = x + bw // 2, y + bh // 2

        mx = int(max(bw * 0.8, 60))
        my = int(max(bh * 0.8, 60))
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(W, x + bw + mx), min(H, y + bh + my)
        roi = gray[y0:y1, x0:x1]
        if roi.size == 0:
            return None

        _, white = cv2.threshold(roi, 188, 255, cv2.THRESH_BINARY)
        ink = cv2.morphologyEx(
            cv2.bitwise_not(white), cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        white = cv2.bitwise_not(ink)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(white, 8)
        rh, rw = roi.shape[:2]
        border = set(labels[0, :]) | set(labels[rh - 1, :]) | set(labels[:, 0]) | set(labels[:, rw - 1])

        lcx, lcy = cx - x0, cy - y0
        lbl = 0
        if 0 <= lcy < rh and 0 <= lcx < rw:
            lbl = int(labels[lcy, lcx])
        if lbl in border:
            lbl = 0
        if lbl == 0:
            best, best_a = 0, 0
            for i in range(1, num):
                if i in border:
                    continue
                a = stats[i, cv2.CC_STAT_AREA]
                if a > best_a:
                    best, best_a = i, a
            lbl = best
        if lbl == 0:
            return None

        comp = (labels == lbl).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        cnt = max(cnts, key=cv2.contourArea)
        filled = np.zeros_like(comp)
        cv2.drawContours(filled, [cnt], -1, 255, -1)

        area = int(cv2.countNonZero(filled))
        if area < page_area * 0.0003 or area > page_area * 0.30:
            return None
        rx, ry, rw2, rh2 = cv2.boundingRect(cnt)
        if rw2 * rh2 == 0 or area / float(rw2 * rh2) < 0.45:
            return None

        eroded = cv2.erode(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
        vals = roi[eroded > 0]
        dark = bool(vals.size > 0 and float(vals.mean()) < 110)

        full = np.zeros((H, W), np.uint8)
        full[y0:y1, x0:x1] = filled
        return full, (x0 + rx, y0 + ry, rw2, rh2), dark

    def _overlaps(self, a, b) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        xi, yi = max(ax, bx), max(ay, by)
        xf, yf = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        if xi >= xf or yi >= yf:
            return False
        inter = (xf - xi) * (yf - yi)
        return inter / max(min(aw * ah, bw * bh), 1) > 0.5

    def _inner_rect(self, mask):
        """Largest axis-aligned rectangle inside the mask, grown greedily from
        the point furthest from any edge (the balloon's 'pole of inaccessibility')."""
        m = mask > 0
        H, W = m.shape
        dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, maxv, _, loc = cv2.minMaxLoc(dt)
        if maxv < 3:
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
