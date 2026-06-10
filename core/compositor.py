import math
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional, Dict

from .renderer import TextRenderer

SFX_TYPES = {"sfx", "sound", "sound_effect", "soundeffect", "onomatopoeia"}


def _is_expressive(text: str, it: dict) -> bool:
    """Sound effects and expression beats (rendered as *GRIN*, *GASP*, a placed
    BOOM, ...) get a slanted italic treatment so they read distinctly from
    ordinary dialogue."""
    kind = (it.get("type") or "").lower().replace(" ", "_")
    if kind in SFX_TYPES:
        return True
    t = (text or "").strip()
    return len(t) >= 2 and t.startswith("*") and t.endswith("*")


class Compositor:
    """Replaces balloon text. Given a precise interior mask per region it wipes
    the whole interior (so the original Japanese vanishes completely) and fits
    the translation inside the true balloon shape. When no mask is supplied it
    recovers one from the bounding box; failing that it wipes an inscribed
    ellipse — never a bare rectangle that would spill past the outline."""

    def __init__(self, font_path: Optional[str] = None, font_scale: float = 1.0,
                 use_lama: bool = True, uppercase: bool = True):
        self.renderer = TextRenderer(font_path, font_scale=font_scale,
                                     uppercase=uppercase)
        self.lama = None
        if use_lama:
            try:
                from .lama import LamaInpaint
                self.lama = LamaInpaint()
            except Exception as e:
                print(f"[compositor] LaMa unavailable: {e}")
        # GPU text-pixel segmentation: precise stroke masks for clean removal.
        # Optional — when absent, the ink-deviation heuristic is used alone.
        self.text_seg = None
        self._seg_mask = None
        try:
            from .text_seg import TextSegmenter
            self.text_seg = TextSegmenter()
        except Exception as e:
            print(f"[compositor] text segmentation unavailable: {e}")

    def compose(
        self,
        image: np.ndarray,
        items: List[dict],
        masks: Optional[Dict] = None,
        offsets: Optional[Dict] = None,
        covers: Optional[List] = None,
    ) -> np.ndarray:
        masks = masks or {}
        offsets = offsets or {}
        h, w = image.shape[:2]
        page_area = h * w
        result = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Page-level text stroke mask from the GPU segmentation model (when
        # available): tells us exactly which pixels are lettering, so erasure
        # covers whole characters and never guesses at art.
        self._seg_mask = None
        if self.text_seg is not None and self.text_seg.ok:
            try:
                self._seg_mask = self.text_seg.mask(image)
            except Exception as e:
                print(f"[compositor] text-seg mask failed: {e}")
        # Every region we actually edit. At the end we restore ALL other pixels
        # from the original, so the art / background is never touched — not a
        # pixel more than the exact text areas we cover.
        edited_rects = []

        # Manual cover/erase regions the user drew to wipe leftover or
        # untranslated text. Erase them before placing anything else.
        for cb in (covers or []):
            try:
                cx, cy, cw, ch = [int(v) for v in cb]
            except Exception:
                continue
            if cw > 2 and ch > 2:
                cap = self._detect_caption_box(gray, cx, cy, cw, ch)
                if cap is not None and not cap[4]:
                    self._fill_caption(result, cap)
                    edited_rects.append((cap[0], cap[1], cap[2], cap[3]))
                else:
                    touched = self._inpaint_text(result, cx, cy, cw, ch)
                    edited_rects.append(touched or (cx, cy, cw, ch))

        placements = []     # (rect, text, color)
        used_boxes = []

        def offset_rect(item, rect):
            off = offsets.get(item["id"])
            if off is None:
                off = offsets.get(str(item["id"]))
            if not off:
                return rect
            dx, dy = int(off[0]), int(off[1])
            return (rect[0] + dx, rect[1] + dy, rect[2], rect[3])

        for it in items:
            it["placed"] = False
            text = (it.get("translation") or "").strip()
            if not text:
                continue
            kind = (it.get("type") or "").lower().replace(" ", "_")
            if kind in SFX_TYPES and it.get("in_bubble") is False:
                continue
            ital = _is_expressive(text, it)
            bbox = it.get("bbox")
            if not bbox:
                continue
            bx, by, bw, bh = [int(v) for v in bbox]

            rotation = float(it.get("rotation", 0))
            # English text at steep angles (>45°) is unreadable sideways;
            # render it horizontally in the (tall-narrow) rect instead.
            if abs(rotation) > 45:
                rotation = 0

            # Manually added text: the user drew this box over a missed or
            # leftover region. Erase whatever's there and place the typed text
            # using exactly the box they drew (no auto-refinement).
            if it.get("manual"):
                bx = max(0, min(bx, w - 1))
                by = max(0, min(by, h - 1))
                bw = min(bw, w - bx)
                bh = min(bh, h - by)
                if bw < 6 or bh < 6:
                    continue
                # A bordered caption box gets a clean solid fill; text drawn over
                # bare artwork has just its strokes inpainted out (no slab).
                cap, bb = self._plan_free_region(gray, bx, by, bw, bh, refine=False)
                rect, dark, touched = self._apply_free_region(result, gray, cap, bb)
                edited_rects.append(tuple(int(v) for v in touched))
                color = self._pick_color(dark, it)
                placements.append((offset_rect(it, rect), text, color, ital, rotation))
                it["placed"] = True
                continue

            if it.get("in_bubble") is False:
                bx = max(0, min(bx, w - 1))
                by = max(0, min(by, h - 1))
                bw = min(bw, w - bx)
                bh = min(bh, h - by)
                if bw < 10 or bh < 10:
                    continue
                # Plan the region first (caption interior or refined ink box) so
                # overlaps are rejected before anything is painted.
                cap, bb = self._plan_free_region(gray, bx, by, bw, bh, refine=True)
                if any(self._overlaps(bb, ub) for ub in used_boxes):
                    continue
                used_boxes.append(bb)
                rect, dark, touched = self._apply_free_region(result, gray, cap, bb)
                edited_rects.append(tuple(int(v) for v in touched))
                # When no caption frame was found the refined bbox may have
                # ballooned (union with nearby ink). Constrain text to where
                # the original Japanese actually was (seg mask), falling back
                # to the original AI bbox with inset padding.
                if cap is None:
                    seg_r = self._seg_text_rect(bx, by, bw, bh)
                    if seg_r is not None:
                        sx, sy, sw, sh = seg_r
                        pad = max(3, min(sw, sh) // 10)
                        rect = (sx - pad, sy - pad,
                                max(sw + 2 * pad, 8), max(sh + 2 * pad, 8))
                    else:
                        pad = max(3, min(bw, bh) // 12)
                        rect = (bx + pad, by + pad,
                                max(bw - 2 * pad, 8), max(bh - 2 * pad, 8))
                it["bbox"] = [int(v) for v in bb]
                color = self._pick_color(dark, it)
                placements.append((offset_rect(it, rect), text, color, ital, rotation))
                it["placed"] = True
                continue

            bx = max(0, min(bx, w - 1))
            by = max(0, min(by, h - 1))
            bw = min(bw, w - bx)
            bh = min(bh, h - by)
            if bw < 10 or bh < 10:
                continue

            mask = masks.get(it["id"])
            if mask is None:
                mask = masks.get(str(it["id"]))
            dark = bool(it.get("dark", False))
            from_detector = mask is not None  # precise mask (seg/CV) — trust it

            # No precise mask (AI-located bubble): try to recover the real
            # enclosed bubble from the box, but reject a recovery that grabs
            # far more than the box (that means it leaked into the background).
            if mask is None:
                resolved = self._resolve_bubble(gray, bbox, page_area)
                if resolved is not None:
                    rmask, rbb, rdark = resolved
                    box_area = max(bw * bh, 1)
                    if rbb[2] * rbb[3] <= box_area * 2.6:
                        mask, dark = rmask, rdark

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
                    rect = (bb[0] + 2, bb[1] + 2, max(bb[2] - 4, 10), max(bb[3] - 4, 10))
            else:
                # No reliable bubble shape. Don't draw a big white ellipse
                # (that's what put boxes in random / out-of-bounds places).
                # Instead clear just the original text strokes inside the box
                # and place the translation there — tight and always in-bounds.
                bb = (bx, by, bw, bh)
                if any(self._overlaps(bb, ub) for ub in used_boxes):
                    continue
                used_boxes.append(bb)
                touched = self._inpaint_text(result, bx, by, bw, bh)
                if touched:
                    bb = touched
                dark = self._is_dark_region(gray, bx, by, bw, bh)
                pad = max(2, min(bw, bh) // 16)
                rect = (bx + pad, by + pad, bw - 2 * pad, bh - 2 * pad)

            edited_rects.append(tuple(int(v) for v in bb))
            color = self._pick_color(dark, it)
            placements.append((offset_rect(it, rect), text, color, ital, 0))
            it["placed"] = True

        # Placement rects must stay on the page — a dragged offset or a loose
        # AI box can push one past the edge, which is how text ended up out of
        # bounds. Clamp every rect to the page before anything is drawn.
        placements = [
            (self._clamp_rect(r, w, h), t, c, i, ro)
            for r, t, c, i, ro in placements
        ]
        placements = [p for p in placements if p[0] is not None]

        if placements:
            pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            for rect, text, color, ital, rot in placements:
                self.renderer.draw_in_rect(pil, rect, text, color, italic=ital,
                                           rotation=rot)
            result = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        # Hard guarantee: only the exact regions we edited may differ from the
        # original. Restore every other pixel byte-for-byte — no global cleanup,
        # no "fixing" the art or background. Text placements are included so a
        # dragged/offset line that sits outside its cover box is still kept.
        placement_rects = []
        for rect, text, color, ital, rot in placements:
            placement_rects.append(self._rotated_aabb(rect, rot))

        edited = np.zeros((h, w), np.uint8)
        for rx, ry, rw, rh in edited_rects + placement_rects:
            x0, y0 = max(0, int(rx)), max(0, int(ry))
            x1, y1 = min(w, int(rx) + int(rw)), min(h, int(ry) + int(rh))
            if x1 > x0 and y1 > y0:
                edited[y0:y1, x0:x1] = 255
        # A little dilation so antialiased text/halo at a region's edge isn't clipped.
        edited = cv2.dilate(edited, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
        keep = edited == 0
        result[keep] = image[keep]
        return result

    def _final_cleanup(self, image):
        """Light cleanup: melt scanner grain and tidy the very brightest / darkest
        pixels, but leave every gray tone (shading, screentone, pencil work)
        exactly where it is. No auto-levels — they stretch the histogram and
        push the whole page toward black-and-white."""
        out = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)

        g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        out[g > 250] = 255
        out[g < 6] = 0
        return out

    def _pick_color(self, dark, it):
        """Text color: honor a manual override ("black"/"white") when set,
        otherwise pick automatically (white on dark bubbles, black on light)."""
        ov = (it.get("color") or "auto").lower()
        if ov == "white":
            return (255, 255, 255)
        if ov == "black":
            return (0, 0, 0)
        return (255, 255, 255) if dark else (0, 0, 0)

    def _wipe(self, result, mask, dark):
        """Fill the bubble interior, pulling the fill boundary well inside the
        inked outline so the wipe never eats the bubble's own border line.

        A segmentation mask usually reaches the outline (sometimes a touch past
        it); eroding by only ~1px left the white fill sitting on the border and
        nibbling it away. Erode by a size-aware margin that clears the line."""
        _, _, bw, bh = cv2.boundingRect(mask)
        r = int(np.clip(round(min(bw, bh) * 0.04), 3, 7))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        inner = cv2.erode(mask, k)
        # Tiny / thin bubble: a big erosion would swallow it — back off so we
        # still cover the original text.
        if cv2.countNonZero(inner) < max(1, int(0.25 * cv2.countNonZero(mask))):
            inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        result[inner > 0] = (0, 0, 0) if dark else (255, 255, 255)

    def _ink_mask(self, gray_roi):
        """Mask of pixels that deviate from the smooth local background — i.e.
        text / ink of EITHER polarity, including faint low-contrast narration.
        Low-frequency shading lives in the background estimate and is ignored, so
        only the high-frequency strokes light up."""
        h, w = gray_roi.shape[:2]
        if h < 3 or w < 3:
            return np.zeros((max(h, 1), max(w, 1)), np.uint8)
        sigma = max(3.0, min(h, w) / 6.0)
        bg = cv2.GaussianBlur(cv2.medianBlur(gray_roi, 3), (0, 0), sigma)
        diff = cv2.absdiff(gray_roi, bg)
        _, mask = cv2.threshold(diff, 14, 255, cv2.THRESH_BINARY)
        return cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        )

    def _inpaint_text(self, result, x, y, w, h):
        """Remove text from a free-text region. Builds the stroke mask from where
        the image deviates from its smooth background, so faint / low-contrast
        narration of either polarity is caught and fully covered, then inpaints.

        The region is padded outward so characters that extend beyond the AI
        bounding box are also cleaned. Returns the rect actually touched
        (x, y, w, h) so the surgical restore keeps every cleaned pixel, or
        None when the region is empty."""
        H, W = result.shape[:2]
        pad = max(4, min(w, h) // 8)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        if x1 <= x0 or y1 <= y0:
            return None
        touched = (x0, y0, x1 - x0, y1 - y0)
        roi = result[y0:y1, x0:x1]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        strokes = self._ink_mask(gray_roi)
        # Union with the GPU stroke mask: the model marks whole characters
        # (including thick fills the deviation heuristic under-covers).
        if self._seg_mask is not None:
            strokes = cv2.bitwise_or(strokes, self._seg_mask[y0:y1, x0:x1])
        text_mask = cv2.dilate(strokes, kernel, iterations=3)

        if self.lama is not None and self.lama.ok:
            full_mask = np.zeros(result.shape[:2], np.uint8)
            full_mask[y0:y1, x0:x1] = cv2.dilate(text_mask, kernel, iterations=2)
            out = self.lama.inpaint(result, full_mask)
            if out is not None:
                result[:] = out
                return touched
        inpainted = cv2.inpaint(roi, text_mask, 5, cv2.INPAINT_TELEA)
        result[y0:y1, x0:x1] = inpainted
        return touched

    def _refine_free_bbox(self, gray, x, y, w, h):
        """Lock an AI-estimated free-text box onto the ACTUAL ink. The model box
        can sit a little off, so search a PADDED window, find the ink (faint or
        bold, either polarity), and return the UNION of the AI box and the ink
        bbox — so the clean and the placed translation cover everything."""
        H, W = gray.shape[:2]
        px = max(8, int(w * 0.15))
        py = max(8, int(h * 0.25))
        x0, y0 = max(0, x - px), max(0, y - py)
        x1, y1 = min(W, x + w + px), min(H, y + h + py)
        if x1 <= x0 or y1 <= y0:
            return x, y, w, h
        # Prefer the GPU stroke mask (only marks real lettering, never art
        # lines); fall back to the deviation heuristic when it's absent or
        # finds nothing in the window.
        ink = None
        if self._seg_mask is not None:
            seg_win = self._seg_mask[y0:y1, x0:x1]
            if cv2.countNonZero(seg_win) >= 10:
                ink = seg_win
        if ink is None:
            ink = self._ink_mask(gray[y0:y1, x0:x1])
        ys, xs = np.where(ink > 0)
        if xs.size < 10:
            return x, y, w, h
        rx, ry = int(xs.min()), int(ys.min())
        rw, rh = int(xs.max()) - rx + 1, int(ys.max()) - ry + 1
        if rw < 5 or rh < 5:
            return x, y, w, h
        if rw * rh > 2.0 * max(w * h, 1):
            return x, y, w, h
        if rw > w * 1.5 or rh > h * 1.5:
            return x, y, w, h
        # Union of the AI box and the ink bbox: ensures we never shrink below
        # the AI's estimate (which covers the full text column).
        ink_x = x0 + rx
        ink_y = y0 + ry
        ux = min(x, ink_x)
        uy = min(y, ink_y)
        ux2 = max(x + w, ink_x + rw)
        uy2 = max(y + h, ink_y + rh)
        pad = max(4, min(ux2 - ux, uy2 - uy) // 8)
        fx = max(0, ux - pad)
        fy = max(0, uy - pad)
        fw = min(W - fx, (ux2 - ux) + 2 * pad)
        fh = min(H - fy, (uy2 - uy) + 2 * pad)
        return fx, fy, fw, fh

    def _seg_text_rect(self, x, y, w, h):
        """Bounding box of actual text strokes within (x,y,w,h) from the
        page-level seg mask.  Returns (sx, sy, sw, sh) in page coords, or
        None when the mask is absent or the region is nearly empty."""
        if self._seg_mask is None:
            return None
        H, W = self._seg_mask.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return None
        roi = self._seg_mask[y0:y1, x0:x1]
        if cv2.countNonZero(roi) < 10:
            return None
        ys, xs = np.where(roi > 0)
        rx, ry = int(xs.min()), int(ys.min())
        rw = int(xs.max()) - rx + 1
        rh = int(ys.max()) - ry + 1
        if rw < 8 or rh < 8:
            return None
        return (x0 + rx, y0 + ry, rw, rh)

    def _is_dark_region(self, gray, x, y, w, h):
        H, W = gray.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return False
        return float(np.median(gray[y0:y1, x0:x1])) < 128

    # ── Free / manual text regions: caption-box fill vs. inpaint ──
    def _detect_caption_box(self, gray, x, y, w, h):
        """Detect a caption / narration / title slab with a flat interior.

        Light box (dark text on white): we search a slightly PADDED window and
        return the framed white interior that encloses the AI's text box — so a
        loose or clipped AI box snaps to the real frame and the border survives.
        We only accept it when the interior is genuinely enclosed by a frame; an
        unframed bright patch (text lying on light artwork) returns None so the
        caller inpaints the strokes tightly instead of stamping a giant white
        rectangle at the wrong size.

        Dark slab (light text on black — e.g. a full-bleed vertical title bar)
        returns its whole extent, so big characters that split the black field
        into chunks can't leave broken slivers behind.
        Returns (ix, iy, iw, ih, dark), or None for textured artwork."""
        H, W = gray.shape[:2]
        ox0, oy0 = max(0, x), max(0, y)
        ox1, oy1 = min(W, x + w), min(H, y + h)
        if ox1 - ox0 < 14 or oy1 - oy0 < 14:
            return None
        # Decide light vs dark from the AI box itself, so padding into a black
        # gutter (light case) or white margin (dark case) can't flip it.
        inner = gray[oy0:oy1, ox0:ox1]
        dark = float(np.median(inner)) < 110

        if dark:
            roi = inner
            roi_area = roi.shape[0] * roi.shape[1]
            _, field = cv2.threshold(roi, 80, 255, cv2.THRESH_BINARY_INV)
            ks = int(np.clip(min(roi.shape[:2]) // 10, 7, 25))
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks))
            field = cv2.morphologyEx(field, cv2.MORPH_CLOSE, k)
            # Union bbox of the whole dark extent (not the largest blob) keeps a
            # full-height title bar from fragmenting around big characters.
            ys, xs = np.where(field > 0)
            if xs.size == 0:
                return None
            bx, by = int(xs.min()), int(ys.min())
            bw, bh = int(xs.max()) - bx + 1, int(ys.max()) - by + 1
            if bw < 10 or bh < 10 or bw * bh < roi_area * 0.35:
                return None
            dens = float(np.count_nonzero(field[by:by + bh, bx:bx + bw])) / float(bw * bh)
            if dens < 0.55:
                return None
            return (ox0 + bx, oy0 + by, bw, bh, True)

        # Light box: search a padded window so we can recover a frame the AI box
        # clipped, then snap to the white interior that holds the AI box centre.
        px, py = int(w * 0.30), int(h * 0.30)
        X0, Y0 = max(0, x - px), max(0, y - py)
        X1, Y1 = min(W, x + w + px), min(H, y + h + py)
        roi = gray[Y0:Y1, X0:X1]
        rh, rw = roi.shape[:2]
        if rh < 14 or rw < 14:
            return None
        _, field = cv2.threshold(roi, 185, 255, cv2.THRESH_BINARY)
        ks = int(np.clip(min(rh, rw) // 10, 7, 25))
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks))
        field = cv2.morphologyEx(field, cv2.MORPH_CLOSE, k)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(field, 8)
        if num <= 1:
            return None
        # Prefer the blob covering the AI box's centre; else the largest blob.
        cx = int(np.clip((x + w // 2) - X0, 0, rw - 1))
        cy = int(np.clip((y + h // 2) - Y0, 0, rh - 1))
        pick = int(labels[cy, cx])
        if pick == 0:
            best_a = 0
            for i in range(1, num):
                a = int(stats[i, cv2.CC_STAT_AREA])
                if a > best_a:
                    pick, best_a = i, a
        if pick == 0:
            return None
        bx = int(stats[pick, cv2.CC_STAT_LEFT])
        by = int(stats[pick, cv2.CC_STAT_TOP])
        bw = int(stats[pick, cv2.CC_STAT_WIDTH])
        bh = int(stats[pick, cv2.CC_STAT_HEIGHT])

        # The interior must be ENCLOSED: if the bright blob runs to the edge of
        # the padded window it bled into surrounding artwork (no frame) — bail so
        # the caller inpaints the text instead of pasting an oversized box.
        if bx <= 1 or by <= 1 or bx + bw >= rw - 1 or by + bh >= rh - 1:
            return None
        if bw < 12 or bh < 12:
            return None
        dens = float(np.count_nonzero(field[by:by + bh, bx:bx + bw])) / float(bw * bh)
        if dens < 0.6:
            return None
        return (X0 + bx, Y0 + by, bw, bh, False)

    def _fill_caption(self, result, cap):
        """Fill a detected caption interior with a solid clean color (white, or
        black for an inverted box), preserving its border frame. Returns the
        filled rect (fx, fy, fw, fh) for text placement.

        The fill follows the ACTUAL interior shape: hand-drawn frames wander,
        so a straight inset rectangle left a ring of original paper between
        the fill and the line — a visible seam that read as a doubled border
        (and the API page finish inked it into a real second line). Painting
        the interior component itself, shrunk a few px off the line, reaches
        the frame everywhere without ever touching it."""
        ix, iy, iw, ih, dark = cap
        fill = (0, 0, 0) if dark else (255, 255, 255)
        roi = result[iy:iy + ih, ix:ix + iw]
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if dark:
            _, field = cv2.threshold(g, 80, 255, cv2.THRESH_BINARY_INV)
        else:
            _, field = cv2.threshold(g, 185, 255, cv2.THRESH_BINARY)
        ks = int(np.clip(min(iw, ih) // 10, 7, 25))
        field = cv2.morphologyEx(
            field, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks)))
        num, labels, stats, _ = cv2.connectedComponentsWithStats(field, 8)
        pick, best = 0, 0
        for i in range(1, num):
            a = int(stats[i, cv2.CC_STAT_AREA])
            if a > best:
                pick, best = i, a
        if pick:
            comp = (labels == pick).astype(np.uint8) * 255
            # Fill the component's holes (text strokes) via its outer contour
            # so the original lettering can't peek through the fill.
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                comp = np.zeros_like(comp)
                cv2.drawContours(comp, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
            inner = cv2.erode(comp, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (7, 7)))
            if cv2.countNonZero(inner) >= 0.55 * iw * ih:
                roi[inner > 0] = fill
                bx, by, bw, bh = cv2.boundingRect(inner)
                return ix + bx, iy + by, bw, bh
        # Fallback (interior shape not recovered): inset rectangle as before.
        m = max(2, min(iw, ih) // 22)
        fx, fy = ix + m, iy + m
        fw, fh = max(iw - 2 * m, 4), max(ih - 2 * m, 4)
        result[fy:fy + fh, fx:fx + fw] = fill
        return fx, fy, fw, fh

    def _plan_free_region(self, gray, x, y, w, h, refine):
        """Decide the bbox a free/manual region will occupy, without touching the
        image, so overlaps can be rejected first. Returns (caption_or_None, bbox)."""
        cap = self._detect_caption_box(gray, x, y, w, h)
        if cap is not None:
            return cap, (cap[0], cap[1], cap[2], cap[3])
        if refine:
            return None, self._refine_free_bbox(gray, x, y, w, h)
        return None, (x, y, w, h)

    def _apply_free_region(self, result, gray, cap, bbox):
        """Clear a planned free region and return (text_rect, dark, touched).

        A LIGHT caption box (framed white interior) gets a solid clean white
        fill — that's how official releases look and the paper really is flat.
        A DARK slab does NOT get stamped solid black: the field around the
        lettering is usually textured (grain, gradients, screentone), so a
        flat black rectangle reads as an obvious patch. Instead only the
        strokes are erased and inpainted, letting the texture continue, and
        the translation is drawn straight onto it in white.
        Free text over artwork has just its strokes inpainted."""
        if cap is not None and not cap[4]:
            fx, fy, fw, fh = self._fill_caption(result, cap)
            pad = max(3, min(fw, fh) // 12)
            rect = (fx + pad, fy + pad, max(fw - 2 * pad, 8), max(fh - 2 * pad, 8))
            return rect, False, (cap[0], cap[1], cap[2], cap[3])
        if cap is not None:
            ix, iy, iw, ih, _ = cap
            touched = self._inpaint_text(result, ix, iy, iw, ih) or (ix, iy, iw, ih)
            pad = max(3, min(iw, ih) // 12)
            rect = (ix + pad, iy + pad, max(iw - 2 * pad, 8), max(ih - 2 * pad, 8))
            return rect, True, touched
        rx, ry, rw, rh = [int(v) for v in bbox]
        touched = self._inpaint_text(result, rx, ry, rw, rh) or (rx, ry, rw, rh)
        dark = self._is_dark_region(gray, rx, ry, rw, rh)
        pad = max(2, min(rw, rh) // 16)
        rect = (rx + pad, ry + pad, max(rw - 2 * pad, 8), max(rh - 2 * pad, 8))
        return rect, dark, touched

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

    @staticmethod
    def _clamp_rect(rect, w, h):
        """Intersect a placement rect with the page; None if nothing remains."""
        x, y, rw, rh = [int(v) for v in rect]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + rw), min(h, y + rh)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        return (x0, y0, x1 - x0, y1 - y0)

    @staticmethod
    def _rotated_aabb(rect, rotation):
        """Axis-aligned bounding box that covers *rect* after clockwise
        rotation by *rotation* degrees."""
        if abs(rotation) < 2:
            return rect
        x, y, w, h = rect
        rad = math.radians(abs(rotation))
        c, s = abs(math.cos(rad)), abs(math.sin(rad))
        rw = int(w * c + h * s) + 4
        rh = int(w * s + h * c) + 4
        cx, cy = x + w // 2, y + h // 2
        return (cx - rw // 2, cy - rh // 2, rw, rh)

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
        step = 3

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
