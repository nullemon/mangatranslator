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
                 use_lama: bool = True, uppercase: bool = True,
                 translate_sfx: bool = False, replace_watermark: bool = False,
                 watermark_text: str = ""):
        self.renderer = TextRenderer(font_path, font_scale=font_scale,
                                     uppercase=uppercase)
        # Background SFX (out-of-bubble onomatopoeia) are left in the artwork
        # unless the user opts in to translating + typesetting them.
        self.translate_sfx = bool(translate_sfx)
        # Watermark items (type "watermark" / erase flag) are wiped from the art;
        # optionally the user's own watermark is dropped in their place.
        self.replace_watermark = bool(replace_watermark)
        self.watermark_text = (watermark_text or "").strip()
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

    @staticmethod
    def _item_scale(it: dict) -> float:
        """Per-region font-size multiplier from the editor (A- / A+)."""
        try:
            s = float(it.get("font_scale", 1.0))
        except (TypeError, ValueError):
            s = 1.0
        return max(0.4, min(s, 3.0))

    @staticmethod
    def _item_glow(it: dict) -> bool:
        """Per-region soft-glow style (editor toggle), for stylized lines."""
        return bool(it.get("glow"))

    def clean(self, image: np.ndarray, bubble_masks=None, det_image=None) -> np.ndarray:
        """Remove ALL text from the page (no translation): inpaint every text
        stroke the GPU detector marks, content-aware, so bubbles go blank-white
        and free text over art is healed — a clean raw to use as you please.
        Only the text pixels change; the art is preserved.

        `bubble_masks` (detected speech-balloon interiors) are added to the
        erase mask, so text INSIDE a bubble that the stroke detector misses is
        still cleared — the balloon interior is uniform anyway, so inpainting it
        just yields clean white."""
        result = image.copy()
        h, w = image.shape[:2]
        # Detect text on a (possibly contrast-boosted) copy so faint raws read
        # well, but ERASE from the original pixels.
        src = det_image if det_image is not None and det_image.shape[:2] == (h, w) else image
        text_mask = np.zeros((h, w), np.uint8)
        if self.text_seg is not None and self.text_seg.ok:
            try:
                text_mask = self.text_seg.mask(src)
            except Exception as e:
                print(f"[compositor] clean: text-seg mask failed: {e}")

        # Flat-fill each detected speech balloon with its OWN background colour,
        # snapped to pure white (or black for dark bubbles). A balloon interior is
        # uniform, so filling it gives a perfectly clean box — far better than
        # inpainting it, which reconstructs from neighbours and looks smudged.
        # The inked outline is protected by eroding the mask first, and any text
        # inside the balloon is wiped by the fill.
        erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for bm in (bubble_masks or []):
            if bm is None or bm.shape[:2] != (h, w):
                continue
            interior = cv2.erode((bm > 0).astype(np.uint8) * 255, erode_k, iterations=2) > 0
            if int(interior.sum()) < 64:
                continue
            med = np.median(result[interior].reshape(-1, 3), axis=0)
            lum = 0.114 * med[0] + 0.587 * med[1] + 0.299 * med[2]   # BGR luma
            fill = (255, 255, 255) if lum >= 165 else (0, 0, 0) if lum <= 70 else med
            result[interior] = fill
            text_mask[interior] = 0      # handled by the flat fill — don't inpaint

        if cv2.countNonZero(text_mask) == 0:
            return result
        # Inpaint only the remaining text strokes (free text sitting over artwork).
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        text_mask = cv2.dilate(text_mask, k, iterations=2)   # cover antialiased halos
        if self.lama is not None and self.lama.ok:
            out = self.lama.inpaint(result, text_mask)
            if out is not None:
                # HARD GUARANTEE: LaMa re-synthesizes the WHOLE frame (its
                # padding/rescale softens untouched art — the page came back
                # blurred). Take its pixels ONLY inside the erase mask; every
                # other pixel stays byte-identical to the page.
                if out.shape[:2] != result.shape[:2]:
                    out = cv2.resize(out, (result.shape[1], result.shape[0]),
                                     interpolation=cv2.INTER_CUBIC)
                m = text_mask > 0
                result[m] = out[m]
                return result
        return cv2.inpaint(result, text_mask, 5, cv2.INPAINT_TELEA)

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
            # Free-form lasso: {"poly": [[x,y], ...]} — content-aware heal the
            # whole outlined shape (for weird-shaped leftovers the box can't hug).
            if isinstance(cb, dict) and cb.get("poly"):
                touched = self._inpaint_poly(result, cb["poly"])
                if touched:
                    edited_rects.append(touched)
                continue
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
                    touched = self._inpaint_text(result, cx, cy, cw, ch, contain=True)
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
            kind = (it.get("type") or "").lower().replace(" ", "_")

            # Credit / TL name: small clean text in the margin/gutter — NO erase
            # (it sits in white space or over art), just an outlined overlay you
            # can drag per page. Drawn before everything so it's the base layer.
            if it.get("credit") or kind == "credit":
                ctext = (it.get("translation") or "").strip()
                cbox = it.get("bbox")
                if ctext and cbox:
                    cx, cy, cw, ch = self._clamp_rect([int(v) for v in cbox], w, h)
                    if cw >= 8 and ch >= 8:
                        dark = self._is_dark_region(gray, cx, cy, cw, ch)
                        color = (255, 255, 255) if dark else (0, 0, 0)
                        placements.append((offset_rect(it, (cx, cy, cw, ch)), ctext,
                                           color, False, 0, self._item_scale(it), False))
                        it["placed"] = True
                continue

            # Site watermark / URL: erase it from the art (no translation). If the
            # user opted to replace it, drop their own watermark in the same spot.
            if it.get("erase") or kind == "watermark":
                wbox = it.get("bbox")
                if wbox:
                    wx, wy, ww, wh = self._clamp_rect([int(v) for v in wbox], w, h)
                    if ww >= 6 and wh >= 6:
                        cap, bb = self._plan_free_region(gray, wx, wy, ww, wh, refine=True)
                        rect, dark, touched = self._apply_free_region(result, gray, cap, bb, contain=True)
                        edited_rects.append(tuple(int(v) for v in touched))
                        if self.replace_watermark and self.watermark_text:
                            placements.append((rect, self.watermark_text,
                                               self._pick_color(dark, it), False, 0, 1.0, False))
                        it["placed"] = True
                continue

            text = (it.get("translation") or "").strip()
            if not text:
                continue
            if (kind in SFX_TYPES and it.get("in_bubble") is False
                    and not self.translate_sfx and not it.get("manual_box")):
                continue
            ital = _is_expressive(text, it)
            bbox = it.get("bbox")
            if not bbox:
                continue
            bx, by, bw, bh = [int(v) for v in bbox]

            rotation = float(it.get("rotation", 0))
            if it.get("manual_rot"):
                # User-set tilt: honour it as-is (safety-capped only).
                rotation = max(-80.0, min(80.0, rotation))
            elif abs(rotation) > 45:
                # English text at steep angles (>45°) is unreadable sideways;
                # render it horizontally in the (tall-narrow) rect instead.
                rotation = 0

            # Manually added text, OR any box the user resized by hand: erase
            # whatever's inside and fit the translation to EXACTLY that box, with
            # no auto-refine and no bubble-mask. This covers giant title text the
            # detector wrongly treats as a bubble (so the translation lands in a
            # tiny pocket inside the lettering instead of replacing the whole
            # thing) — resizing the box now fixes it whatever its classification.
            if it.get("manual") or it.get("manual_box"):
                bx = max(0, min(bx, w - 1))
                by = max(0, min(by, h - 1))
                bw = min(bw, w - bx)
                bh = min(bh, h - by)
                if bw < 6 or bh < 6:
                    continue
                # A bordered caption box gets a clean solid fill; text drawn over
                # bare artwork has just its strokes inpainted out (no slab).
                cap, bb = self._plan_free_region(gray, bx, by, bw, bh, refine=False)
                rect, dark, touched = self._apply_free_region(result, gray, cap, bb, contain=True)
                # Point-selected outline: the translation must sit inside the
                # user's shape — and a strip-shaped selection runs ALONG the
                # strip at its own angle (a tilted banner gets tilted text).
                if it.get("poly"):
                    pr, prot = self._poly_placement(it["poly"], w, h)
                    if pr is not None:
                        rect = pr
                        if not it.get("manual_rot") and abs(prot) >= 1.0:
                            rotation = prot
                # A strip-shaped box means ONE line running along it. The model
                # often returns the translation with hard line breaks (which the
                # renderer honours for bubbles) — in a strip they'd stack tiny
                # lines instead, so collapse them into a single flowing line.
                if rect[2] >= 3 * rect[3]:
                    text = " ".join(text.split())
                edited_rects.append(tuple(int(v) for v in touched))
                color = self._pick_color(dark, it)
                placements.append((offset_rect(it, rect), text, color, ital, rotation,
                               self._item_scale(it), self._item_glow(it)))
                it["placed"] = True
                continue

            if it.get("in_bubble") is False:
                bx = max(0, min(bx, w - 1))
                by = max(0, min(by, h - 1))
                bw = min(bw, w - bx)
                bh = min(bh, h - by)
                if bw < 10 or bh < 10:
                    continue
                # Better tilt logic: when the detector called this horizontal
                # but the ORIGINAL ink is a confidently tilted elongated block
                # (a diagonal banner / slanted title bar), typeset the
                # translation at the ink's own angle so it sits like the source.
                if abs(rotation) < 3:
                    est = self._estimate_text_angle(bx, by, bw, bh)
                    if est is not None:
                        rotation = est
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
                # Vertical source column (すごい… style): a tall-narrow rect
                # width-crushes horizontal English into a tiny font. Re-shape it
                # into a horizontal box at the column's center, sized to the
                # SOURCE glyphs, growing sideways only over quiet background.
                if abs(rotation) < 3 and not it.get("manual_rot"):
                    wided = self._widen_vertical_rect(rect, result, used_boxes)
                    if wided != tuple(int(v) for v in rect):
                        rect = wided
                        used_boxes.append(tuple(int(v) for v in rect))
                # Store the TIGHT text rect as the region's box (not the ballooned
                # refine box) so the editor handle hugs the words — "same size as
                # the text or a touch bigger", not a giant rectangle.
                it["bbox"] = [int(v) for v in rect]
                # Strip-shaped region (title/credits bar): one flowing line —
                # collapse any hard line breaks the model returned.
                if rect[2] >= 3 * rect[3]:
                    text = " ".join(text.split())
                color = self._pick_color(dark, it)
                placements.append((offset_rect(it, rect), text, color, ital, rotation,
                               self._item_scale(it), self._item_glow(it)))
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
            placements.append((offset_rect(it, rect), text, color, ital,
                               rotation if it.get("manual_rot") else 0,
                               self._item_scale(it), self._item_glow(it)))
            it["placed"] = True

        # Placement rects must stay on the page — a dragged offset or a loose
        # AI box can push one past the edge, which is how text ended up out of
        # bounds. Clamp every rect to the page before anything is drawn.
        placements = [
            (self._clamp_rect(r, w, h), t, c, i, ro, fs, gl)
            for r, t, c, i, ro, fs, gl in placements
        ]
        placements = [p for p in placements if p[0] is not None]

        if placements:
            pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            for rect, text, color, ital, rot, fscale, glow in placements:
                self.renderer.draw_in_rect(pil, rect, text, color, italic=ital,
                                           rotation=rot, scale=fscale, glow=glow)
            result = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        # Hard guarantee: only the exact regions we edited may differ from the
        # original. Restore every other pixel byte-for-byte — no global cleanup,
        # no "fixing" the art or background. Text placements are included so a
        # dragged/offset line that sits outside its cover box is still kept.
        placement_rects = []
        for rect, text, color, ital, rot, fscale, glow in placements:
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
        interior = inner > 0
        if not interior.any():
            return
        # Fill with the balloon's OWN background colour so the wipe blends in — a
        # grey/screentoned bubble stays grey instead of being bleached to white.
        # The median ignores the (minority) text strokes; near-white/near-black
        # snap to clean so normal bubbles come out crisp.
        med = np.median(result[interior].reshape(-1, 3), axis=0)
        lum = 0.114 * med[0] + 0.587 * med[1] + 0.299 * med[2]   # BGR luma
        if lum >= 205:
            fill = (255, 255, 255)
        elif lum <= 50:
            fill = (0, 0, 0)
        else:
            fill = (int(med[0]), int(med[1]), int(med[2]))
        result[interior] = fill

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

    def _inpaint_poly(self, result, pts):
        """Content-aware fill an arbitrary free-form (lasso) region — the whole
        outlined shape is reconstructed from its surroundings (LaMa, or cv2
        fallback). Returns the touched bbox or None."""
        H, W = result.shape[:2]
        try:
            poly = np.array([[int(p[0]), int(p[1])] for p in pts], np.int32)
        except Exception:
            return None
        if len(poly) < 3:
            return None
        mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        x, y, w, h = cv2.boundingRect(poly)
        if w < 3 or h < 3:
            return None
        # Fill from a LOCAL padded window and write back ONLY the masked
        # pixels — the rest of the page is never resampled or repainted.
        pad = int(np.clip(0.5 * max(w, h), 24, 200))
        wx0, wy0 = max(0, x - pad), max(0, y - pad)
        wx1, wy1 = min(W, x + w + pad), min(H, y + h + pad)
        sub = result[wy0:wy1, wx0:wx1]
        mwin = mask[wy0:wy1, wx0:wx1]
        out = None
        if self.lama is not None and self.lama.ok:
            out = self.lama.inpaint(sub, mwin)
        if out is None:
            out = cv2.inpaint(sub, mwin, 5, cv2.INPAINT_TELEA)
        m = mwin > 0
        sub[m] = out[m]
        return (x, y, w, h)

    def _stroke_halo_mask(self, gray_roi, seg_roi=None):
        """Tight mask of the LETTERING ONLY: glyph strokes plus their white
        outline/glow — and nothing else.

        The deviation mask marks ink of either polarity, but that includes
        ART lines running through the box, and inpainting those is exactly
        how a busy panel turns to mush. So the mask is built around a TEXT
        ANCHOR: the GPU seg strokes when available, else the bright glow
        mass (free text over art always wears one — that's what keeps it
        readable). Only deviation components touching the anchor survive;
        stray art lines away from the lettering are left alone. When there
        is no anchor and the background is flat paper (nothing to protect),
        every deviation is text and all of it is taken."""
        dev = self._ink_mask(gray_roi)
        h_, w_ = gray_roi.shape[:2]
        have_seg = seg_roi is not None and cv2.countNonZero(seg_roi) >= 10
        if cv2.countNonZero(dev) == 0 and not have_seg:
            return np.zeros((h_, w_), np.uint8)
        sigma = max(3.0, min(h_, w_) / 6.0)
        bg = cv2.GaussianBlur(cv2.medianBlur(gray_roi, 3), (0, 0), sigma)
        delta = gray_roi.astype(np.int16) - bg.astype(np.int16)
        # Glow = genuinely WHITE pixels that stand off the background. Big
        # dark glyphs drag the blurred background estimate down, which makes
        # plain mid-gray tone "deviate" — requiring near-white keeps tone out.
        glow = (((delta > 8) & (gray_roi >= 225) & (dev > 0)) * 255).astype(np.uint8)

        anchor = None
        if have_seg:
            anchor = cv2.bitwise_or(seg_roi, glow)
        elif cv2.countNonZero(glow) >= 30:
            anchor = glow
        if anchor is not None:
            near = cv2.dilate(anchor, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (13, 13)))
            # Candidates exclude the glow itself: glyph cores are then islands
            # inside their halo rings, cleanly separated from any surrounding
            # tone/art deviation they'd otherwise be 8-connected to.
            cand = cv2.bitwise_and(dev, cv2.bitwise_not(glow))
            n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
            strokes = anchor.copy()
            for i in range(1, n):
                cx_, cy_, cw_, ch_ = (int(stats[i, cv2.CC_STAT_LEFT]),
                                      int(stats[i, cv2.CC_STAT_TOP]),
                                      int(stats[i, cv2.CC_STAT_WIDTH]),
                                      int(stats[i, cv2.CC_STAT_HEIGHT]))
                comp = labels[cy_:cy_ + ch_, cx_:cx_ + cw_] == i
                # MOSTLY inside the anchor zone — an art line whose END pokes
                # into the glow must not drag its whole length into the mask.
                inside = float((near[cy_:cy_ + ch_, cx_:cx_ + cw_][comp] > 0).mean())
                if inside >= 0.45:
                    strokes[cy_:cy_ + ch_, cx_:cx_ + cw_][comp] = 255
        else:
            strokes = dev.copy()
            flat = gray_roi[dev == 0]
            if flat.size and float(np.std(flat)) > 14:
                # Textured art, no anchor to lean on: at least drop the
                # obvious long thin lines that cross most of the region.
                n, labels, stats, _ = cv2.connectedComponentsWithStats(dev, 8)
                long_side = max(h_, w_)
                for i in range(1, n):
                    span = max(int(stats[i, cv2.CC_STAT_WIDTH]),
                               int(stats[i, cv2.CC_STAT_HEIGHT]))
                    area = int(stats[i, cv2.CC_STAT_AREA])
                    if span > 0.6 * long_side and area / max(span, 1) < 4.5:
                        strokes[labels == i] = 0
        if cv2.countNonZero(strokes) == 0:
            return strokes

        # Halo band: catch the soft outer edge of the glow that falls under
        # the deviation threshold (left behind, it reads as a ghostly white
        # ring once the strokes vanish). Radius follows stroke thickness —
        # big title glyphs wear big glows. Only pixels BRIGHTER than the
        # local background are admitted, so dark art lines running next to
        # the text are never swallowed.
        dist = cv2.distanceTransform((strokes > 0).astype(np.uint8), cv2.DIST_L2, 3)
        vals = dist[dist > 0]
        r = int(np.clip(3.0 * float(np.median(vals)), 2, 24)) if vals.size else 3
        band = cv2.dilate(strokes, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        halo = np.where((delta > 8) & (gray_roi >= 210), band, 0).astype(np.uint8)
        mask = cv2.bitwise_or(strokes, halo)
        return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                          iterations=1)

    def _inpaint_text(self, result, x, y, w, h, contain=False):
        """Remove text from a free-text region. Builds the stroke mask from where
        the image deviates from its smooth background, so faint / low-contrast
        narration of either polarity is caught and fully covered, then inpaints.

        Normally the region is padded outward so characters that extend past the
        AI box are also cleaned. When `contain` is set (a box the USER drew/
        resized) NO outward padding is used — the edit stays strictly inside the
        box, so it never bleeds into surrounding art. Returns the rect actually
        touched (x, y, w, h), or None when the region is empty."""
        H, W = result.shape[:2]
        pad = 0 if contain else max(4, min(w, h) // 8)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        if x1 <= x0 or y1 <= y0:
            return None
        touched = (x0, y0, x1 - x0, y1 - y0)
        gray_roi = cv2.cvtColor(result[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        seg_roi = self._seg_mask[y0:y1, x0:x1] if self._seg_mask is not None else None
        tight = self._stroke_halo_mask(gray_roi, seg_roi)
        if cv2.countNonZero(tight) == 0:
            return touched

        # Letter groups: close small gaps so a column/line of glyphs shares one
        # LOCAL window (the closing shapes the windows, never the fill mask).
        # Each group is content-aware filled from its own padded surroundings:
        # the model sees the art right around the glyphs and continues its
        # lines through the thin stroke holes. Only masked pixels are written
        # back — the art between and around characters is never resampled.
        # (The old single whole-page pass replaced the ENTIRE page with the
        # model's resynthesis and turned huge text regions into flat mush.)
        gk = int(np.clip(max(x1 - x0, y1 - y0) // 40, 5, 31)) | 1
        groups = cv2.morphologyEx(tight, cv2.MORPH_CLOSE, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (gk, gk)))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(groups, 8)
        boxes = [(int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                  int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
                 for i in range(1, n)]
        if not boxes or len(boxes) > 12:
            boxes = [(0, 0, x1 - x0, y1 - y0)]

        tight_full = np.zeros((H, W), np.uint8)
        tight_full[y0:y1, x0:x1] = tight
        done = np.zeros((H, W), np.uint8)
        use_lama = self.lama is not None and self.lama.ok
        for bx, by, bw2, bh2 in boxes:
            gx, gy = x0 + bx, y0 + by
            wpad = int(np.clip(0.5 * max(bw2, bh2), 24, 160))
            wx0, wy0 = max(0, gx - wpad), max(0, gy - wpad)
            wx1, wy1 = min(W, gx + bw2 + wpad), min(H, gy + bh2 + wpad)
            mwin = tight_full[wy0:wy1, wx0:wx1]
            fresh = cv2.bitwise_and(mwin, cv2.bitwise_not(done[wy0:wy1, wx0:wx1]))
            if cv2.countNonZero(fresh) == 0:
                continue
            sub = result[wy0:wy1, wx0:wx1]
            out = self.lama.inpaint(sub, mwin) if use_lama else None
            if out is None:
                out = cv2.inpaint(sub, mwin, 5, cv2.INPAINT_TELEA)
            m = mwin > 0
            sub[m] = out[m]
            done[wy0:wy1, wx0:wx1] |= mwin
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
        # A real caption interior is enclosed by a DRAWN FRAME — verify the ink
        # ring is actually there. Bright haze on artwork (the glow around free
        # lettering, a hazy sky) also forms enclosed-looking blobs, but their
        # boundary is mid-gray tone, not a near-black line; stamping solid
        # white over those was the "correction slab" behind free text and
        # erased watermarks. Sample a thin ring just outside the component.
        comp = (labels == pick).astype(np.uint8) * 255
        ring = cv2.subtract(cv2.dilate(comp, np.ones((7, 7), np.uint8)), comp)
        ring_vals = roi[ring > 0]
        # Frame lines are INK (near-black) — mid-gray art or tone around a
        # bright patch is not a frame, however enclosed the patch looks.
        if ring_vals.size < 20 or float((ring_vals < 90).mean()) < 0.55:
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

    def _apply_free_region(self, result, gray, cap, bbox, contain=False):
        """Clear a planned free region and return (text_rect, dark, touched).
        `contain` keeps the erase strictly inside the box (user-drawn boxes).

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
            touched = self._inpaint_text(result, ix, iy, iw, ih, contain=contain) or (ix, iy, iw, ih)
            pad = max(3, min(iw, ih) // 12)
            rect = (ix + pad, iy + pad, max(iw - 2 * pad, 8), max(ih - 2 * pad, 8))
            return rect, True, touched
        rx, ry, rw, rh = [int(v) for v in bbox]
        touched = self._inpaint_text(result, rx, ry, rw, rh, contain=contain) or (rx, ry, rw, rh)
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
    def _poly_inner_rect(poly, w, h):
        """Largest comfortable axis-aligned rectangle INSIDE a user-drawn
        polygon, so a point-selected translation is typeset strictly within
        the shape the user outlined. Grows greedily from the polygon's
        incenter; returns (x, y, w, h) or None if the shape is too small."""
        try:
            pts = np.array([[int(p[0]), int(p[1])] for p in poly], np.int32)
        except (TypeError, ValueError, IndexError):
            return None
        bx, by, bw, bh = cv2.boundingRect(pts)
        bx, by = max(0, bx), max(0, by)
        bw, bh = min(bw, w - bx), min(bh, h - by)
        if bw < 8 or bh < 8:
            return None
        mask = np.zeros((bh, bw), np.uint8)
        cv2.fillPoly(mask, [pts - [bx, by]], 255)
        dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        _, r, _, (cx, cy) = cv2.minMaxLoc(dist)
        if r < 4:
            return None
        half = max(1, int(r * 0.7))          # inscribed square to start from
        x0, x1 = max(0, cx - half), min(bw - 1, cx + half)
        y0, y1 = max(0, cy - half), min(bh - 1, cy + half)

        def col_ok(xx, ya, yb):
            return 0 <= xx < bw and bool((mask[ya:yb + 1, xx] > 0).all())

        def row_ok(yy, xa, xb):
            return 0 <= yy < bh and bool((mask[yy, xa:xb + 1] > 0).all())

        moved = True
        while moved:
            moved = False
            if col_ok(x0 - 1, y0, y1):
                x0 -= 1; moved = True
            if col_ok(x1 + 1, y0, y1):
                x1 += 1; moved = True
            if row_ok(y0 - 1, x0, x1):
                y0 -= 1; moved = True
            if row_ok(y1 + 1, x0, x1):
                y1 += 1; moved = True
        rw, rh = x1 - x0 + 1, y1 - y0 + 1
        if rw < 8 or rh < 8:
            return None
        return (bx + x0, by + y0, rw, rh)

    def _poly_placement(self, poly, w, h):
        """Placement for a point-selected shape. If the outline is an elongated
        TILTED strip (a slanted title bar), return the strip's OWN box and
        angle so the text runs along the selection, filling it — the largest
        axis-aligned rectangle inside a thin diagonal strip is a tiny square,
        which crammed the text into one end. Otherwise fall back to the
        largest axis-aligned inside rectangle. Returns (rect_or_None, angle)."""
        try:
            pts = np.array([[float(p[0]), float(p[1])] for p in poly], np.float32)
        except (TypeError, ValueError):
            return None, 0.0
        if len(pts) < 3:
            return None, 0.0
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
        if rw < rh:
            rw, rh = rh, rw
            ang += 90.0
        while ang > 90.0:
            ang -= 180.0
        while ang <= -90.0:
            ang += 180.0
        if rw >= 1.7 * rh and abs(ang) <= 40.0:
            # Elongated strip (title bar / banner) — even a 1-2° tilt makes the
            # axis-aligned inside-rectangle collapse to a small box at the
            # strip's high end. Use the strip's OWN full-length box, at its own
            # angle, so the text runs along the whole selection.
            bw, bh = rw * 0.94, rh * 0.72
            rect = (int(cx - bw / 2), int(cy - bh / 2),
                    max(int(bw), 8), max(int(bh), 8))
            return rect, (float(ang) if abs(ang) >= 1.0 else 0.0)
        return self._poly_inner_rect(poly, w, h), 0.0

    def _estimate_text_angle(self, x, y, w, h):
        """Measure the tilt of the ORIGINAL lettering from its ink strokes, for
        free text the detector reported as horizontal. Returns a clockwise
        angle in degrees only when the ink is confidently a tilted, elongated
        block (a diagonal banner / slanted bar) — otherwise None, and the
        translation stays horizontal. Conservative by design: squarish
        paragraphs, steep verticals and sparse ink are all rejected."""
        if self._seg_mask is None:
            return None
        H, W = self._seg_mask.shape[:2]
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(W, int(x + w)), min(H, int(y + h))
        if x1 - x0 < 24 or y1 - y0 < 12:
            return None
        roi = (self._seg_mask[y0:y1, x0:x1] > 0).astype(np.uint8)
        pts = cv2.findNonZero(roi)
        if pts is None or len(pts) < 80:
            return None
        (_, _), (rw, rh), ang = cv2.minAreaRect(pts)
        if rw < rh:
            rw, rh = rh, rw
            ang += 90.0
        while ang > 90.0:
            ang -= 180.0
        while ang <= -90.0:
            ang += 180.0
        if rh <= 0 or rw < 2.2 * rh:
            return None      # not an elongated line/bar — angle unreliable
        if not (3.0 <= abs(ang) <= 40.0):
            return None      # horizontal enough, or too steep for English
        return float(ang)

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

    def _widen_vertical_rect(self, rect, result, used_boxes):
        """A tall-narrow free-text rect means the SOURCE was a vertical
        Japanese column. English renders horizontally, so auto-fitting it
        into the column forces a width-constrained, near-invisible font.
        Convert to a horizontal box: keep the column's center, make the box
        about two source glyphs tall (column width ~= one JP character, so
        the fitted English matches the source presence), and grow sideways
        only while the cleaned page under the band stays quiet — art strokes
        and panel borders stop the growth. Returns the original rect when
        the shape isn't a column or there's no room to win."""
        x, y, rw, rh = [int(v) for v in rect]
        orig = (x, y, rw, rh)
        H, W = result.shape[:2]
        if rw < 8 or rh < int(1.8 * rw):
            return orig
        char = rw
        cy = y + rh // 2
        nh = int(min(rh, max(2.6 * char, 24)))
        ny = max(0, min(cy - nh // 2, H - nh))
        band = cv2.cvtColor(result[ny:ny + nh], cv2.COLOR_BGR2GRAY)
        quiet = (band < 160).mean(axis=0) < 0.10
        limit = int(5 * char)
        left = x
        while left > max(0, x - limit) and quiet[left - 1]:
            left -= 1
        right = x + rw
        while right < min(W, x + rw + limit) and quiet[right]:
            right += 1
        pad = max(2, char // 8)
        left, right = left + pad, right - pad
        if right - left <= rw * 1.5:
            return orig
        cand = (int(left), int(ny), int(right - left), int(nh))
        for ub in used_boxes:
            if self._overlaps(orig, ub):
                continue  # our own planned box
            if self._overlaps(cand, ub):
                return orig
        return cand

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

        # The strictly-inscribed rectangle only covers ~70% of a round/oval
        # balloon, which makes short lines look tiny with lots of empty space.
        # Grow it partway toward the balloon's bounding box so the text fills the
        # bubble like real lettering (a little reach toward the curved edges is
        # fine — text rarely fills the very corners).
        ix, iy, iw, ih = px - l, py - t, l + r, t + b
        bx, by, bw, bh = cv2.boundingRect(mask)
        g = 0.45
        nx = int(round(ix - (ix - bx) * g))
        ny = int(round(iy - (iy - by) * g))
        nx2 = int(round((ix + iw) + ((bx + bw) - (ix + iw)) * g))
        ny2 = int(round((iy + ih) + ((by + bh) - (iy + ih)) * g))
        return (nx, ny, max(nx2 - nx, 8), max(ny2 - ny, 8))
