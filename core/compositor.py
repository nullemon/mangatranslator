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
                            # Modest, single line, with a halo — never a giant
                            # slab-filling banner in the middle of the art.
                            wh_cap = max(16, int(0.035 * h))
                            wx_, wy_, ww_, whh_ = rect
                            if whh_ > wh_cap:
                                wy_ += (whh_ - wh_cap) // 2
                                whh_ = wh_cap
                            placements.append(((wx_, wy_, ww_, whh_),
                                               " ".join(self.watermark_text.split()),
                                               self._pick_color(dark, it), False, 0, 1.0, True))
                            used_boxes.append((int(wx_), int(wy_), int(ww_), int(whh_)))
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
                # Same halo rule as auto free text: floating letters get a
                # contrasting stroke; only a light caption fill stays plain.
                mglow = (self._item_glow(it) or cap is None
                         or (cap is not None and cap[4]))
                placements.append((offset_rect(it, rect), text, color, ital, rotation,
                               self._item_scale(it), mglow))
                it["placed"] = True
                continue

            if it.get("in_bubble") is False:
                bx = max(0, min(bx, w - 1))
                by = max(0, min(by, h - 1))
                bw = min(bw, w - bx)
                bh = min(bh, h - by)
                if bw < 10 or bh < 10:
                    continue

                # Giant title banner → ONE modest centred caption, pro style.
                # The banner is erased cleanly (a light caption band flat-fills;
                # anything else gets the stroke-tight inpaint) and the English
                # is set SMALL and centred — never auto-fitted up to the size
                # of the artwork lettering, never stamped word-by-word.
                if it.get("title_caption"):
                    cap, bb = self._plan_free_region(gray, bx, by, bw, bh,
                                                     refine=False)
                    if any(self._overlaps(bb, ub) for ub in used_boxes):
                        continue
                    used_boxes.append(bb)
                    _r, dark, touched = self._apply_free_region(result, gray,
                                                                cap, bb)
                    edited_rects.append(tuple(int(v) for v in touched))
                    ch = min(bh, max(int(0.034 * h), 22))
                    rect = (bx + bw // 14, by + max(0, (bh - ch) // 2),
                            bw - bw // 7, ch)
                    it["bbox"] = [int(v) for v in rect]
                    color = self._pick_color(dark, it)
                    tglow = (self._item_glow(it) or cap is None
                             or (cap is not None and cap[4]))
                    placements.append((offset_rect(it, rect),
                                       " ".join(text.split()), color, ital, 0,
                                       self._item_scale(it), tglow))
                    it["placed"] = True
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
                    elif it.get("src_rect"):
                        # Re-render of a box WE computed: take it as-is. The
                        # inset below is for raw AI boxes; applying it again
                        # every render shrank the box a few px per edit until
                        # it re-grew — a visible breathing loop.
                        rect = (bx, by, bw, bh)
                    else:
                        pad = max(3, min(bw, bh) // 12)
                        rect = (bx + pad, by + pad,
                                max(bw - 2 * pad, 8), max(bh - 2 * pad, 8))
                # Vertical source column (すごい… style): a tall-narrow rect
                # width-crushes horizontal English into a tiny font. Re-shape it
                # into a horizontal box at the column's center, sized to the
                # SOURCE glyphs, growing sideways only over quiet background.
                # Stable source basis: the FIRST pass's tight rect, persisted
                # on the item. Without it, a re-render would measure the
                # source glyphs from the already-grown box and grow again —
                # every edit inflating the caption until it spans the page.
                if it.get("src_rect") and len(it["src_rect"]) == 4:
                    src_rect = tuple(int(v) for v in it["src_rect"])
                else:
                    src_rect = tuple(int(v) for v in rect)
                    it["src_rect"] = list(src_rect)
                own_boxes = [tuple(int(v) for v in bb),
                             tuple(int(v) for v in rect)]
                # TALL SINGLE-COLUMN source (one vertical JP run, like a
                # monologue down the page edge): set the English VERTICALLY,
                # reading bottom-to-top — far better than a skinny horizontal
                # stack squeezed into the column. Wide sources (multiple JP
                # columns side by side) keep normal horizontal wrapping.
                if (abs(rotation) < 3 and not it.get("manual_rot")
                        and src_rect[3] >= 3.2 * src_rect[2]
                        and src_rect[2] <= 0.14 * w
                        and len(" ".join(text.split())) >= 12):
                    rotation = -90
                    it["rotation"] = -90
                if abs(rotation) < 3 and not it.get("manual_rot"):
                    wided = self._widen_vertical_rect(rect, result, used_boxes,
                                                      own_boxes)
                    if wided != tuple(int(v) for v in rect):
                        rect = wided
                        used_boxes.append(tuple(int(v) for v in rect))
                        own_boxes.append(tuple(int(v) for v in rect))
                # Pro presence: grow the box over quiet background until the
                # English renders at ~70% of the source glyph size. Never for
                # SFX (pros keep those small beside the art), and never into
                # the box of an item that hasn't been placed yet — a bubble
                # processed later must not find its spot already eaten.
                if (abs(rotation) <= 20 and not it.get("manual_rot")
                        and kind not in SFX_TYPES):
                    avoid = used_boxes + [
                        tuple(int(v) for v in o["bbox"]) for o in items
                        if o is not it and o.get("bbox") and not o.get("placed")
                    ]
                    grown = self._grow_for_presence(rect, src_rect, text, it,
                                                    result, avoid, own_boxes)
                    if grown != tuple(int(v) for v in rect):
                        rect = grown
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
                # Floating text always wears a contrasting stroke halo, like
                # every pro release — bare letters vanish into the art. Only
                # a light caption fill (clean paper behind) stays plain.
                fglow = (self._item_glow(it) or cap is None
                         or (cap is not None and cap[4]))
                placements.append((offset_rect(it, rect), text, color, ital, rotation,
                               self._item_scale(it), fglow))
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

            # Balloon seg sometimes claims big haloed display text as a
            # "bubble"; flat-filling that mask stamps a giant blob over the
            # art. Only wipe interiors that really look like balloon paper —
            # anything else goes through the free-text machinery below.
            if mask is not None and not self._is_real_balloon(gray, mask):
                # Not a balloon (headset mic, ornament, art blob). If nothing
                # was ever OCR'd here there is no text to move — placing the
                # LLM's stray translation would stamp English on a prop. Skip
                # the item and leave the art alone.
                if not (it.get("original") or "").strip():
                    continue
                mask = None

            fglow = False
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
                # No reliable balloon. Treat like floating text: framed white
                # interiors still get a clean caption fill; text over art has
                # only its strokes healed, sized to the source, with a halo.
                cap, pb = self._plan_free_region(gray, bx, by, bw, bh, refine=True)
                if any(self._overlaps(pb, ub) for ub in used_boxes):
                    continue
                used_boxes.append(pb)
                rect, dark, touched = self._apply_free_region(result, gray, cap, pb)
                bb = tuple(int(v) for v in touched)
                if cap is None:
                    seg_r = self._seg_text_rect(bx, by, bw, bh)
                    if seg_r is not None:
                        sx2, sy2, sw2, sh2 = seg_r
                        p2 = max(3, min(sw2, sh2) // 10)
                        rect = (sx2 - p2, sy2 - p2,
                                max(sw2 + 2 * p2, 8), max(sh2 + 2 * p2, 8))
                if it.get("src_rect") and len(it["src_rect"]) == 4:
                    src_rect = tuple(int(v) for v in it["src_rect"])
                else:
                    src_rect = tuple(int(v) for v in rect)
                    it["src_rect"] = list(src_rect)
                if not it.get("manual_rot") and kind not in SFX_TYPES:
                    avoid = used_boxes + [
                        tuple(int(v) for v in o["bbox"]) for o in items
                        if o is not it and o.get("bbox") and not o.get("placed")
                    ]
                    grown = self._grow_for_presence(
                        rect, src_rect, text, it, result, avoid,
                        [tuple(int(v) for v in pb), tuple(int(v) for v in rect)])
                    if grown != tuple(int(v) for v in rect):
                        rect = grown
                        used_boxes.append(tuple(int(v) for v in rect))
                it["bbox"] = [int(v) for v in rect]
                if rect[2] >= 3 * rect[3]:
                    text = " ".join(text.split())
                fglow = (self._item_glow(it) or cap is None
                         or (cap is not None and cap[4]))

            edited_rects.append(tuple(int(v) for v in bb))
            color = self._pick_color(dark, it)
            placements.append((offset_rect(it, rect), text, color, ital,
                               rotation if it.get("manual_rot") else 0,
                               self._item_scale(it), fglow or self._item_glow(it)))
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

    def _is_real_balloon(self, gray, mask):
        """A real balloon's interior (minus the text strokes) is near-uniform
        paper enclosed by an inked outline. Balloon segmentation sometimes
        claims big haloed DISPLAY TEXT as a bubble — its "interior" is art —
        and flat-filling that stamps a giant blob over the panel."""
        if mask is None or cv2.countNonZero(mask) < 40:
            return False
        inner = cv2.erode(mask, np.ones((5, 5), np.uint8))
        if cv2.countNonZero(inner) < 40:
            inner = mask
        vals = gray[inner > 0]
        med = float(np.median(vals))
        if med >= 165:
            body = vals[vals > 120]     # paper side, strokes excluded
        elif med <= 90:
            body = vals[vals < 120]     # black balloon
            # A black BALLOON carries light lettering; a solid black prop
            # (headset mic, silhouette) doesn't. No light strokes = not a
            # bubble.
            if float((vals > 180).mean()) < 0.02:
                return False
        else:
            return False                # mid-gray interior = artwork
        if body.size < 50:
            return False
        if float(np.std(body)) > 22.0:
            return False
        # Decisive signature: a balloon keeps a clean paper MARGIN between
        # its lettering and the outline; artwork's lines run right across
        # that ring. Without this, line art on white paper reads as "flat
        # paper + strokes" and gets flat-filled into a giant blob.
        if med >= 165:
            _, _, mw, mh = cv2.boundingRect(mask)
            depth = max(6, int(0.08 * min(mw, mh)))
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (2 * depth + 1, 2 * depth + 1))
            core = cv2.erode(mask, k)
            ring = (inner > 0) & (core == 0)
            if int(ring.sum()) >= 40:
                ring_dark = float((gray[ring] < 120).mean())
                if ring_dark > 0.06:
                    return False
        return True

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
        # Clamp to the page: a lasso drawn partly (or fully) off-page must
        # not produce an empty window slice and crash the whole compose.
        x0c, y0c = max(0, x), max(0, y)
        x1c, y1c = min(W, x + w), min(H, y + h)
        if x1c - x0c < 3 or y1c - y0c < 3:
            return None
        x, y, w, h = x0c, y0c, x1c - x0c, y1c - y0c
        # Fill from a LOCAL padded window and write back ONLY the masked
        # pixels — the rest of the page is never resampled or repainted.
        pad = int(np.clip(0.5 * max(w, h), 24, 200))
        wx0, wy0 = max(0, x - pad), max(0, y - pad)
        wx1, wy1 = min(W, x + w + pad), min(H, y + h + pad)
        sub = result[wy0:wy1, wx0:wx1]
        mwin = mask[wy0:wy1, wx0:wx1]
        if cv2.countNonZero(mwin) == 0:
            return (x, y, w, h)
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
        how a busy panel turns to mush. Decision tree:

        - FLAT paper background (low spread, sparse deviation): everything
          that deviates is lettering — take all of it. This also covers big
          bold text on white pages, where the blurred background estimate
          gets dragged down and even plain paper "deviates".
        - TEXTURED art: drop long thin lines crossing the region (art
          strokes, speed lines), then anchor on real lettering — the GPU seg
          strokes when substantial, else the near-white glow mass free text
          wears over art — and keep only deviation groups near that anchor.
          The proximity radius scales with the anchor's stroke thickness so
          bold glyph cores aren't orphaned.
        - No anchor on textured art: best effort — the long-thin filter
          alone (the historical behavior minus obvious art lines).
        """
        h_, w_ = gray_roi.shape[:2]
        if h_ < 3 or w_ < 3:
            return np.zeros((max(h_, 1), max(w_, 1)), np.uint8)
        # One background estimate shared by deviation, glow and halo (it was
        # computed twice per region before — the dominant CPU cost here).
        sigma = max(3.0, min(h_, w_) / 6.0)
        bg = cv2.GaussianBlur(cv2.medianBlur(gray_roi, 3), (0, 0), sigma)
        diff = cv2.absdiff(gray_roi, bg)
        _, dev = cv2.threshold(diff, 14, 255, cv2.THRESH_BINARY)
        dev = cv2.morphologyEx(
            dev, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        have_seg = seg_roi is not None and cv2.countNonZero(seg_roi) >= 10
        dev_px = cv2.countNonZero(dev)
        gsrc = gray_roi
        if dev_px > 0.45 * h_ * w_:
            # Deviation fires on nearly half the region — classic screentone:
            # every dot "deviates", the map is useless, and treating it as
            # text wipes the whole tone field flat. Melt the periodic texture
            # with an escalating median and re-measure; only structures
            # thicker than the tone dots (i.e. the lettering) survive.
            bk_ = 31 if min(h_, w_) >= 31 else (min(h_, w_) // 2 * 2 + 1)
            for mk_ in (5, 7, 9):
                gs_ = cv2.medianBlur(gray_roi, mk_)
                # Large-MEDIAN background: unlike a Gaussian mean it doesn't
                # dip next to big dark lettering, so the paper around the
                # text doesn't spuriously "deviate" and get wiped with it.
                bg_ = cv2.medianBlur(gs_, bk_)
                dv_ = cv2.absdiff(gs_, bg_)
                _, dv_ = cv2.threshold(dv_, 24, 255, cv2.THRESH_BINARY)
                # Safety net for glyph cores wider than the background window
                # (the median absorbs them): near-black ink is always text
                # on a melted mid/bright field.
                dv_ = cv2.bitwise_or(dv_, ((gs_ < 70) * 255).astype(np.uint8))
                dv_ = cv2.morphologyEx(
                    dv_, cv2.MORPH_OPEN,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
                gsrc, bg, dev = gs_, bg_, dv_
                dev_px = cv2.countNonZero(dev)
                if dev_px <= 0.30 * h_ * w_:
                    break
        if dev_px == 0 and not have_seg:
            return np.zeros((h_, w_), np.uint8)
        delta = gsrc.astype(np.int16) - bg.astype(np.int16)
        long_side = max(h_, w_)

        nontext = gsrc[dev == 0]
        flat = (nontext.size > 0 and float(np.std(nontext)) <= 14
                and dev_px <= 0.35 * h_ * w_)
        if flat:
            # Even on flat paper, a long thin line crossing the region is a
            # panel border or art stroke passing through — not lettering.
            strokes = self._drop_long_thin(dev, long_side)
        else:
            base = self._drop_long_thin(dev, long_side)
            # Glow = genuinely WHITE pixels standing off the background (big
            # dark glyphs drag the background estimate down, so requiring
            # near-white keeps plain tone out). Long-thin filtering keeps
            # white speed lines on dark panels from posing as glow.
            glow = (((delta > 8) & (gsrc >= 225) & (dev > 0)) * 255).astype(np.uint8)
            glow = self._drop_long_thin(glow, long_side)
            if cv2.countNonZero(glow) > 0.35 * h_ * w_:
                # A "glow" covering a third of the region is background paper
                # showing between tone dots, not a text halo.
                glow = np.zeros_like(glow)
            seg_px = cv2.countNonZero(seg_roi) if have_seg else 0
            anchor = None
            if have_seg and seg_px >= 60 and seg_px >= 0.05 * max(dev_px, 1):
                anchor = seg_roi
            elif cv2.countNonZero(glow) >= 30:
                anchor = glow
            if anchor is not None:
                # Proximity radius scales with the anchor's stroke thickness
                # so the middle of a fat bold stroke still counts as "near".
                adist = cv2.distanceTransform((anchor > 0).astype(np.uint8),
                                              cv2.DIST_L2, 3)
                avals = adist[adist > 0]
                t = float(np.median(avals)) if avals.size else 2.0
                k = int(np.clip(13 + 6 * t, 13, 61)) | 1
                near = cv2.dilate(anchor, cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (k, k)))
                # Candidates exclude the glow: glyph cores become islands
                # inside their halo rings, cleanly separated from tone/art
                # deviation they'd otherwise be 8-connected to.
                cand = cv2.bitwise_and(base, cv2.bitwise_not(glow))
                n, labels, _st, _c = cv2.connectedComponentsWithStats(cand, 8)
                lab = labels.ravel()
                tot = np.bincount(lab, minlength=n).astype(np.float64)
                ins = np.bincount(lab, weights=(near.ravel() > 0).astype(np.float64),
                                  minlength=n)
                frac = ins / np.maximum(tot, 1)
                keep = frac >= 0.30
                keep[0] = False
                kept = (keep[labels] * 255).astype(np.uint8)
                strokes = cv2.bitwise_or(anchor, kept)
            else:
                strokes = base
        if have_seg:
            strokes = cv2.bitwise_or(strokes, seg_roi)
        if cv2.countNonZero(strokes) == 0:
            return strokes

        # Halo band: catch the soft outer edge of the glow that falls under
        # the deviation threshold (left behind, it reads as a ghostly white
        # ring once the strokes vanish). Radius follows stroke thickness —
        # big title glyphs wear big glows. Only near-white pixels brighter
        # than the local background are admitted, so dark art lines running
        # next to the text are never swallowed.
        dist = cv2.distanceTransform((strokes > 0).astype(np.uint8), cv2.DIST_L2, 3)
        vals = dist[dist > 0]
        r = int(np.clip(3.0 * float(np.median(vals)), 2, 24)) if vals.size else 3
        band = cv2.dilate(strokes, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        # Gate the band against ITS OWN surroundings: the feathered outer
        # skirt of a glow sits under the deviation threshold (the blurred
        # background estimate absorbs the glow) but is still clearly brighter
        # than the tone just outside the band — left behind it reads as a
        # ghost ring. Absolute floor 160 keeps mid-gray art on dark panels
        # from being swallowed.
        ring = cv2.subtract(cv2.dilate(band, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (9, 9))), band)
        rvals = gsrc[ring > 0]
        localbg = float(np.median(rvals)) if rvals.size else float(np.median(gsrc))
        gate = max(localbg + 10.0, 160.0)
        halo = np.where((band > 0) & (gsrc.astype(np.float32) > gate), 255, 0).astype(np.uint8)
        mask = cv2.bitwise_or(strokes, halo)
        return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                          iterations=1)

    def _drop_long_thin(self, mask, long_side):
        """Remove components that run most of the way across the region while
        staying thin — art lines, speed lines, panel borders."""
        if cv2.countNonZero(mask) == 0:
            return mask
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        keep = np.ones(n, bool)
        keep[0] = False
        dropped = False
        for i in range(1, n):
            span = max(int(stats[i, cv2.CC_STAT_WIDTH]),
                       int(stats[i, cv2.CC_STAT_HEIGHT]))
            area = int(stats[i, cv2.CC_STAT_AREA])
            if span > 0.6 * long_side and area / max(span, 1) < 8.0:
                keep[i] = False
                dropped = True
        if not dropped:
            return mask
        return (keep[labels] * 255).astype(np.uint8)

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
        # HARD constraint for AUTOMATIC erasure only: with the text-pixel
        # model present, only pixels IT calls lettering may be erased — the
        # local deviation mask alone happily eats hair and face lines when a
        # det box sits on a character (the melted-head bug). But a USER-DRAWN
        # region (contain=True — the eraser/cover tools) is an explicit
        # command: the model gets no veto there, or dragging over a leftover
        # the model doesn't recognise would silently do nothing ("erase
        # doesn't work").
        if seg_roi is not None and not contain:
            if cv2.countNonZero(seg_roi) >= 40:
                allow = cv2.dilate(seg_roi, np.ones((9, 9), np.uint8))
                tight = cv2.bitwise_and(tight, allow)
            else:
                tight[:] = 0
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

        done = np.zeros_like(tight)      # roi-sized; the mask lives in the roi
        use_lama = self.lama is not None and self.lama.ok
        for bx, by, bw2, bh2 in boxes:
            gx, gy = x0 + bx, y0 + by
            wpad = int(np.clip(0.5 * max(bw2, bh2), 24, 160))
            wx0, wy0 = max(0, gx - wpad), max(0, gy - wpad)
            wx1, wy1 = min(W, gx + bw2 + wpad), min(H, gy + bh2 + wpad)
            # overlap of the (context-padded) window with the roi
            ox0, oy0 = max(wx0, x0), max(wy0, y0)
            ox1, oy1 = min(wx1, x1), min(wy1, y1)
            if ox1 <= ox0 or oy1 <= oy0:
                continue
            tsub = tight[oy0 - y0:oy1 - y0, ox0 - x0:ox1 - x0]
            dsub = done[oy0 - y0:oy1 - y0, ox0 - x0:ox1 - x0]
            if cv2.countNonZero(cv2.bitwise_and(tsub, cv2.bitwise_not(dsub))) == 0:
                continue
            mwin = np.zeros((wy1 - wy0, wx1 - wx0), np.uint8)
            mwin[oy0 - wy0:oy1 - wy0, ox0 - wx0:ox1 - wx0] = tsub
            sub = result[wy0:wy1, wx0:wx1]
            out = self.lama.inpaint(sub, mwin) if use_lama else None
            if out is None:
                out = cv2.inpaint(sub, mwin, 5, cv2.INPAINT_TELEA)
            m = mwin > 0
            sub[m] = out[m]
            dsub |= tsub
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
        ring = cv2.subtract(cv2.dilate(comp, np.ones((3, 3), np.uint8)), comp)
        # Frame lines are INK (near-black) — mid-gray art or tone around a
        # bright patch is not a frame, however enclosed the patch looks.
        # Sample the DARKEST pixel within 2px of each boundary point so even
        # a crisp 1px frame registers everywhere along the ring.
        gmin = cv2.erode(roi, np.ones((5, 5), np.uint8))
        ring_vals = gmin[ring > 0]
        if ring_vals.size < 20 or float((ring_vals < 90).mean()) < 0.6:
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

    def _em_ratio(self):
        """Average glyph advance of the ACTIVE font in em, measured once.
        The box math used to assume 0.62em; comic display faces run much
        narrower (Anton ~0.40) and the auto-fit then overshot the target
        size by 30-45%."""
        emr = getattr(self, "_emr", None)
        if emr is not None:
            return emr
        emr = 0.62
        try:
            font = self.renderer._get_font(100)
            sample = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            adv = font.getlength(sample) / len(sample)
            emr = float(min(0.8, max(0.3, adv / 100.0 * 1.12)))
        except Exception:
            pass
        self._emr = emr
        return emr

    def _est_fit(self, text, w, h):
        """Rough estimate of the font size draw_in_rect will settle on for
        `text` in a w x h box (measured glyph advance, line height ~1.22em)."""
        t = " ".join((text or "").split())
        if not t or w < 8 or h < 8:
            return 0.0
        em = self._em_ratio()
        n = max(len(t), 1)
        best = 0.0
        for lines in range(1, 13):
            f = min(h / (lines * 1.22), w * lines / (n * em))
            best = max(best, f)
        return best

    def _grow_for_presence(self, rect, src_rect, text, it, result, used_boxes,
                           own_boxes=()):
        """Scanlators size free text to the SOURCE lettering's visual weight,
        not to whatever sliver the detector found. Estimate the source glyph
        size from the original region and its character count; if the English
        would auto-fit well below ~70% of that, grow the box over quiet
        cleaned background — stopping at art, panel borders and other
        translations — until it can render at the target size."""
        x, y, rw, rh = [int(v) for v in rect]
        orig = (x, y, rw, rh)
        t = " ".join((text or "").split())
        if not t or rw < 8 or rh < 8:
            return orig
        H, W = result.shape[:2]
        sx, sy, sw, sh = [int(v) for v in src_rect]
        src = (it.get("original") or "").strip()
        if not src:
            # No OCR text -> no glyph-size estimate. Guessing from the region
            # alone maximizes the target and blows up short interjections.
            return orig
        n_src = max(sum(1 for c in src if not c.isspace()), 1)
        char_px = (max(sw, 1) * max(sh, 1) / n_src) ** 0.5
        char_px = float(min(max(char_px, 14.0), 110.0))
        em = self._em_ratio()
        # Pros conserve the source block's FOOTPRINT: cap the target so the
        # English occupies at most ~1.5x the source lettering's area. A
        # two-character hand-lettered aside (huge glyphs) translated into a
        # full sentence must not become a five-line 76px paragraph.
        n_en = max(len(t), 1)
        f_area = ((1.5 * max(sw, 1) * max(sh, 1)) / (em * 1.22 * n_en)) ** 0.5
        target = min(0.70 * char_px, f_area)
        if target < 11:
            return orig
        cur = self._est_fit(t, rw, rh)
        if cur >= 0.92 * target:
            # Wide acceptance band: re-renders re-pad the stored box a
            # little each time, and a tight band makes grow/shrink ping-pong.
            if cur <= 1.45 * target:
                return orig
            # The box is far too roomy (e.g. a widened column) and the
            # auto-fit would overshoot the source size — shrink to the
            # needed box, centered inside the current one. Always safe:
            # a subset of an already-approved box.
            for lines in range(1, 13):
                nw = int(len(t) * em * target / lines) + 4
                nh = int(lines * 1.22 * target) + 4
                if nw <= rw and nh <= rh:
                    return (int(x + (rw - nw) / 2),
                            int(y + (rh - nh) / 2), nw, nh)
            return orig
        # Deterministic growth: from here on, anchor every computation on
        # the STABLE source basis, so repeated re-renders derive the exact
        # same box instead of ping-ponging between wrap arrangements as the
        # stored (already grown, then re-refined) bbox mutates each pass.
        x, y = sx, sy
        rw, rh = max(sw, 8), max(sh, 8)
        # How far may we grow? Scan quiet rows/cols on the cleaned page.
        Lx, Ly = int(rw * 1.8) + 24, int(rh * 1.2) + 24
        wx0, wy0 = max(0, x - Lx), max(0, y - Ly)
        wx1, wy1 = min(W, x + rw + Lx), min(H, y + rh + Ly)
        if wx1 - wx0 < 8 or wy1 - wy0 < 8:
            return orig
        win = cv2.cvtColor(result[wy0:wy1, wx0:wx1], cv2.COLOR_BGR2GRAY)
        rows = slice(max(y, wy0) - wy0, max(min(y + rh, wy1) - wy0, max(y, wy0) - wy0 + 1))
        cols = slice(max(x, wx0) - wx0, max(min(x + rw, wx1) - wx0, max(x, wx0) - wx0 + 1))
        quiet_col = (win[rows] < 160).mean(axis=0) < 0.10
        quiet_row = (win[:, cols] < 160).mean(axis=1) < 0.10
        left, right = x, x + rw
        while left - 1 >= wx0 and quiet_col[left - 1 - wx0]:
            left -= 1
        while right < wx1 and quiet_col[right - wx0]:
            right += 1
        top, bot = y, y + rh
        while top - 1 >= wy0 and quiet_row[top - 1 - wy0]:
            top -= 1
        while bot < wy1 and quiet_row[bot - wy0]:
            bot += 1
        # Second pass over the FULL bands the box will actually occupy —
        # the first walk only checked the original rect's rows/columns, so
        # art sitting diagonally (new rows x new columns) slipped through.
        rows2 = slice(max(top, wy0) - wy0, max(min(bot, wy1) - wy0,
                                               max(top, wy0) - wy0 + 1))
        quiet_col = (win[rows2] < 160).mean(axis=0) < 0.10
        left, right = x, x + rw
        while left - 1 >= wx0 and quiet_col[left - 1 - wx0]:
            left -= 1
        while right < wx1 and quiet_col[right - wx0]:
            right += 1
        cols2 = slice(max(left, wx0) - wx0, max(min(right, wx1) - wx0,
                                                max(left, wx0) - wx0 + 1))
        quiet_row = (win[:, cols2] < 160).mean(axis=1) < 0.10
        top, bot = y, y + rh
        while top - 1 >= wy0 and quiet_row[top - 1 - wy0]:
            top -= 1
        while bot < wy1 and quiet_row[bot - wy0]:
            bot += 1
        pad = max(3, int(char_px) // 8)
        left, right = left + pad, right - pad
        top, bot = top + pad, bot - pad
        availW, availH = right - left, bot - top
        cx, cy = x + rw / 2.0, y + rh / 2.0
        own = {tuple(int(v) for v in b) for b in (own_boxes or ())}
        own.add(orig)

        def clear_of_others(c):
            for ub in used_boxes:
                if tuple(int(v) for v in ub) in own:
                    continue  # our own planned/widened box, not a neighbor
                if self._overlaps(c, ub):
                    return False
            return True

        # Tier 1: the smallest box that reaches the target size using QUIET
        # background only (prefer fewer lines).
        quiet = None
        if availW > rw or availH > rh:
            need = None
            for lines in range(1, 13):
                nw = int(len(t) * em * target / lines) + 4
                nh = int(lines * 1.22 * target) + 4
                if nw <= availW and nh <= availH:
                    need = (nw, nh)
                    break
            if need is None and self._est_fit(t, availW, availH) > self._est_fit(t, rw, rh):
                need = (availW, availH)
            if need is not None:
                nw, nh = need
                nx = int(min(max(left, cx - nw / 2.0), right - nw))
                ny = int(min(max(top, cy - nh / 2.0), bot - nh))
                cand = (nx, ny, int(nw), int(nh))
                croi = win[max(ny, wy0) - wy0:max(ny + int(nh), wy0) - wy0,
                           max(nx, wx0) - wx0:max(nx + int(nw), wx0) - wx0]
                if croi.size and float((croi < 160).mean()) < 0.10 and clear_of_others(cand):
                    quiet = cand
        best = quiet if quiet is not None else orig
        if self._est_fit(t, best[2], best[3]) >= 0.55 * target:
            return best
        # Tier 2: no quiet room anywhere (dense crowd/detail panels). Pros
        # typeset AT SIZE right on the art and let the stroke halo carry
        # readability — tiny fine-print in a busy panel reads far worse than
        # haloed text over it. Center the needed box on the source; only
        # other text still blocks it.
        bx0, by0 = wx0 + 4, wy0 + 4
        bx1, by1 = wx1 - 4, wy1 - 4
        for lines in range(1, 13):
            nw = int(len(t) * em * target / lines) + 4
            nh = int(lines * 1.22 * target) + 4
            if nw <= bx1 - bx0 and nh <= by1 - by0:
                nx = int(min(max(bx0, cx - nw / 2.0), bx1 - nw))
                ny = int(min(max(by0, cy - nh / 2.0), by1 - nh))
                cand2 = (nx, ny, int(nw), int(nh))
                if clear_of_others(cand2):
                    return cand2
                break
        return best

    def _widen_vertical_rect(self, rect, result, used_boxes, own_boxes=()):
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
        own = {tuple(int(v) for v in b) for b in (own_boxes or ())}
        own.add(orig)
        for ub in used_boxes:
            if tuple(int(v) for v in ub) in own:
                continue  # our own planned box, not a neighbor
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
