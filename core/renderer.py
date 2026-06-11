import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features
from typing import List, Tuple, Optional, Dict
import os


class TextRenderer:
    FONT_CANDIDATES = [
        "fonts/AnimeAce.ttf",
        "fonts/anime-ace.ttf",
        "fonts/Anime Ace.ttf",
        "fonts/animeace2_reg.ttf",
        "fonts/animeace_b.ttf",
        "fonts/animeace2_bld.ttf",
        "fonts/manga.ttf",
        "fonts/Bangers-Regular.ttf",
        "fonts/ComicNeue-Bold.ttf",
        "fonts/cc-wild-words.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]

    def __init__(self, font_path: Optional[str] = None, font_scale: float = 1.0,
                 uppercase: bool = True):
        self.font_path = font_path or self._find_font()
        self.font_scale = max(0.5, min(font_scale, 3.0))
        self.uppercase = uppercase
        self.min_font_size = 10
        self.padding_ratio = max(0.0, 0.04 / self.font_scale)
        self.line_spacing_ratio = max(0.04, 0.12 / self.font_scale)
        self._font_cache: Dict[tuple, ImageFont.FreeTypeFont] = {}
        self._cov_cache: Dict[str, Optional[set]] = {}
        self._active_font_path: Optional[str] = None
        # Right-to-left (Arabic / Hebrew) typesetting state for the current
        # draw_in_rect call. With libraqm, Pillow shapes + reorders natively when
        # we pass direction="rtl"; without it we pre-shape via arabic_reshaper +
        # python-bidi (if installed) and draw the visual-order string LTR.
        self._has_raqm = bool(features.check("raqm"))
        self._draw_dir: Optional[str] = None   # "rtl" when raqm handles a RTL line
        self._reshape_text = False             # True when we pre-shape (no raqm)
        self._mix = False                      # True = per-glyph font fallback (LTR)

    def _find_font(self) -> Optional[str]:
        for p in self.FONT_CANDIDATES:
            if os.path.exists(p):
                return p
        return None

    def _get_font(self, size: int, font_path: Optional[str] = None) -> ImageFont.FreeTypeFont:
        path = font_path or self._active_font_path or self.font_path
        key = (path, size)
        if key in self._font_cache:
            return self._font_cache[key]
        font = None
        if path:
            try:
                font = ImageFont.truetype(path, size)
            except (IOError, OSError):
                pass
        if font is None:
            font = ImageFont.load_default()
        self._font_cache[key] = font
        return font

    # ── Glyph coverage / fallback so ♪ ♫ ― … never render as boxes (□) ──
    _NORMALIZE = {
        "―": "—",   # ― horizontal bar → — em dash (comic fonts have —)
        "─": "—",   # ─ box-drawing dash → —
        "〜": "~",        # 〜 wave dash → ~
        "～": "~",        # ～ fullwidth tilde → ~
    }
    _FALLBACK_CANDIDATES = [
        "fonts/NotoSansSymbols2-Regular.ttf",
        "fonts/Symbola.ttf",
        # Arabic / RTL-capable fonts. A bundled comic-style Arabic face (drop one
        # into fonts/) wins; otherwise a proper Naskh, then DejaVu/FreeSerif —
        # all carry Arabic glyphs AND the GSUB tables raqm needs to join letters.
        "fonts/Arabic-Comic.ttf",
        "fonts/NotoNaskhArabic-Bold.ttf",
        "fonts/Amiri-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]

    # Unicode blocks that read right-to-left (Arabic family + Hebrew + the
    # Arabic presentation forms emitted by arabic_reshaper).
    _RTL_RANGES = (
        (0x0590, 0x05FF), (0x0600, 0x06FF), (0x0750, 0x077F),
        (0x08A0, 0x08FF), (0xFB1D, 0xFDFF), (0xFE70, 0xFEFF),
    )

    def _normalize_text(self, text: str) -> str:
        return "".join(self._NORMALIZE.get(c, c) for c in text)

    def _coverage(self, path: Optional[str]):
        """Codepoints a font file can render (None if it can't be introspected)."""
        if not path:
            return None
        if path in self._cov_cache:
            return self._cov_cache[path]
        cov = None
        try:
            from fontTools.ttLib import TTFont
            f = TTFont(path, fontNumber=0, lazy=True)
            cov = set()
            for tbl in f["cmap"].tables:
                cov.update(tbl.cmap.keys())
            f.close()
        except Exception:
            cov = None
        self._cov_cache[path] = cov
        return cov

    def _covers(self, path: Optional[str], chars: set) -> bool:
        cov = self._coverage(path)
        if cov is None:
            return True  # can't introspect — assume fine, don't switch fonts
        return all(ord(c) in cov for c in chars)

    def _fallback_font_paths(self):
        return [p for p in self._FALLBACK_CANDIDATES if os.path.exists(p)]

    def _effective_font_path(self, text: str) -> Optional[str]:
        """Pick a font that can actually render `text`. The comic font wins when
        it covers every character; otherwise fall back to a Unicode font that
        has the missing glyphs (♪ ♫ ― …) so they never show as □."""
        chars = {c for c in text if not c.isspace()}
        if not chars or self._covers(self.font_path, chars):
            return self.font_path
        for fb in self._fallback_font_paths():
            if self._covers(fb, chars):
                return fb
        # Nothing covers everything — use whichever renders the most characters.
        def score(p):
            cov = self._coverage(p)
            return sum(1 for c in chars if cov and ord(c) in cov)
        return max([self.font_path] + self._fallback_font_paths(), key=score)

    # ── Right-to-left (Arabic / Hebrew) shaping & ordering ──
    def _is_rtl(self, text: str) -> bool:
        for ch in text:
            o = ord(ch)
            if any(lo <= o <= hi for lo, hi in self._RTL_RANGES):
                return True
        return False

    def _dir_kw(self) -> dict:
        """direction kwarg for textbbox/textlength/text. Only set when libraqm
        is present — passing it without raqm raises in Pillow."""
        return {"direction": self._draw_dir} if self._draw_dir else {}

    def _shape(self, line: str) -> str:
        """For the no-raqm fallback only: turn logical Arabic into a visually
        ordered, glyph-joined string so a plain LTR draw looks correct. A no-op
        when raqm is doing the shaping (or the libs aren't installed)."""
        if not self._reshape_text or not line:
            return line
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
            return get_display(arabic_reshaper.reshape(line))
        except Exception:
            return line

    def _bbox(self, draw, s, font, stroke_width=0):
        return draw.textbbox((0, 0), self._shape(s), font=font,
                             stroke_width=stroke_width, **self._dir_kw())

    # ── Per-glyph font fallback (LTR): keep the comic font, borrow only the
    #    missing glyphs (♪ ♫ …) from a fallback so lettering stays in style ──
    def _glyph_font_path(self, ch: str) -> Optional[str]:
        if ch.isspace() or self._covers(self.font_path, {ch}):
            return self.font_path
        for fb in self._fallback_font_paths():
            if self._covers(fb, {ch}):
                return fb
        return self.font_path

    def _needs_mix(self, line: str) -> bool:
        return self._mix and any(
            not (c.isspace() or self._covers(self.font_path, {c})) for c in line
        )

    def _runs(self, line: str) -> List[Tuple[str, Optional[str]]]:
        """Split `line` into maximal runs that share a font (the comic font
        where it has the glyph, a fallback where it doesn't)."""
        runs: List[list] = []
        for ch in line:
            p = self._glyph_font_path(ch)
            if runs and runs[-1][1] == p:
                runs[-1][0] += ch
            else:
                runs.append([ch, p])
        return [(s, p) for s, p in runs]

    def _line_w(self, draw, line, font) -> int:
        """Ink width of a line, accounting for per-glyph fallback so wrapping
        and auto-sizing stay accurate."""
        if self._needs_mix(line):
            return int(sum(
                draw.textlength(s, font=self._get_font(font.size, p))
                for s, p in self._runs(line)
            ))
        bb = self._bbox(draw, line, font)
        return bb[2] - bb[0]

    def _draw_mixed_line(self, draw, top, inner_x, inner_w, line, font,
                         color, stroke_w, stroke_c):
        """Draw one LTR line whose glyphs span more than one font, laying the
        runs out left-to-right by advance width and centering the whole line."""
        runs = self._runs(line)
        fonts = [self._get_font(font.size, p) for _, p in runs]
        widths = [draw.textlength(s, font=f) for (s, _), f in zip(runs, fonts)]
        total = int(sum(widths))
        # Vertical placement mirrors the single-font path (ink top at `top`).
        bb = draw.textbbox((0, 0), line, font=font)
        ty = top - bb[1]
        cx = inner_x + max(0, (inner_w - total) // 2)
        for (s, _), f, wdt in zip(runs, fonts, widths):
            draw.text((cx, ty), s, fill=color, font=f,
                      stroke_width=stroke_w, stroke_fill=stroke_c)
            cx += wdt

    def render(
        self,
        image: np.ndarray,
        regions: list,
        translations: dict,
    ) -> np.ndarray:
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        for region in regions:
            if region.id not in translations:
                continue
            tr = translations[region.id]
            text = tr.get("translation", "").strip()
            if not text:
                continue
            pil_img = self._render_in_region(pil_img, region, text)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _render_in_region(
        self, image: Image.Image, region, text: str
    ) -> Image.Image:
        return self.draw_in_rect(image, region.bbox, text, (0, 0, 0))

    def draw_in_rect(
        self,
        image: Image.Image,
        rect: Tuple[int, int, int, int],
        text: str,
        color: Tuple[int, int, int] = (0, 0, 0),
        italic: bool = False,
        rotation: float = 0,
    ) -> Image.Image:
        """Fit `text` (wrapped, auto-sized, centered) inside `rect`.

        When `italic` is set the text is sheared into a slanted style — used to
        set sound effects / expression beats apart from ordinary dialogue.
        When `rotation` is nonzero the text is rendered upright into a temporary
        layer, rotated to the specified clockwise angle, and composited onto
        *image* so it sits inside the target rect at the same tilt as the
        original Japanese."""
        if self.uppercase:
            text = text.upper()
        text = self._normalize_text(text)
        # Use a font that can actually render every glyph in this text (so ♪, ―
        # etc. don't come out as boxes), for the whole call — wrap, measure, draw.
        prev_font = self._active_font_path
        prev_dir, prev_reshape, prev_mix = self._draw_dir, self._reshape_text, self._mix
        rtl = self._is_rtl(text)
        if rtl:
            # Right-to-left text must be shaped + reordered as one run, so pick a
            # single font that covers it. raqm does it via direction="rtl";
            # without raqm we pre-shape each line into visual order.
            self._active_font_path = self._effective_font_path(text)
            self._draw_dir = "rtl" if self._has_raqm else None
            self._reshape_text = not self._has_raqm
            self._mix = False
        else:
            # Left-to-right: keep the comic font and fill ONLY the glyphs it
            # lacks (♪ ♫ …) from a fallback, per glyph-run — so lettering stays
            # in-style instead of the whole line switching to a plain font.
            self._active_font_path = self.font_path or self._effective_font_path(text)
            self._draw_dir = None
            self._reshape_text = False
            self._mix = True
        try:
            return self._draw_in_rect_inner(image, rect, text, color, italic, rotation)
        finally:
            self._active_font_path = prev_font
            self._draw_dir, self._reshape_text, self._mix = prev_dir, prev_reshape, prev_mix

    def _draw_in_rect_inner(self, image, rect, text, color, italic, rotation):
        x, y, w, h = rect

        if abs(rotation) >= 2:
            return self._draw_rotated(image, x, y, w, h, text, color, italic, rotation)

        pad_x = max(int(w * self.padding_ratio), 1)
        pad_y = max(int(h * self.padding_ratio), 1)
        inner_x, inner_y = x + pad_x, y + pad_y
        inner_w, inner_h = w - 2 * pad_x, h - 2 * pad_y

        if inner_w < 16 or inner_h < 12:
            # Tight bubble — drop padding and use the whole rect. We never
            # bail out (a wiped-but-empty bubble loses content): a tiny bubble
            # gets text at the minimum size even if it slightly overflows.
            inner_x, inner_y = x + 1, y + 1
            inner_w, inner_h = max(w - 2, 1), max(h - 2, 1)

        draw = ImageDraw.Draw(image)
        font_size = self._optimal_size(text, inner_w, inner_h, draw)
        font = self._get_font(font_size)
        sw = max(1, font_size // 18) * 2
        wrap_w = max(inner_w - sw, 8)
        lines = self._wrap(text, font, wrap_w, draw)

        heights = []
        for line in lines:
            bb = self._bbox(draw, line, font)
            heights.append(bb[3] - bb[1])

        spacing = max(int(font_size * self.line_spacing_ratio), 1)
        total_h = sum(heights) + spacing * max(0, len(lines) - 1)

        cur_y = inner_y + max(0, (inner_h - total_h) // 2)

        stroke_c = (255, 255, 255) if color[0] < 128 else (0, 0, 0)
        stroke_w = max(1, font_size // 18)

        for i, line in enumerate(lines):
            if self._needs_mix(line) and not italic:
                # Line mixes the comic font with a fallback for glyphs it lacks.
                self._draw_mixed_line(draw, cur_y, inner_x, inner_w, line, font,
                                      color, stroke_w, stroke_c)
                cur_y += heights[i] + spacing
                continue
            bb = self._bbox(draw, line, font)
            lw = bb[2] - bb[0]
            lx = inner_x + max(0, (inner_w - lw) // 2)
            if italic:
                self._draw_italic_line(image, lx - bb[0], cur_y - bb[1], line,
                                       font, color, stroke_w, stroke_c)
            else:
                draw.text((lx - bb[0], cur_y - bb[1]), self._shape(line), fill=color,
                          font=font, stroke_width=stroke_w, stroke_fill=stroke_c,
                          **self._dir_kw())
            cur_y += heights[i] + spacing

        return image

    def _draw_rotated(self, image, x, y, w, h, text, color, italic, angle_deg):
        """Render *text* at *angle_deg* clockwise, composited onto *image*
        centered on the (x, y, w, h) rect.  The text is first drawn upright
        into an RGBA layer whose dimensions are chosen so that, after rotation,
        the result fits within the target rect."""
        rad = math.radians(abs(angle_deg))
        c, s = abs(math.cos(rad)), abs(math.sin(rad))

        det = c * c - s * s
        if abs(det) > 0.05:
            rw = (w * c - h * s) / det
            rh = (h * c - w * s) / det
        else:
            rw = rh = min(w, h) * 0.71

        if rw <= 0 or rh <= 0:
            rw = max(w, h)
            rh = max(16, min(w, h))

        rw = max(16, int(rw))
        rh = max(12, int(rh))

        tmp = Image.new("RGBA", (rw, rh), (0, 0, 0, 0))
        self.draw_in_rect(tmp, (0, 0, rw, rh), text, color, italic=italic, rotation=0)

        rotated = tmp.rotate(-angle_deg, expand=True, resample=Image.BICUBIC)

        if rotated.width > w or rotated.height > h:
            scale = min(w / max(rotated.width, 1), h / max(rotated.height, 1))
            nw = max(1, int(rotated.width * scale))
            nh = max(1, int(rotated.height * scale))
            rotated = rotated.resize((nw, nh), Image.BICUBIC)

        cx = x + w // 2
        cy = y + h // 2
        px = int(cx - rotated.width / 2)
        py = int(cy - rotated.height / 2)
        px = max(0, min(px, image.width - rotated.width))
        py = max(0, min(py, image.height - rotated.height))
        image.paste(rotated, (px, py), rotated)
        return image

    def _draw_italic_line(self, image, ax, ay, line, font, color, stroke_w, stroke_c):
        """Render one line slanted (faux-italic) and composite it onto `image`,
        landing where an upright draw.text((ax, ay), ...) would have placed it."""
        probe = ImageDraw.Draw(image)
        bb = self._bbox(probe, line, font, stroke_width=stroke_w)
        lw = max(bb[2] + stroke_w + 2, 1)
        lh = max(bb[3] + stroke_w + 2, 1)
        layer = Image.new("RGBA", (lw, lh), (0, 0, 0, 0))
        ImageDraw.Draw(layer).text(
            (0, 0), self._shape(line), font=font, fill=color + (255,),
            stroke_width=stroke_w, stroke_fill=stroke_c + (255,), **self._dir_kw(),
        )
        shear = 0.24
        ext = int(np.ceil(shear * lh))
        # AFFINE maps output->input: top rows sample further right, so the glyph
        # leans right while its baseline stays put.
        sheared = layer.transform(
            (lw + ext, lh), Image.AFFINE, (1, shear, -shear * lh, 0, 1, 0),
            resample=Image.BICUBIC,
        )
        image.paste(sheared, (int(ax), int(ay)), sheared)

    def _optimal_size(
        self, text: str, max_w: int, max_h: int, draw: ImageDraw.ImageDraw
    ) -> int:
        upper = min(max_w, max_h, 150)
        upper = max(upper, self.min_font_size)

        best = self.min_font_size
        lo, hi = self.min_font_size, upper

        while lo <= hi:
            mid = (lo + hi) // 2
            if self._text_fits(text, mid, max_w, max_h, draw):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1

        return best

    def _text_fits(
        self,
        text: str,
        size: int,
        max_w: int,
        max_h: int,
        draw: ImageDraw.ImageDraw,
    ) -> bool:
        font = self._get_font(size)
        sw = max(1, size // 18) * 2
        eff_w = max_w - sw
        eff_h = max_h - sw
        if eff_w < 8 or eff_h < 8:
            eff_w, eff_h = max_w, max_h
        lines = self._wrap(text, font, eff_w, draw)

        total = 0
        spacing = max(int(size * self.line_spacing_ratio), 1)
        for i, line in enumerate(lines):
            bb = self._bbox(draw, line, font)
            lw = self._line_w(draw, line, font)
            lh = bb[3] - bb[1]
            if lw > eff_w:
                return False
            total += lh
            if i < len(lines) - 1:
                total += spacing

        return total <= eff_h

    def _wrap(
        self,
        text: str,
        font: ImageFont.FreeTypeFont,
        max_w: int,
        draw: ImageDraw.ImageDraw,
    ) -> List[str]:
        """Wrap into lines of even length (professional manga lettering keeps
        lines balanced, not greedy-ragged). Greedy first to find the natural
        line count, then the narrowest width that still fits that count —
        every line stays ≤ max_w, so fit checks remain valid."""
        words = text.split()
        if not words:
            return [text]

        greedy = self._wrap_greedy(words, font, max_w, draw)
        if len(greedy) <= 1:
            return greedy

        k = len(greedy)
        longest = max(
            self._line_w(draw, wd, font) for wd in words
        )
        lo, hi = max(longest, 8), max_w
        best = greedy
        while lo < hi:
            mid = (lo + hi) // 2
            cand = self._wrap_greedy(words, font, mid, draw)
            if len(cand) <= k:
                best = cand
                hi = mid
            else:
                lo = mid + 1
        return best

    def _wrap_greedy(self, words, font, max_w, draw) -> List[str]:
        lines: List[str] = []
        current = ""
        for word in words:
            test = f"{current} {word}".strip() if current else word
            if self._line_w(draw, test, font) <= max_w:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines if lines else [" ".join(words)]
