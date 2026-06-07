import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
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

    def __init__(self, font_path: Optional[str] = None, font_scale: float = 1.0):
        self.font_path = font_path or self._find_font()
        self.font_scale = max(0.5, min(font_scale, 3.0))
        self.min_font_size = 10
        self.padding_ratio = max(0.0, 0.04 / self.font_scale)
        self.line_spacing_ratio = max(0.04, 0.12 / self.font_scale)
        self._font_cache: Dict[int, ImageFont.FreeTypeFont] = {}

    def _find_font(self) -> Optional[str]:
        for p in self.FONT_CANDIDATES:
            if os.path.exists(p):
                return p
        return None

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        if size in self._font_cache:
            return self._font_cache[size]
        font = None
        if self.font_path:
            try:
                font = ImageFont.truetype(self.font_path, size)
            except (IOError, OSError):
                pass
        if font is None:
            font = ImageFont.load_default()
        self._font_cache[size] = font
        return font

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
    ) -> Image.Image:
        """Fit `text` (wrapped, auto-sized, centered) inside `rect`.

        When `italic` is set the text is sheared into a slanted style — used to
        set sound effects / expression beats apart from ordinary dialogue."""
        text = text.upper()
        x, y, w, h = rect

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
            bb = draw.textbbox((0, 0), line, font=font)
            heights.append(bb[3] - bb[1])

        spacing = max(int(font_size * self.line_spacing_ratio), 1)
        total_h = sum(heights) + spacing * max(0, len(lines) - 1)

        cur_y = inner_y + max(0, (inner_h - total_h) // 2)

        stroke_c = (255, 255, 255) if color[0] < 128 else (0, 0, 0)
        stroke_w = max(1, font_size // 18)

        for i, line in enumerate(lines):
            bb = draw.textbbox((0, 0), line, font=font)
            lw = bb[2] - bb[0]
            lx = inner_x + max(0, (inner_w - lw) // 2)
            if italic:
                self._draw_italic_line(image, lx - bb[0], cur_y - bb[1], line,
                                       font, color, stroke_w, stroke_c)
            else:
                draw.text((lx - bb[0], cur_y - bb[1]), line, fill=color, font=font,
                          stroke_width=stroke_w, stroke_fill=stroke_c)
            cur_y += heights[i] + spacing

        return image

    def _draw_italic_line(self, image, ax, ay, line, font, color, stroke_w, stroke_c):
        """Render one line slanted (faux-italic) and composite it onto `image`,
        landing where an upright draw.text((ax, ay), ...) would have placed it."""
        probe = ImageDraw.Draw(image)
        bb = probe.textbbox((0, 0), line, font=font, stroke_width=stroke_w)
        lw = max(bb[2] + stroke_w + 2, 1)
        lh = max(bb[3] + stroke_w + 2, 1)
        layer = Image.new("RGBA", (lw, lh), (0, 0, 0, 0))
        ImageDraw.Draw(layer).text(
            (0, 0), line, font=font, fill=color + (255,),
            stroke_width=stroke_w, stroke_fill=stroke_c + (255,),
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
            bb = draw.textbbox((0, 0), line, font=font)
            lw = bb[2] - bb[0]
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
        words = text.split()
        if not words:
            return [text]

        lines: List[str] = []
        current = ""

        for word in words:
            test = f"{current} {word}".strip() if current else word
            bb = draw.textbbox((0, 0), test, font=font)
            if bb[2] - bb[0] <= max_w:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word

        if current:
            lines.append(current)

        return lines if lines else [text]
