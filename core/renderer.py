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

    def __init__(self, font_path: Optional[str] = None):
        self.font_path = font_path or self._find_font()
        self.min_font_size = 8
        self.padding_ratio = 0.10
        self.line_spacing_ratio = 0.25
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
    ) -> Image.Image:
        """Fit `text` (wrapped, auto-sized, centered) inside `rect`."""
        x, y, w, h = rect

        pad_x = max(int(w * self.padding_ratio), 4)
        pad_y = max(int(h * self.padding_ratio), 4)
        inner_x, inner_y = x + pad_x, y + pad_y
        inner_w, inner_h = w - 2 * pad_x, h - 2 * pad_y

        if inner_w < 16 or inner_h < 12:
            # Tight bubble — drop padding and try anyway
            inner_x, inner_y = x + 2, y + 2
            inner_w, inner_h = max(w - 4, 1), max(h - 4, 1)
            if inner_w < 10 or inner_h < 8:
                return image

        draw = ImageDraw.Draw(image)
        font_size = self._optimal_size(text, inner_w, inner_h, draw)
        font = self._get_font(font_size)
        lines = self._wrap(text, font, inner_w, draw)

        heights = []
        for line in lines:
            bb = draw.textbbox((0, 0), line, font=font)
            heights.append(bb[3] - bb[1])

        spacing = max(int(font_size * self.line_spacing_ratio), 1)
        total_h = sum(heights) + spacing * max(0, len(lines) - 1)

        cur_y = inner_y + max(0, (inner_h - total_h) // 2)

        for i, line in enumerate(lines):
            bb = draw.textbbox((0, 0), line, font=font)
            lw = bb[2] - bb[0]
            lx = inner_x + max(0, (inner_w - lw) // 2)
            # offset by the bbox origin so the glyph sits where we expect
            draw.text((lx - bb[0], cur_y - bb[1]), line, fill=color, font=font)
            cur_y += heights[i] + spacing

        return image

    def _optimal_size(
        self, text: str, max_w: int, max_h: int, draw: ImageDraw.ImageDraw
    ) -> int:
        upper = min(max_w, max_h // 2, 72)
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
        lines = self._wrap(text, font, max_w, draw)

        total = 0
        spacing = max(int(size * self.line_spacing_ratio), 1)
        for i, line in enumerate(lines):
            bb = draw.textbbox((0, 0), line, font=font)
            lw = bb[2] - bb[0]
            lh = bb[3] - bb[1]
            if lw > max_w:
                return False
            total += lh
            if i < len(lines) - 1:
                total += spacing

        return total <= max_h

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
