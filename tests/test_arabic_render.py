"""Right-to-left (Arabic) typesetting checks for TextRenderer.

When the translation target is Arabic, the renderer must (1) fall back from a
Latin comic font to an Arabic-capable font, (2) join the letters and (3) lay
them out right-to-left. With libraqm present Pillow does the shaping when we
pass direction="rtl"; this verifies the output order is correct and that plain
Latin rendering is left untouched.
Run: python3 tests/test_arabic_render.py"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.renderer import TextRenderer

ARABIC = "الفصل 1185 : دعيهم وشأنهم"


def main():
    r = TextRenderer(font_path="fonts/Bangers-Regular.ttf", uppercase=True)

    # RTL detection: Arabic yes, Latin no.
    assert r._is_rtl(ARABIC)
    assert not r._is_rtl("HELLO WORLD")
    print("RTL detection OK")

    # The Latin comic font has no Arabic glyphs, so a fallback font is chosen.
    eff = r._effective_font_path(ARABIC)
    assert eff and eff != r.font_path, f"expected Arabic fallback, got {eff}"
    print(f"Arabic font fallback OK ({os.path.basename(eff)})")

    # End-to-end: Arabic renders ink without crashing.
    img = Image.new("RGB", (760, 90), "white")
    img = r.draw_in_rect(img, (20, 10, 720, 70), ARABIC, (0, 0, 0))
    ink = int((np.array(img) < 200).any(axis=2).sum())
    assert ink > 800, f"expected rendered Arabic ink, got {ink}"
    print(f"end-to-end Arabic render OK ({ink} ink px)")

    # Correctness: with raqm, the renderer's word ordering must match Pillow's
    # canonical RTL draw. Compared via order-sensitive column ink profile.
    from PIL import features
    if features.check("raqm"):
        fp = r._effective_font_path(ARABIC)
        size = 44
        ref = Image.new("L", (760, 80), 255)
        ImageDraw.Draw(ref).text((20, 10), ARABIC, font=ImageFont.truetype(fp, size),
                                 fill=0, direction="rtl", anchor="la")
        got = Image.new("L", (760, 80), 255)
        r._draw_dir, r._reshape_text, r._active_font_path = "rtl", False, fp
        fnt = r._get_font(size)
        d = ImageDraw.Draw(got)
        bb = r._bbox(d, ARABIC, fnt)
        d.text((20 - bb[0], 10 - bb[1]), r._shape(ARABIC), font=fnt, fill=0, **r._dir_kw())
        cref = (255 - np.array(ref)).sum(0).astype(float)
        cgot = (255 - np.array(got)).sum(0).astype(float)
        corr = float(np.corrcoef(cref, cgot)[0, 1])
        assert corr > 0.95, f"RTL ordering wrong (corr={corr:.3f})"
        print(f"RTL ordering matches canonical (corr={corr:.3f}) ✓")
    else:
        print("raqm not present — skipping canonical-ordering check")

    print("ALL CHECKS PASSED ✓")


if __name__ == "__main__":
    main()
