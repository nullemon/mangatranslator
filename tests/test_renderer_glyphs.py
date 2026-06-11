"""Per-glyph font-fallback checks for TextRenderer.

Comic display fonts (Bangers, ComicNeue) lack glyphs the translator emits —
the horizontal-bar dash and music notes. This verifies:
  1. substitutable chars (―, ～) are normalized to comic-font equivalents, and
  2. genuinely pictographic glyphs (♪) are drawn from a fallback font WHILE the
     surrounding letters stay in the comic font (per-glyph mixing, not a
     whole-line font switch).
Run: python3 tests/test_renderer_glyphs.py"""
import os
import sys

import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.renderer import TextRenderer


def main():
    r = TextRenderer(font_path="fonts/Bangers-Regular.ttf", uppercase=True)
    assert r.font_path and os.path.exists(r.font_path), "no comic font found"

    # 1) Horizontal bar / fullwidth tilde get normalized to supported glyphs.
    assert r._normalize_text("a―b") == "a—b", "― should map to em dash —"
    assert r._normalize_text("a～b") == "a~b", "～ should map to ~"
    print("normalization OK")

    # 2) Per-glyph mixing: letters stay in the comic font, only ♪ swaps out.
    comic = os.path.basename(r.font_path)
    r._mix = True  # set by draw_in_rect for LTR text; force on for the unit check
    runs = [(s, os.path.basename(p) if p else p) for s, p in r._runs("SING ♪")]
    assert runs[0][1] == comic, f"letters should stay comic, got {runs}"
    assert runs[-1][0] == "♪" and runs[-1][1] != comic, f"♪ should use fallback, got {runs}"
    assert r._needs_mix("SING ♪") and not r._needs_mix("HELLO")
    print(f"per-glyph mixing OK (♪ via {runs[-1][1]}, letters via {comic})")

    # 3) End-to-end: rendering both offenders lays down real ink, no crash.
    img = np.full((140, 420, 3), 255, np.uint8)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil = r.draw_in_rect(pil, (10, 10, 400, 120), "Wait― sing ♪", (0, 0, 0))
    out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    ink = int((out < 200).any(axis=2).sum())
    assert ink > 500, f"expected rendered ink, got {ink}"
    print(f"end-to-end render OK ({ink} ink px)")

    print("ALL CHECKS PASSED ✓")


if __name__ == "__main__":
    main()
