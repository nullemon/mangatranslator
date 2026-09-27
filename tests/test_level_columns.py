"""One Piece: a big vertical Japanese column is lettered LEVEL, like TCB.

A column like だが待て!! (tall, fat glyphs) was typeset sideways (-90°) —
the default for other series. TCB never turns dialogue sideways, so the One
Piece preset turns that off, for the first render and the editor's
re-renders alike.

Run:  python tests/test_level_columns.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core import series  # noqa: E402
from core.compositor import Compositor  # noqa: E402

FONT = os.path.join(os.path.dirname(__file__), "..", "fonts", "ReggaeOne-Regular.ttf")


def rotations(sideways):
    H, W = 900, 1200
    page = np.full((H, W, 3), 245, np.uint8)
    rng = np.random.default_rng(3)
    for _ in range(600):                      # light art around the column
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        cv2.line(page, (x, y), (x + 20, y + 7), (90, 90, 90), 1)
    x0, y0 = 560, 200                         # one fat vertical column: だが待て!!
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(page)
    d = ImageDraw.Draw(im)
    ft = ImageFont.truetype(FONT, 76)
    for k, ch in enumerate("だが待て！"):
        d.text((x0 + 7, y0 + k * 88), ch, font=ft, fill=(0, 0, 0))
    page = np.array(im)
    it = dict(id=1, bbox=[x0, y0, 90, 450], in_bubble=False, type="dialogue",
              original="だが待て！！", translation="WAIT A SEC!!")
    comp = Compositor(use_lama=False, sideways_columns=sideways)
    seen = []
    orig = Compositor._rotated_aabb

    def spy(rect, rot):
        seen.append(rot)
        return orig(rect, rot)
    comp._rotated_aabb = spy
    comp.compose(page, [it])
    return seen


def main():
    assert series.lettering('SERIES: this page is from "One Piece".') == {"sideways_columns": False}
    assert series.lettering(key="one piece") == {"sideways_columns": False}
    assert series.lettering("no series") == {}
    print("One Piece preset keeps columns level OK")

    default = rotations(True)
    assert any(abs(r) > 45 for r in default), \
        f"other series: a big column should still go sideways (got {default})"
    level = rotations(False)
    assert level and all(abs(r) < 3 for r in level), f"One Piece: lettered at {level}"
    print("big vertical column: sideways by default, level for One Piece OK")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
