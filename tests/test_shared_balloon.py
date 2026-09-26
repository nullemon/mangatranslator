"""Two blocks of text in ONE balloon are both lettered, right column first.

Found against TCB's release of One Piece 1194: a balloon holding two
columns (ブハァ!! ... beside ゲホ!! ...) lost its second line — it hit the
"balloon already used" check and was dropped.

Run:  python tests/test_shared_balloon.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.compositor import Compositor  # noqa: E402


def main():
    H, W = 700, 700
    page = np.full((H, W, 3), 200, np.uint8)
    rng = np.random.default_rng(1)
    for _ in range(1500):                     # art around the balloon
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        cv2.line(page, (x, y), (x + 25, y + 9), (70, 70, 70), 1)
    cv2.ellipse(page, (350, 350), (230, 170), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(page, (350, 350), (230, 170), 0, 0, 360, (0, 0, 0), 3)
    for x0 in (380, 200):                     # two blocks of vertical text
        for c in range(3):
            for k in range(6):
                cv2.rectangle(page, (x0 + c * 36, 225 + k * 42),
                              (x0 + c * 36 + 24, 250 + k * 42), (0, 0, 0), -1)
    right = dict(id=1, bbox=[372, 215, 118, 270], in_bubble=True, type="dialogue",
                 original="ブハァ", translation="BWAHH!! HUFF...")
    left = dict(id=2, bbox=[192, 215, 118, 270], in_bubble=True, type="dialogue",
                original="ゲホ", translation="KOFF!!")
    comp = Compositor(use_lama=False)
    drawn = []
    orig = comp.renderer.draw_in_rect

    def spy(image, rect, text, *a, **k):
        drawn.append(text)
        return orig(image, rect, text, *a, **k)
    comp.renderer.draw_in_rect = spy
    # listed left-first on purpose: reading order must still put the right column first
    comp.compose(page.copy(), [left, right])
    assert left.get("placed") and right.get("placed"), "both lines must be placed"
    print("drawn:", drawn)
    joined = " | ".join(drawn)
    assert "KOFF!!" in joined and "BWAHH!!" in joined, joined
    first = [t for t in drawn if "BWAHH" in t or "KOFF" in t][0]
    assert first.index("BWAHH") < first.index("KOFF"), f"right column first: {first!r}"
    print("both lines lettered in one balloon, right column first OK")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
