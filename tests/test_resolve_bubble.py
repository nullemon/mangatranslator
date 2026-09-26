"""Regression: a translation must never be typeset into ANOTHER balloon.

Found on a real One Piece page: a small balloon holding one big bold kanji
(豹) had its text box centred on a glyph stroke, so balloon recovery fell back
to "the largest enclosed white region in the search window" — and the retry
window reaches 600 px, so it returned a big burst balloon in another panel.
The translation landed there, and that balloon's own line was skipped as a
collision ("box 1's translation goes in box 2").

Run:  python tests/test_resolve_bubble.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.compositor import Compositor  # noqa: E402


def main():
    H, W = 1400, 1000
    page = np.full((H, W), 255, np.uint8)
    # Art everywhere except the balloons, so the page itself isn't "white".
    rng = np.random.default_rng(0)
    for _ in range(4000):
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        cv2.line(page, (x, y), (x + 30, y + 12), 90, 1)

    # Small balloon with ONE fat glyph filling its middle (centre on ink).
    cv2.ellipse(page, (300, 400), (70, 80), 0, 0, 360, 255, -1)
    cv2.ellipse(page, (300, 400), (70, 80), 0, 0, 360, 0, 3)
    cv2.rectangle(page, (270, 350), (330, 450), 0, -1)          # the "kanji"
    small_box = [262, 342, 76, 116]

    # Big balloon further down (inside the 600 px retry window).
    cv2.ellipse(page, (560, 900), (150, 140), 0, 0, 360, 255, -1)
    cv2.ellipse(page, (560, 900), (150, 140), 0, 0, 360, 0, 3)
    for i in range(4):
        cv2.putText(page, "TEXT", (480, 850 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, 0, 3)

    c = Compositor(use_lama=False)
    r = c._resolve_bubble(page, small_box, H * W)
    assert r is not None, "the small balloon should be recovered"
    rx, ry, rw, rh = r[1]
    cx, cy = small_box[0] + small_box[2] // 2, small_box[1] + small_box[3] // 2
    assert rx <= cx <= rx + rw and ry <= cy <= ry + rh, \
        f"recovered {r[1]} does not contain its own text box {small_box}"
    assert rw < 200 and rh < 220, f"recovered the big balloon {r[1]} instead"
    print(f"small balloon -> {r[1]} (its own) OK")

    big_box = [470, 820, 180, 160]
    r2 = c._resolve_bubble(page, big_box, H * W)
    assert r2 is not None and r2[1][2] > 250, f"big balloon lost: {r2 and r2[1]}"
    print(f"big balloon   -> {r2[1]} OK")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
