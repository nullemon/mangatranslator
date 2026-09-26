"""Move works on text inside a balloon.

Balloon text follows the balloon's shape, and that shape was taken from the
balloon's place on the page — a Move offset shifted the text's box but not
the shape, so the text was laid out right back where it was.

Run:  python tests/test_move_balloon.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.compositor import Compositor  # noqa: E402


def main():
    page = np.full((600, 700, 3), 190, np.uint8)
    rng = np.random.default_rng(5)
    for _ in range(1500):
        x, y = int(rng.integers(0, 700)), int(rng.integers(0, 600))
        cv2.line(page, (x, y), (x + 20, y + 8), (60, 60, 60), 1)
    cv2.ellipse(page, (330, 280), (170, 130), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(page, (330, 280), (170, 130), 0, 0, 360, (0, 0, 0), 3)
    for c in range(3):                         # three vertical "columns"
        for k in range(5):
            cv2.rectangle(page, (280 + c * 36, 200 + k * 34), (302 + c * 36, 224 + k * 34), (0, 0, 0), -1)
    item = lambda: [dict(id=1, bbox=[272, 190, 110, 180], in_bubble=True, type="dialogue",
                         original="テスト", translation="WHERE DID I PUT IT?")]
    comp = Compositor(use_lama=False)
    shapes = []
    orig = comp.renderer.draw_in_rect

    def spy(image, rect, text, *a, **k):
        shapes.append(comp.renderer._shape_mask is not None)
        return orig(image, rect, text, *a, **k)
    comp.renderer.draw_in_rect = spy
    a = comp.compose(page.copy(), item())
    b = comp.compose(page.copy(), item(), offsets={"1": [80, 40]})
    assert any(shapes), "the balloon-shaped layout should be in use for this test to mean anything"
    moved = int((cv2.absdiff(a, b).max(axis=2) > 40).sum())
    assert moved > 300, f"moving the line changed only {moved} px"
    ink_a = np.argwhere(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)[150:420, 160:520] < 60).mean(axis=0)
    ink_b = np.argwhere(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)[150:420, 160:600] < 60).mean(axis=0)
    assert ink_b[1] - ink_a[1] > 20, f"text did not move right: {ink_a} -> {ink_b}"
    print(f"balloon text moved ({moved} px changed) OK")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
