"""A sentence read across two JOINED balloons is split between them,
right-hand lobe first, at a natural break — and a single balloon is never
split. (TCB, One Piece 1194: "THERE IS ANOTHER TYPE OF HAKI..." right,
"ONE WHICH CANNOT BE ACQUIRED THROUGH TRAINING." left.)

Run:  python tests/test_joined_balloon.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.compositor import Compositor  # noqa: E402


def page_with(shapes, texts):
    H, W = 700, 900
    page = np.full((H, W, 3), 190, np.uint8)
    rng = np.random.default_rng(3)
    for _ in range(2000):
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        cv2.line(page, (x, y), (x + 20, y + 8), (60, 60, 60), 1)
    fill = np.zeros((H, W), np.uint8)
    for (cx, cy, rx, ry) in shapes:
        cv2.ellipse(fill, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    page[fill > 0] = 255
    edge = cv2.subtract(cv2.dilate(fill, np.ones((5, 5), np.uint8)), fill)
    page[edge > 0] = 0
    for (x0, y0) in texts:                     # vertical "Japanese" columns
        for c in range(3):
            for k in range(5):
                cv2.rectangle(page, (x0 + c * 30, y0 + k * 34), (x0 + c * 30 + 20, y0 + k * 34 + 22), (0, 0, 0), -1)
    return page


def draws(comp, page, item):
    out = []
    orig = comp.renderer.draw_in_rect

    def spy(image, rect, text, *a, **k):
        out.append((tuple(int(v) for v in rect), text))
        return orig(image, rect, text, *a, **k)
    comp.renderer.draw_in_rect = spy
    comp.compose(page.copy(), [item])
    comp.renderer.draw_in_rect = orig
    return out


class _Seg:
    """The text model's stroke mask: exactly the drawn columns."""
    ok = True

    def __init__(self, page):
        g = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        self.m = np.zeros(g.shape, np.uint8)
        # RETR_LIST: the balloon outline encloses the letters, so an
        # external-only search would never see them
        for c in cv2.findContours((g < 30).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]:
            x, y, w, h = cv2.boundingRect(c)
            if w <= 22 and h <= 24:
                self.m[y:y + h, x:x + w] = 255

    def mask(self, _img):
        return self.m

    def text_mask(self, _img):
        return self.m

    def detect_blocks(self, _img):
        return []


def main():
    text = "THERE IS ANOTHER TYPE OF HAKI... ONE WHICH CANNOT BE ACQUIRED THROUGH TRAINING."
    # two balloons drawn touching (a waist between them)
    joined = page_with([(330, 350, 170, 150), (570, 350, 170, 150)], [(250, 270), (500, 270)])
    comp = Compositor(use_lama=False); comp.text_seg = _Seg(joined)
    d = draws(comp, joined, dict(id=1, bbox=[240, 260, 360, 180], in_bubble=True,
                                 type="dialogue", original="もう一つの", translation=text))
    d = list(dict.fromkeys(d))
    assert len(d) == 2, f"joined balloon should hold two blocks, got {d}"
    right, left = sorted(d, key=lambda r: -r[0][0])
    assert right[1] == "THERE IS ANOTHER TYPE OF HAKI...", right
    assert left[1] == "ONE WHICH CANNOT BE ACQUIRED THROUGH TRAINING.", left
    print("joined balloon: split at the natural break, right lobe first OK")
    # one oval with the same two columns: ONE block, never split
    single = page_with([(450, 350, 300, 160)], [(300, 270), (500, 270)])
    comp = Compositor(use_lama=False); comp.text_seg = _Seg(single)
    d = list(dict.fromkeys(draws(comp, single, dict(id=1, bbox=[290, 260, 320, 180], in_bubble=True,
                                                   type="dialogue", original="もう一つの", translation=text))))
    assert len(d) == 1 and d[0][1] == text, f"single balloon must stay one block, got {d}"
    print("single balloon: one block OK")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
