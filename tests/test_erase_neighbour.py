"""Erasing a line leaves the lettering next to it alone.

The automatic erase looks at a window padded past the line's box. On a
chapter's last page that window reached the end badge under a caption, and
the badge's "ONE PIECE" and its white frame were erased with the caption:
every text stroke in the window was taken, and so was any near-white pixel
(allowed through as "glow behind the letters").

Run:  python tests/test_erase_neighbour.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.compositor import Compositor  # noqa: E402


def main():
    H, W = 320, 320
    page = np.full((H, W, 3), 120, np.uint8)            # grey tone art
    rng = np.random.default_rng(3)
    for _ in range(400):
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        cv2.line(page, (x, y), (x + 12, y + 5), (70, 70, 70), 1)
    # the caption: dark glyphs in a white glow, box (60, 60, 150, 100)
    cv2.rectangle(page, (60, 60), (210, 160), (255, 255, 255), -1)
    ink = np.zeros((H, W), np.uint8)
    for c in range(4):
        for r in range(2):
            p0 = (75 + c * 34, 75 + r * 40)
            cv2.rectangle(page, p0, (p0[0] + 20, p0[1] + 26), (0, 0, 0), -1)
            cv2.rectangle(ink, p0, (p0[0] + 20, p0[1] + 26), 255, -1)
    # the neighbour: a white badge with its own lettering just below
    cv2.rectangle(page, (40, 163), (260, 225), (255, 255, 255), -1)
    cv2.rectangle(page, (40, 163), (260, 225), (0, 0, 0), 3)
    for c in range(5):
        p0 = (70 + c * 36, 166)
        cv2.rectangle(page, p0, (p0[0] + 22, p0[1] + 24), (0, 0, 0), -1)
        cv2.rectangle(ink, p0, (p0[0] + 22, p0[1] + 24), 255, -1)

    comp = Compositor(use_lama=False)

    class Fill:                    # stands in for the inpainting model:
        ok = True                  # paints every masked pixel pure white

        @staticmethod
        def inpaint(img, mask):
            return np.full_like(img, 255)
    comp.lama = Fill()
    comp._seg_mask = ink.copy()
    comp._dialog_mask = ink.copy()
    comp._raw_mask = ink.copy()
    out = page.copy()
    comp._inpaint_text(out, 60, 60, 150, 100)

    gray_in, gray_out = (cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) for a in (page, out))
    glyphs = ink[:150] > 0
    hit = int((cv2.absdiff(gray_in, gray_out)[:150] > 40)[glyphs].sum())
    assert hit > 0.8 * int(glyphs.sum()), f"the caption was not erased ({hit}/{int(glyphs.sum())} px)"
    badge = cv2.absdiff(gray_in, gray_out)[160:230, 36:264] > 40
    assert int(badge.sum()) == 0, f"erasing the caption changed {int(badge.sum())} px of the badge below"
    print("caption erased, badge below untouched OK")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
