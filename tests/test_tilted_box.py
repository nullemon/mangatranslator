"""A TILTED BOX: the whole box turns with its text.

Tilting a line only turned its text, fitted so the turned text stayed inside
the LEVEL box: a wide banner line at -14° came out tiny and broken over
lines, and the banner's ends (outside the level box) kept their Japanese.
With "Tilt the whole box" the text is fitted to the box's own width and
height and turned with it, and the erase follows the turned box — and
nothing outside it.

Run:  python tests/test_tilted_box.py
"""
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.compositor import Compositor  # noqa: E402

W, H = 800, 500
BOX = [190, 220, 420, 60]            # level box, centre (400, 250)
ANGLE = -14.0


def along(t, off=0.0):
    """A point `t` px along the tilted banner's centre line, `off` px across."""
    a = math.radians(ANGLE)
    return (400 + t * math.cos(a) - off * math.sin(a),
            250 + t * math.sin(a) + off * math.cos(a))


def page_with_banner_text():
    img = np.full((H, W, 3), 245, np.uint8)
    for t in range(-190, 191, 24):                   # "lettering" along the slant
        x, y = along(t)
        cv2.rectangle(img, (int(x) - 7, int(y) - 12), (int(x) + 7, int(y) + 12), (0, 0, 0), -1)
    cv2.circle(img, (400, 400), 25, (0, 0, 0), 4)    # art well outside the box
    return img


def ink_box(before, after):
    d = np.any(np.abs(before.astype(int) - after.astype(int)) > 60, axis=2)
    ys, xs = np.nonzero(d)
    return (xs.min(), ys.min(), xs.max(), ys.max()) if xs.size else None


def letter(tilt_box):
    comp = Compositor(use_lama=False)
    comp.text_seg = None
    blank = np.full((H, W, 3), 245, np.uint8)
    it = dict(id=1, bbox=list(BOX), in_bubble=False, type="dialogue", original="クリア者",
              translation="CLEARERS ASSEMBLY ROOM", rotation=ANGLE, manual_rot=True,
              tilt_box=tilt_box, color="black")
    out = comp.compose(blank.copy(), [it])
    return blank, out, it


def main():
    # ── fitting: the text runs the length of the turned box ──
    b0, o0, _ = letter(False)
    b1, o1, it = letter(True)
    assert it.get("placed"), "tilted-box line not lettered"
    level, turned = ink_box(b0, o0), ink_box(b1, o1)
    lw, tw = level[2] - level[0], turned[2] - turned[0]
    assert tw >= 0.75 * BOX[2] * math.cos(math.radians(ANGLE)), \
        f"text doesn't run along the tilted box ({tw}px of ~{BOX[2]}px)"
    assert tw > 1.3 * lw, f"tilted box no bigger than text-only tilt ({tw} vs {lw}px)"
    print(f"text fills the tilted box ({tw}px wide vs {lw}px text-only) OK")

    # ── erase: follows the turned box, reaching past the level box ──
    comp = Compositor(use_lama=False)
    comp.text_seg = None
    page = page_with_banner_text()
    it = dict(id=1, bbox=list(BOX), in_bubble=False, type="dialogue", original="クリア者",
              translation="X", rotation=ANGLE, manual_rot=True, tilt_box=True)
    comp.renderer.draw_in_rect = lambda *a, **k: a[0]     # judge the erase alone
    out = comp.compose(page.copy(), [it])
    for t in (-178, 178):          # the banner's ends: outside the level box
        x, y = along(t)
        assert not (BOX[1] <= y <= BOX[1] + BOX[3]), "test point should be off the level box"
        patch = cv2.cvtColor(out[int(y) - 8:int(y) + 9, int(x) - 5:int(x) + 6], cv2.COLOR_BGR2GRAY)
        assert patch.min() > 150, f"banner end at t={t} not erased (min {patch.min()})"
    art = cv2.cvtColor(out[370:431, 370:431], cv2.COLOR_BGR2GRAY)
    assert (art == cv2.cvtColor(page[370:431, 370:431], cv2.COLOR_BGR2GRAY)).all(), \
        "art outside the tilted box changed"
    print("erase follows the tilted box and nothing outside it OK")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
