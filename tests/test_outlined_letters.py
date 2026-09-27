"""Outlined lettering is recognised, erased whole, and lettered outlined.

Sounds in balloons and an editor's teaser on the art are often drawn as
white (or screentone) letters inside a black outline. The text model sees
little of them, so they were left half-erased and the English went on in
plain black. Ordinary black lettering — whose counters (口, 日) are enclosed
light shapes too — must not be taken for it.

Run:  python tests/test_outlined_letters.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.compositor import Compositor  # noqa: E402


def page_with(outlined):
    page = np.full((360, 420, 3), 255, np.uint8)
    for x in range(0, 420, 9):                       # speed lines behind
        cv2.line(page, (x, 0), (x, 359), (40, 40, 40), 1)
    cv2.rectangle(page, (60, 70), (360, 250), (255, 255, 255), -1)
    for c in range(4):
        x0 = 80 + c * 70
        if outlined:
            # a white letter body in a thick black outline
            cv2.rectangle(page, (x0 - 6, 94), (x0 + 46, 226), (0, 0, 0), -1)
            cv2.rectangle(page, (x0, 100), (x0 + 40, 220), (255, 255, 255), -1)
        else:
            # a black 口: a heavy stroke around a small counter
            cv2.rectangle(page, (x0, 110), (x0 + 44, 154), (0, 0, 0), -1)
            cv2.rectangle(page, (x0 + 10, 120), (x0 + 34, 144), (255, 255, 255), -1)
            cv2.rectangle(page, (x0 + 4, 170), (x0 + 40, 210), (0, 0, 0), -1)
    return page


def main():
    comp = Compositor(use_lama=False)
    box = [70, 88, 290, 144]

    pos = page_with(True)
    o = comp._outlined_glyphs(pos, cv2.cvtColor(pos, cv2.COLOR_BGR2GRAY), box)
    assert o is not None, "white letters in a black outline were not recognised"
    assert o["tone"] >= 230, f"body tone read as {o['tone']}, want white"
    assert o["outline"] >= 4, f"outline width read as {o['outline']} px, want the drawn 6"
    print(f"outlined letters found (tone {o['tone']}, outline {o['outline']} px) OK")

    neg = page_with(False)
    assert comp._outlined_glyphs(neg, cv2.cvtColor(neg, cv2.COLOR_BGR2GRAY), box) is None, \
        "black lettering was taken for outlined letters"
    print("black lettering not taken for outlined OK")

    item = [dict(id=1, bbox=box, in_bubble=False, type="dialogue", tone="shout",
                 original="パキパキ", translation="CRACK")]
    out = comp.compose(pos.copy(), item)
    g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    x, y, w, h = [int(v) for v in item[0]["bbox"]]
    roi = g[y:y + h, x:x + w]
    # no black outline of the old letters is left standing tall in the box
    cols = (cv2.cvtColor(pos, cv2.COLOR_BGR2GRAY)[100:220, 74:366] < 60).mean(axis=0)
    left = (g[100:220, 74:366] < 60).mean(axis=0)
    assert (left[cols > 0.9] > 0.9).sum() == 0, "an outline of the old letters is still standing"
    # the English is light letters in a dark outline: light pixels enclosed by dark
    light, dark = (roi > 200), (roi < 60)
    assert light.any() and dark.any(), "no outlined English drawn"
    print("old letters erased, English lettered outlined OK")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
