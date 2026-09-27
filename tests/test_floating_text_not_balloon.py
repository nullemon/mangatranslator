"""Lettering floating on the art is not a balloon.

A monologue lettered straight onto a white face has no outline round it,
but the balloon detector drew a blob round it; with text inside, the old
check saw "white paper plus lettering", called it a balloon and flat-filled
the blob white over the art. A real balloon — outline all round, however
full of text — must still pass.

Run:  python tests/test_floating_text_not_balloon.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.compositor import Compositor  # noqa: E402


def lettering(img, ink, x0, y0):
    for c in range(3):
        for r in range(5):
            p = (x0 + c * 30, y0 + r * 28)
            cv2.rectangle(img, p, (p[0] + 18, p[1] + 20), (0, 0, 0), -1)
            cv2.rectangle(ink, p, (p[0] + 18, p[1] + 20), 255, -1)


def main():
    comp = Compositor(use_lama=False)

    # a white face with a few art lines, text floating on it
    art = np.full((400, 400, 3), 245, np.uint8)
    cv2.line(art, (20, 60), (380, 90), (30, 30, 30), 3)
    cv2.circle(art, (330, 300), 40, (30, 30, 30), 3)
    ink = np.zeros((400, 400), np.uint8)
    lettering(art, ink, 130, 120)
    blob = np.zeros((400, 400), np.uint8)
    cv2.ellipse(blob, (160, 185), (80, 100), 0, 0, 360, 255, -1)
    comp._seg_mask = ink
    g = cv2.cvtColor(art, cv2.COLOR_BGR2GRAY)
    assert not comp._is_real_balloon(g, blob, True), \
        "text floating on the art was taken for a balloon"
    print(f"floating text not a balloon ({comp._balloon_why}) OK")

    # a real balloon: outlined, packed with text
    bal = np.full((400, 400, 3), 245, np.uint8)
    cv2.ellipse(bal, (160, 185), (80, 100), 0, 0, 360, (0, 0, 0), 4)
    ink2 = np.zeros((400, 400), np.uint8)
    lettering(bal, ink2, 130, 120)
    inside = np.zeros((400, 400), np.uint8)
    cv2.ellipse(inside, (160, 185), (76, 96), 0, 0, 360, 255, -1)
    comp._seg_mask = ink2
    g2 = cv2.cvtColor(bal, cv2.COLOR_BGR2GRAY)
    assert comp._is_real_balloon(g2, inside, True), \
        f"a real balloon was refused: {comp._balloon_why}"
    print("outlined balloon still a balloon OK")

    # a balloon drawn with a THIN (1 px, soft) outline, the mask running a
    # pixel or two inside it — most of chapter 1194's balloons; the old
    # band test read them 0-47% outlined and refused them
    thin = np.full((400, 400, 3), 245, np.uint8)
    cv2.ellipse(thin, (160, 185), (80, 100), 0, 0, 360, (90, 90, 90), 1, cv2.LINE_AA)
    ink3 = np.zeros((400, 400), np.uint8)
    lettering(thin, ink3, 130, 120)
    inside3 = np.zeros((400, 400), np.uint8)
    cv2.ellipse(inside3, (160, 185), (78, 98), 0, 0, 360, 255, -1)
    comp._seg_mask = ink3
    g3 = cv2.cvtColor(thin, cv2.COLOR_BGR2GRAY)
    assert comp._is_real_balloon(g3, inside3, True), \
        f"a thin-outlined balloon was refused: {comp._balloon_why}"
    print("thin-outlined balloon a balloon OK")

    # a FACE the detector took for a balloon, a line of text beside it:
    # outlined all round, but with eyes and a mouth inside — flat-filling it
    # would wipe the face
    face = np.full((400, 400, 3), 245, np.uint8)
    cv2.ellipse(face, (200, 200), (90, 120), 0, 0, 360, (0, 0, 0), 3)
    for ex in (165, 235):
        cv2.circle(face, (ex, 170), 14, (0, 0, 0), 3)
    cv2.ellipse(face, (200, 250), (40, 18), 0, 0, 180, (0, 0, 0), 3)
    for k in range(6):                        # cheek hatching
        for cx in (140, 245):
            cv2.line(face, (cx, 200 + 8 * k), (cx + 20, 190 + 8 * k), (0, 0, 0), 2)
    ink4 = np.zeros((400, 400), np.uint8)
    cv2.rectangle(face, (190, 290), (210, 305), (0, 0, 0), -1)
    cv2.rectangle(ink4, (190, 290), (210, 305), 255, -1)
    inside4 = np.zeros((400, 400), np.uint8)
    cv2.ellipse(inside4, (200, 200), (86, 116), 0, 0, 360, 255, -1)
    comp._seg_mask = ink4
    g4 = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    assert not comp._is_real_balloon(g4, inside4, True), \
        "a face with an outline round it was taken for a balloon"
    print(f"outlined face not a balloon ({comp._balloon_why}) OK")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
