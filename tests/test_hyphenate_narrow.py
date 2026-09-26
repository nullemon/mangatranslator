"""A narrow balloon breaks one long word instead of shrinking all its text.

Run:  python tests/test_hyphenate_narrow.py
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.renderer import TextRenderer  # noqa: E402

FONT = os.path.join(os.path.dirname(__file__), "..", "fonts", "ComicNeue-Bold.ttf")


def render(r, text, shape):
    img = Image.new("RGB", (120, 300), "white")
    r._shape_mask = shape
    r.draw_in_rect(img, (5, 5, 110, 290), text, (0, 0, 0))
    return r._last_font_size


def main():
    if not os.path.exists(FONT):
        print("skip: font missing"); return
    r = TextRenderer(FONT)
    shape = np.zeros((300, 120), np.uint8)
    cv2.ellipse(shape, (60, 150), (55, 145), 0, 0, 360, 255, -1)
    orig = r._shape_layout
    text = "YOU CAN'T CONTROL YOUR CONQUEROR'S HAKI!!"
    r._shape_layout = lambda d, t, s, h, allow_hyphen=False: orig(d, t, s, h, False)
    whole = render(r, text, shape)
    r._shape_layout = orig
    broken = render(r, text, shape)
    assert broken >= 1.2 * whole, f"hyphenating should set it bigger: {whole} -> {broken}"
    print(f"narrow balloon: {whole}px unbroken -> {broken}px with one hyphen OK")
    # short words never split: size identical either way
    r._shape_layout = lambda d, t, s, h, allow_hyphen=False: orig(d, t, s, h, False)
    a = render(r, "WHY?! WHAT IS IT?!", shape)
    r._shape_layout = orig
    b = render(r, "WHY?! WHAT IS IT?!", shape)
    assert a == b, f"short words must not be broken ({a} vs {b})"
    print("short words untouched OK")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
