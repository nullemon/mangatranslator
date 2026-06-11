"""Site-watermark detection + erase/replace checks.

Scanlation sites stamp a URL/logo on the art (eshadow.net, …). The app should
recognise those (and NOT mistake dialogue for them), erase them, and optionally
drop the user's own watermark in their place.
Run: python3 tests/test_watermark.py"""
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.pipeline import _is_watermark
from core.compositor import Compositor


def main():
    # 1) Detection: domains/URLs yes; story text and Arabic no.
    yes = ["ESHADOW.NET", "eshadow.net", "www.mangasite.com", "https://x.io/a",
           "READ AT COOLSCANS.ORG"]
    no = ["THIS ONE HERE", "WAY TO GO, YUUTI!!", "NO, IT'S NOT FINE.",
          "I-IT'S FINE...", "أبي وأمي", ""]
    for t in yes:
        assert _is_watermark(t), f"should detect watermark: {t!r}"
    for t in no:
        assert not _is_watermark(t), f"should NOT flag: {t!r}"
    print("watermark detection OK")

    def page():
        img = np.full((300, 500, 3), 245, np.uint8)
        cv2.putText(img, "ESHADOW.NET", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        return img

    def ink(img):
        return int((img[8:60, 12:270] < 120).sum())

    wm = {"id": 1, "bbox": [12, 12, 255, 52], "type": "watermark", "erase": True,
          "translation": "", "in_bubble": False}

    # 2) Erase: the watermark pixels are wiped.
    src = page()
    before = ink(src)
    out = Compositor(use_lama=False).compose(src.copy(), [dict(wm)], {})
    assert before > 1000 and ink(out) < before * 0.4, f"not erased ({before} -> {ink(out)})"
    print(f"erase OK ({before} -> {ink(out)} ink px)")

    # 3) Replace: the user's watermark is placed where the site's was.
    items = [dict(wm)]
    out2 = Compositor(use_lama=False, replace_watermark=True,
                      watermark_text="@MYSCANS").compose(page(), items, {})
    assert items[0].get("placed"), "replace did not place the user's watermark"
    assert int((out2 < 120).sum()) > 500, "replacement watermark drew no ink"
    print("replace OK")

    print("ALL CHECKS PASSED ✓")


if __name__ == "__main__":
    main()
