"""A sound lettered in a balloon (ばっ! → FWOOSH!!, ガチ… → BITE...) is
translated like any balloon line; only loose effects on the art are left.

Two skips went too far. "No lettering under the reading" took an unmeasured
letter size as proof of an OCR phantom, but sound lettering is left out of
the dialogue strokes the size is read from, while the page's full stroke
mask holds it plainly. And "sound-effect-sized, not in a balloon" called a
line the detector put in a balloon "not in a balloon" whenever no balloon
shape came with it.

Run:  python tests/test_sound_balloon_kept.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.compositor import Compositor  # noqa: E402


def main():
    comp = Compositor(use_lama=False)

    # ── phantom check: judged by the strokes actually in the box ──
    comp._seg_mask = np.zeros((400, 400), np.uint8)
    it = {"id": 1, "bbox": [100, 100, 120, 80], "original": "ガチ…",
          "in_bubble": True, "_glyph_px": 0.0}
    assert comp._phantom_read(it), "an empty box should read as a phantom"
    comp._seg_mask[110:170, 110:200:6] = 255          # lettering strokes
    assert not comp._phantom_read(it), "lettering in the box was called a phantom"
    it["_glyph_px"] = 30.0
    assert not comp._phantom_read(it)
    print("phantom only when the box has no lettering OK")

    # ── a big sound in a balloon with no shape to check: lettered ──
    page = np.full((400, 600, 3), 245, np.uint8)
    page[140:180, 90:210] = 20                        # the Japanese
    comp._sfx_sized = lambda it: True                 # judged sound-sized
    comp._phantom_read = lambda it, bm=None: False
    items = [
        {"id": 1, "bbox": [80, 130, 140, 60], "in_bubble": True, "type": "dialogue",
         "original": "ばっ!", "translation": "FWOOSH!!"},
        {"id": 2, "bbox": [380, 130, 140, 60], "in_bubble": False, "type": "dialogue",
         "original": "ゴゴゴ", "translation": "RUMBLE"},
    ]
    comp.compose(page.copy(), items)
    assert items[0].get("placed"), "a sound in a balloon was left untranslated"
    assert not items[1].get("placed"), "a loose effect on the art was lettered over"
    print("sound in a balloon lettered, loose effect left OK")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
