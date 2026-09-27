"""Lettering on the art next to a big balloon is still found and translated.

The text-block pass dropped any block that touched a balloon's box or
another pass's box at all. A spiky burst's box reaches well past its spikes,
and the free-text finder draws loose boxes: on chapter 1194 p73 both
inner-monologue blocks (あ…危なかった… and だが!!「覇王色」は…) were thrown
away without a word and left in Japanese. A block is a balloon's own text
only when it lies mostly inside the balloon's box, and a line another pass
found is matched by its text in the same place. And the two blocks, three
glyphs apart, stay two lines instead of being merged into one run — and
are not taken for one line because they share common kana.

Run:  python tests/test_free_block_beside_balloon.py
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.pipeline import TranslationPipeline  # noqa: E402

MONO_A = (516, 547, 338, 237)     # あ…危なかった… beside the burst
MONO_B = (137, 552, 257, 447)     # だが!!… under a loose box
IN_BALLOON = (1000, 600, 200, 100)
DUP = (300, 1300, 150, 150)

READS = {MONO_A: "あ…危なかった…!! 心臓は完全に狙われていた…!!",
         MONO_B: "だが!! 覇王色は纏っていなかった…!! なぜだ…!?",
         IN_BALLOON: "ブハァ!! ハァ ハァ",
         DUP: "宝の持ち腐れだ!!"}


class Seg:
    ok = True

    def detect_blocks(self, image):
        return [MONO_A, MONO_B, IN_BALLOON, DUP]

    def mask(self, image):
        # 40 px lettering in every block: the two monologues, 122 px apart,
        # are two paragraphs, not one run of giant glyphs
        m = np.zeros(image.shape[:2], np.uint8)
        for x, y, w, h in (MONO_A, MONO_B, IN_BALLOON, DUP):
            for cx in range(x + 10, x + w - 40, 50):
                for cy in range(y + 10, y + h - 40, 50):
                    m[cy:cy + 40, cx:cx + 30] = 255
        return m


class OCR:
    ok = True

    def read_region(self, image, box, _):
        return READS[tuple(box)]


def main():
    p = TranslationPipeline.__new__(TranslationPipeline)
    p.text_seg, p.ocr = Seg(), OCR()
    p.translate_sfx, p.source_lang = False, "Japanese"
    sent = {}

    def translate_map(image, id_to_text, box_map):
        sent.update(id_to_text)
        return {k: {"translation": "EN " + str(k)} for k in id_to_text}
    p._translate_map = translate_map

    # the burst's box grazes block A; a loose box from the free-text finder
    # (a different line) covers block B; DUP was already found by that pass
    bubbles = [SimpleNamespace(id=1, bbox=[800, 520, 560, 480])]
    existing = [
        {"id": 2, "bbox": [100, 500, 420, 560], "original": "☆ゾロVSソマーズ!!"},
        {"id": 3, "bbox": [290, 1290, 170, 170], "original": "宝の持ち腐れだ!!"},
    ]
    items = p._free_text_seg(np.zeros((2048, 1403, 3), np.uint8), bubbles,
                             existing, lambda *a, **k: None)
    got = {tuple(it["bbox"]) for it in items}
    assert MONO_A in got, "a block beside a big balloon was dropped"
    assert MONO_B in got, "a block under another pass's loose box was dropped"
    assert IN_BALLOON not in got, "a balloon's own text was added again"
    assert DUP not in got, "a line another pass already found was added twice"
    # the same line read slightly differently by the two passes is still one
    existing[1]["original"] = "宝の持ち腐れだ!!!"
    items = p._free_text_seg(np.zeros((2048, 1403, 3), np.uint8), bubbles,
                             existing, lambda *a, **k: None)
    assert DUP not in {tuple(it["bbox"]) for it in items}, "a re-read duplicate was added"
    print("monologue beside the burst and under a loose box kept OK")
    print("balloon text and duplicates still skipped OK")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
