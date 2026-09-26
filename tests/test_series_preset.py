"""One Piece preset: picked up from the Manga title, carries TCB's cover
conventions, names and sound renderings, and each page gets only the words
it uses. Also: a chapter title on its strip is lettered on ONE line.

Run:  python tests/test_series_preset.py
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core import series  # noqa: E402
from core.glossary import focus_style  # noqa: E402
from core.translator import make_translator  # noqa: E402
from core.renderer import TextRenderer  # noqa: E402

FONT = os.path.join(os.path.dirname(__file__), "..", "fonts", "Bangers-Regular.ttf")


def main():
    for title, want in [("One PIECE", "one piece"), ("one piece", "one piece"),
                        ("ワンピース", "one piece"), ("Blue Lock", None)]:
        got = series.detect(f'SERIES: this page is from "{title}". Use that manga')
        assert got == want, (title, got)
    print("series detected from the Manga title OK")

    full = series.apply('SERIES: this page is from "One Piece".\nMY OWN RULE')
    assert full.index("SERIES PRESET") < full.index("MY OWN RULE"), "user lines must come after"
    assert series.apply(full) == full, "applying twice must not duplicate"
    for must in ("ODA EIICHIRO", "CHAPTER <number>: <TITLE>", "type \"promo\"",
                 "覇王色 = Conqueror's Haki", "ハァ ハァ = HUFF... HUFF...", "イバランチャー = Thorn Launcher"):
        assert must in full, must
    assert "Summers' laugh" in full, "phrasebook notes should reach the model"
    assert "#" not in "\n".join(l for l in full.splitlines() if " = " in l and "(" not in l), \
        "glossary lines must be sent without their # notes"
    print("preset carries cover rules, glossary and phrasebook OK")

    page = "ブハァ!! ……ハァ ハァ…!! ゲホ!! 「覇王色」"
    f = focus_style(full, page)
    for must in ("ハァ ハァ = HUFF... HUFF...", "ゲホ = KOFF!!", "覇王色 = Conqueror's Haki", "COVER / TITLE PAGE"):
        assert must in f, must
    assert "ブルック = Brook" not in f, "a name not on the page is not sent"
    print("per-page focus keeps the page's words and the rules OK")

    tr = make_translator("claude", "x", "claude-sonnet-4-6",
                         'SERIES: this page is from "One Piece".')
    assert "SERIES PRESET" in tr.style
    tr2 = make_translator("claude", "x", "claude-sonnet-4-6", 'SERIES: this page is from "Blue Lock".')
    assert "SERIES PRESET" not in tr2.style
    print("every engine gets the preset only for One Piece OK")

    if os.path.exists(FONT):
        r = TextRenderer(FONT)
        img = Image.new("RGB", (1300, 200), "white")
        drawn = []
        orig = r._wrap
        r._wrap = lambda text, font, max_w, draw: drawn.append(orig(text, font, max_w, draw)) or drawn[-1]
        r._single_line = True
        r.draw_in_rect(img, (40, 20, 1200, 150), "CHAPTER 1194: THE IMPERMANENCE OF ALL THINGS", (0, 0, 0))
        assert all(len(lines) == 1 for lines in drawn), drawn[-1]
        print("chapter title on its strip: one line OK")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
