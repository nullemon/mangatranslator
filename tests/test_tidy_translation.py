"""Model English is cleaned of Japanese typography a letterer always strips.

Run:  python tests/test_tidy_translation.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.prompts import tidy_translation as t, extract_json_array  # noqa: E402


def main():
    cases = [
        ("YOU CAN'T CONTROL 「CONQUEROR'S HAKI」!!", "YOU CAN'T CONTROL CONQUEROR'S HAKI!!"),
        ("〝THORN LAUNCHER〟!!!", "THORN LAUNCHER!!!"),
        ("AAAH〜〜〜!!!", "AAAH!!!"),
        ("NOPE———!!!", "NOPE—!!!"),
        ("WHAT!?", "WHAT?!"),
        ("WHY...!?", "WHY...?!"),
        ("THIS IS……", "THIS IS......"),
        ("WHAT！？", "WHAT?!"),
        ("THE PAIN  IS   PROOF", "THE PAIN IS PROOF"),
        ("LINE ONE \n LINE TWO", "LINE ONE\nLINE TWO"),
        ("I'M ALIVE — BARELY", "I'M ALIVE — BARELY"),        # a single dash stays
        ("", ""),
    ]
    for src, want in cases:
        got = t(src)
        assert got == want, f"{src!r}: got {got!r}, want {want!r}"
    arr = extract_json_array('[{"id": 1, "original": "「覇王色」", "translation": "「HAKI」!?"}]')
    assert arr[0]["translation"] == "HAKI?!", arr
    assert arr[0]["original"] == "「覇王色」", "the original must be left alone"
    print(f"{len(cases)} cases OK")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
