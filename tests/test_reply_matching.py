"""Each bubble gets ITS OWN translation, whatever the reply's numbering.

A real page: 4 bubbles were read and sent, and the reply — the model had
the page image — came back with 11 entries: it also translated the
narration it could see on the art and numbered everything its own way.
Filed by number, bubble 1 got the page's first narration line and so on:
every bubble showed another line's English, and the tidy pass then threw the
real narration away as a "duplicate" of those misfiled lines.

Run:  python tests/test_reply_matching.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.pipeline import TranslationPipeline as Pipeline, tidy_free_text  # noqa: E402


def main():
    sent = {1: "どっちだ!?", 2: "これは……", 3: "「覇王色」アリかナシか!!", 4: "おうぇ〜〜〜!!!"}
    page_lines = [
        ("あの瞬間だけ「覇王色」を!?いずれにせよこいつの強さにはムラがある!!",
         "WAS HE USING CONQUEROR'S HAKI FOR JUST THAT INSTANT?!"),
        ("奴の力は何らかの熱を帯びている!?──もしくは", "IS HIS POWER GENERATING SOME KIND OF HEAT?! OR PERHAPS..."),
        ("ここはさっさと…", "I NEED TO FINISH THIS QUICKLY..."),
        ("どっちだ!?", "WHICH IS IT?!"),
        ("これは……", "THIS IS..."),
        ("「覇王色」アリかナシか!!", "CONQUEROR'S HAKI OR NOT?!"),
        ("おうぇ〜〜〜!!!", "OUEEHH!!!"),
    ]
    reply = {i + 1: {"id": i + 1, "original": jp, "translation": en}
             for i, (jp, en) in enumerate(page_lines)}
    fixed = Pipeline._match_replies(sent, reply)
    want = {1: "WHICH IS IT?!", 2: "THIS IS...", 3: "CONQUEROR'S HAKI OR NOT?!", 4: "OUEEHH!!!"}
    got = {k: v["translation"] for k, v in fixed.items()}
    assert got == want, f"bubbles got the wrong lines: {got}"
    print("11 replies for 4 bubbles: each bubble got its own line OK")

    # a normal reply (one entry per id, OCR slightly off) is left as it is
    normal = {1: {"id": 1, "original": "どっちだ!?", "translation": "WHICH IS IT?!"},
              2: {"id": 2, "original": "これは…", "translation": "THIS IS..."},
              3: {"id": 3, "original": "覇王色アリかナシか!!", "translation": "HAKI OR NOT?!"},
              4: {"id": 4, "original": "", "translation": "OUEEHH!!!"}}
    assert {k: v["translation"] for k, v in Pipeline._match_replies(sent, normal).items()} == \
        {1: "WHICH IS IT?!", 2: "THIS IS...", 3: "HAKI OR NOT?!", 4: "OUEEHH!!!"}
    print("a normal reply is kept as it is OK")

    # the tidy pass drops a free-text line only when it repeats a bubble AT THE SAME SPOT
    items = [
        {"id": 1, "bbox": [900, 1450, 120, 130], "in_bubble": True, "original": "これは……",
         "translation": "THIS IS..."},
        {"id": 2, "bbox": [1150, 60, 150, 300], "in_bubble": False, "type": "narration",
         "original": "奴の力は何らかの熱を帯びている!?", "translation": "IS HIS POWER..."},
        {"id": 3, "bbox": [100, 1470, 60, 60], "in_bubble": True, "original": "奴の力は何らかの熱を帯びている!?",
         "translation": "(misfiled)"},
        {"id": 4, "bbox": [905, 1455, 110, 120], "in_bubble": False, "type": "narration",
         "original": "これは……", "translation": "THIS IS..."},
    ]
    kept = [it["id"] for it in tidy_free_text(items)]
    assert 2 in kept, "the narration was dropped for matching text in a bubble elsewhere"
    assert 4 not in kept, "a real duplicate (same text, same place) was kept"
    print("tidy drops only same-place duplicates OK")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
