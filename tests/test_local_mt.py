"""Offline engine: a model that loads but can't translate is skipped, a
looping translation is cut back, and sounds / rows of dots never reach the
model — they come from the series phrasebook or keep their punctuation.

Chapter 1194 offline: fugumt-ja-en turned 私は学生です。 into "My school is
in the school of Irreceive…", うわ!! into "jesus christ, jesus jesus, walter
walrust…" and ーー… into "- Oh, my God.", and lines hundreds of characters
long were lettered in microscopic type.

No model is loaded: the translation model is a stub.

Run:  python tests/test_local_mt.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core import local_mt, series  # noqa: E402
from core.translator import LocalTranslator  # noqa: E402

ONE_PIECE = 'SERIES: this page is from "One Piece".'


class FakeModel(local_mt.LocalMT):
    """LocalMT with the model swapped for a dictionary."""

    def __init__(self, model_id, table, sanity=None):
        self.table = table
        self.asked = []
        super().__init__(model_id, sanity)

    def _load(self):
        self.ok = True

    def _generate(self, batch, cap):
        self.asked += list(batch)
        return [self.table.get(t, "Hello.") for t in batch]


def main():
    # 1. the self-test refuses a model that can't translate a textbook line
    broken = FakeModel("broken", {
        "猫はかわいいです。": "My cat is in the cat of Irreceive",
        "私は学生です。": "My school is in the school of Irreceive"},
        local_mt.SANITY_CHECKS["japanese"])
    assert not broken.ok, "a model failing its self-test must not be used"
    assert "self-test" in local_mt.LocalMT.last_error
    good = FakeModel("good", {"猫はかわいいです。": "Cats are cute.",
                              "私は学生です。": "I'm a student."},
                     local_mt.SANITY_CHECKS["japanese"])
    assert good.ok
    print("self-test: broken model refused, working one kept OK")

    # get() falls through to the next candidate when the first fails
    saved = (local_mt.LocalMT, dict(local_mt._CACHE))
    try:
        made = []

        class ByName(FakeModel):
            def __init__(self, mid, sanity=None):
                made.append(mid)
                super().__init__(mid, {"猫はかわいいです。": "Cats are cute.",
                                       "私は学生です。": "I'm a student."}
                                 if "opus" in mid else {}, sanity)
        local_mt._CACHE.clear()
        local_mt.LocalMT = ByName
        got = local_mt.get("Japanese")
        assert got is not None and "opus" in got.model_id, (made, got)
    finally:
        local_mt.LocalMT, cache = saved
        local_mt._CACHE.clear()
        local_mt._CACHE.update(cache)
    print("get(): falls through to the model that passes OK")

    # 2. a runaway loop is cut back to what the line can carry
    src = "妙なまぐれ当たりをかまされる前に消さねェと．．．！"
    loop = ("I'm not gonna get it out of the way... the odd snazzlers are "
            "snatched up, jesus christ, babe. " + "jesus, walter, babe, " * 20)
    cut = local_mt._trim_runaway(loop, src)
    assert len(cut) <= local_mt._budget(src), len(cut)
    assert cut.startswith("I'm not gonna get it out of the way..."), cut
    fine = "If it breaks any more...!!"
    assert local_mt._trim_runaway(fine, "これ以上壊れたら．．．！！") == fine
    print("runaway translation trimmed, normal one untouched OK")

    # 3. punctuation-only lines keep their punctuation
    for jp, en in [("……", "..."), ("．．．！！", "...!!"), ("ーー．．．", "..."),
                   ("！！！", "!!!"), ("…！？", "...?!"), ("ーーー", "...")]:
        assert local_mt.punctuation_only(jp) == en, (jp, local_mt.punctuation_only(jp))
    assert local_mt.punctuation_only("うわ！！") == ""
    print("punctuation-only lines OK")

    # 4. sounds from the phrasebook; exact entry before the loose one
    pairs = series.phrasebook(series.apply(ONE_PIECE))
    assert pairs, "the One Piece preset should have a phrasebook"
    for jp, en in [("ハァ．．．ハァ．．．", "HUFF... HUFF..."), ("うわ！！", "UWAH!!"),
                   ("ばっ！！", "BAM!"), ("ばっ!", "FWOOSH!!"),
                   ("〝三刀流〟．．．；！！", "Three Sword Style...!!"),
                   ("ナシィ——!!!", "NOPE!!")]:
        assert local_mt.fixed_rendering(jp, pairs) == en, (jp, local_mt.fixed_rendering(jp, pairs))
    assert local_mt.fixed_rendering("しまった", pairs) == ""
    assert series.phrasebook("no series named here") == []
    print("phrasebook renderings OK")

    # 5. the offline translator only sends real lines to the model
    fake = FakeModel("good", {"この痛みは生きてる証レィ！！！": "This pain is proof that I'm alive!"})
    real_get = local_mt.get
    local_mt.get = lambda lang="Japanese": fake
    try:
        tr = LocalTranslator(style=series.apply(ONE_PIECE))
        out = tr.translate_texts({1: "うわ！！", 2: "ーー．．．",
                                  3: "この痛みは生きてる証レィ！！！", 4: "……"})
    finally:
        local_mt.get = real_get
    assert fake.asked == ["この痛みは生きてる証レィ！！！"], fake.asked
    assert [out[i]["translation"] for i in (1, 2, 3, 4)] == [
        "UWAH!!", "...", "This pain is proof that I'm alive!", "..."], out
    print("offline translator: sounds and dots kept from the model OK")

    # 6. a subtitle stage direction is not a line: the colon manga-ocr writes
    #    for a vertical … is turned back, and a "(Laughter)" answer is
    #    translated again in pieces
    fake2 = FakeModel("good", {
        "いいかロロノア…万物は変わりゆく": "(Laughter)",
        "いいかロロノア…": "Listen, Roronoa...",
        "万物は変わりゆく": "All things change.",
        "それでも、": "(Laughter) (Applause)"})
    out2 = fake2.translate_many(["いいかロロノア：万物は変わりゆく", "それでも、"])
    assert "いいかロロノア…万物は変わりゆく" in fake2.asked, fake2.asked
    assert out2 == ["Listen, Roronoa... All things change.", ""], out2
    print("stage directions re-translated in pieces or dropped OK")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
