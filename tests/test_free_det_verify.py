"""Free-text detection verification: a genuine line must not be thrown away
just because local OCR on a vertical / furigana / outlined crop is noisy,
while a box that really holds OTHER text must still be rejected.

Regression case (One Piece ch.1194 p.73): the editor's teaser at the top
right, vertical white-on-black-outline lettering with furigana, was found by
the vision model as '☆ゾロVSソマーズ!!' but local OCR of the box read
'．．．ッ！！オンマース！お父' and the det was rejected with
"box reads different text", leaving the teaser untranslated.

The OCR engine and the text-stroke model are stubbed, so this runs without
any downloaded model.
Run: python tests/test_free_det_verify.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.pipeline import TranslationPipeline

TEASER = "☆ゾロVSソマーズ!!\n不死の体、攻略なるか!?"
TEASER_OCR = "．．．ッ！！オンマース！お父"

PAGE_W, PAGE_H = 1403, 2048
BOX = [1199, 229, 154, 204]


class StubOCR:
    """read_region returns `full` for the whole box and `column` for any
    narrower crop (a single column of a vertical block)."""
    ok = True

    def __init__(self, full, column=None):
        self.full, self.column = full, column
        self.calls = []

    def read_region(self, image, bbox, mask=None):
        self.calls.append(list(bbox))
        if self.column is not None and bbox[2] < 0.75 * BOX[2]:
            return self.column
        return self.full


class StubSeg:
    ok = True

    def __init__(self, mask):
        self._m = mask

    def mask(self, image):
        return self._m


def stroke_mask(density):
    """Text-pixel mask for BOX: 'dense' (>=3% coverage), 'medium' (between
    the 0.8% OCR-confirmation floor and the 3% dense bar) or None. The
    lettering is drawn as two main vertical columns plus a thin furigana
    column, the way the teaser is laid out."""
    if density is None:
        return None
    m = np.zeros((PAGE_H, PAGE_W), np.uint8)
    x, y, w, h = BOX
    # (x offset, width) of each column inside the box; right-to-left reading
    cols = [(110, 30), (60, 30), (96, 6)]          # two main + one furigana
    fill = 1.0 if density == "dense" else 0.06     # ~38% vs ~2.3% of the box
    rng = np.random.default_rng(0)
    for cx, cw in cols:
        sub = m[y + 10:y + h - 10, x + cx:x + cx + cw]
        sub[rng.random(sub.shape) < fill] = 255
    return m


def make_pipe(ocr, density):
    p = TranslationPipeline.__new__(TranslationPipeline)
    p.source_lang = "Japanese"
    p.ocr = ocr
    mask = stroke_mask(density)
    p.text_seg = StubSeg(mask) if mask is not None else None
    return p


def main():
    img = np.full((PAGE_H, PAGE_W, 3), 128, np.uint8)

    # 1) The real case: must be ACCEPTED whatever the stroke evidence.
    for dens in ("dense", "medium", None):
        p = make_pipe(StubOCR(TEASER_OCR), dens)
        ok, why = p._verify_text_region(img, BOX, TEASER)
        print(f"teaser, strokes={dens}: ok={ok} ({why})")
        assert ok, f"genuine teaser rejected with strokes={dens}: {why}"
        # the first line alone (as it appears truncated in the log) too
        ok, why = p._verify_text_region(img, BOX, "☆ゾロVSソマーズ!!\n")
        assert ok, f"teaser first line rejected with strokes={dens}: {why}"

    # 2) A box that reads clearly OTHER text is still rejected.
    for dens in ("dense", "medium", None):
        p = make_pipe(StubOCR("ドン"), dens)
        ok, why = p._verify_text_region(img, BOX, "海賊王に俺はなる")
        print(f"wrong text, strokes={dens}: ok={ok} ({why})")
        assert not ok, f"mismatched det accepted with strokes={dens}: {why}"
        assert "different text" in why, why
    p = make_pipe(StubOCR("お前は誰だ"), "dense")
    ok, why = p._verify_text_region(img, BOX, "海賊王に俺はなる")
    assert not ok, f"unrelated dialogue accepted on dense lettering: {why}"

    # 3) Whole-box read is garbage, but reading the box column by column
    #    (furigana column dropped) recovers the claimed text -> accepted.
    ocr = StubOCR("．．．父！！", column="不死の体、攻略なるか")
    p = make_pipe(ocr, "dense")
    ok, why = p._verify_text_region(img, BOX, TEASER)
    print(f"column re-read: ok={ok} ({why}); {len(ocr.calls)} OCR calls")
    assert ok, f"column re-read did not rescue the line: {why}"
    assert len(ocr.calls) > 1, "column re-read never ran"
    assert len(ocr.calls) <= 8, f"too many OCR calls: {len(ocr.calls)}"
    # ... but column re-reads can't rescue a genuinely wrong claim.
    ocr = StubOCR("ドン", column="ドドド")
    p = make_pipe(ocr, "dense")
    ok, why = p._verify_text_region(img, BOX, "海賊王に俺はなる")
    assert not ok, f"column re-read rescued a wrong claim: {why}"

    # 4) Artwork protection is unchanged: no strokes -> rejected even when
    #    OCR "reads" the claimed text into texture.
    p = make_pipe(StubOCR(TEASER), None)
    p.text_seg = StubSeg(np.zeros((PAGE_H, PAGE_W), np.uint8))
    ok, why = p._verify_text_region(img, BOX, TEASER)
    assert not ok and why == "no text strokes", why

    # 5) The tolerant comparison itself.
    from core.pipeline import _read_supports_claim
    assert _read_supports_claim(TEASER, TEASER_OCR)
    assert _read_supports_claim("ソマーズ", "オンマース")
    assert _read_supports_claim("だいじょうぶか", "だいじょうふか")   # dakuten lost
    assert _read_supports_claim("やっぱり", "やつぱり")               # small kana
    assert not _read_supports_claim("海賊王に俺はなる", "ドン")
    assert not _read_supports_claim("海賊王に俺はなる", "ドン", dense=True)
    assert not _read_supports_claim("ありがとう", "ーーー!!", dense=True)
    print("similarity checks OK")

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
