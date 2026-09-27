"""Cover-page art is not text: the series logo, a calligraphy stroke and a
bold black kanji are left as drawn.

Chapter 1194's cover (offline engine): the ONE PIECE logo was read as
今ＷＥＰＩＣは, taken for a black balloon, flat-filled black and lettered over;
a calligraphy brush stroke read as そして、 got "(LAUGHTER)" stamped on it;
the author's name had furigana-sized labels stamped on its black kanji.

Run:  python tests/test_cover_art_not_text.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.compositor import Compositor  # noqa: E402
from core.ocr import _has_japanese  # noqa: E402


def main():
    # 1. a read that is mostly Latin letters is not a Japanese line
    for text, want in [("今ＷＥＰＩＣは", False), ("ＯＮＥ ＰＩＥＣＥ", False),
                       ("Ｎｅｔｆｌｉｘで配信", True), ("ＯＫだ", True),
                       ("アニメ『ＬＥＧＯ ＯＮＥ ＰＩＥＣＥ』ついに明日配信開始！", True),
                       ("違う", True), ("……", False)]:
        assert _has_japanese(text) is want, (text, want)
    print("logo reads are not Japanese lines OK")

    comp = Compositor(use_lama=False)
    H, W = 400, 600

    # 2. a real black balloon: white letters on black, the letters are text
    page = np.full((H, W, 3), 245, np.uint8)
    cv2.ellipse(page, (300, 200), (200, 120), 0, 0, 360, (10, 10, 10), -1)
    seg = np.zeros((H, W), np.uint8)
    for c in range(4):
        for r in range(4):
            p = (220 + c * 45, 130 + r * 38)
            cv2.rectangle(page, p, (p[0] + 26, p[1] + 26), (250, 250, 250), -1)
            cv2.rectangle(seg, p, (p[0] + 26, p[1] + 26), 255, -1)
    mask = np.zeros((H, W), np.uint8)
    cv2.ellipse(mask, (300, 200), (196, 116), 0, 0, 360, 255, -1)
    comp._seg_mask = seg
    g = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    assert comp._is_real_balloon(g, mask, True), comp._balloon_why
    print("black balloon with white lettering is a balloon OK")

    # 3. a calligraphy stroke: black with paper showing through, no lettering
    stroke = np.full((H, W, 3), 245, np.uint8)
    cv2.rectangle(stroke, (60, 120), (540, 280), (10, 10, 10), -1)
    for k in range(6):                       # dry-brush streaks of paper
        cv2.line(stroke, (80 + 70 * k, 130), (120 + 70 * k, 270), (240, 240, 240), 5)
    smask = np.zeros((H, W), np.uint8)
    cv2.rectangle(smask, (60, 120), (540, 280), 255, -1)
    comp._seg_mask = np.zeros((H, W), np.uint8)
    gs = cv2.cvtColor(stroke, cv2.COLOR_BGR2GRAY)
    assert not comp._is_real_balloon(gs, smask, True), "a brush stroke passed as a black balloon"
    print(f"calligraphy stroke not a balloon ({comp._balloon_why}) OK")

    # 4. a bold black kanji: the shape is itself a text stroke
    glyph = np.full((H, W, 3), 245, np.uint8)
    cv2.rectangle(glyph, (250, 150), (350, 250), (10, 10, 10), -1)
    cv2.line(glyph, (260, 200), (340, 200), (240, 240, 240), 4)
    gmask = np.zeros((H, W), np.uint8)
    cv2.rectangle(gmask, (250, 150), (350, 250), 255, -1)
    comp._seg_mask = gmask.copy()
    gg = cv2.cvtColor(glyph, cv2.COLOR_BGR2GRAY)
    assert not comp._is_real_balloon(gg, gmask, True), "a black kanji passed as a balloon"
    print(f"black kanji not a balloon ({comp._balloon_why}) OK")

    # 5. composing: a reading on the brush stroke is left alone, art untouched
    comp2 = Compositor(use_lama=False)

    class NoText:
        ok = True

        def mask(self, image):
            return np.zeros(image.shape[:2], np.uint8)
        text_mask = raw_mask = mask
    comp2.text_seg = NoText()
    item = dict(id=1, bbox=[60, 120, 480, 160], in_bubble=True, type="dialogue",
                original="そして、", translation="And, uh,")
    drawn = []
    orig = comp2.renderer.draw_in_rect
    comp2.renderer.draw_in_rect = lambda image, rect, text, *a, **k: (
        drawn.append(text) or orig(image, rect, text, *a, **k))
    out = comp2.compose(stroke.copy(), [item], masks={1: smask})
    assert not drawn, f"English stamped on the brush stroke: {drawn}"
    diff = cv2.absdiff(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), gs)
    assert int((diff > 40).sum()) < 50, f"the brush stroke was altered ({int((diff > 40).sum())} px)"
    print("reading on the brush stroke left alone OK")

    # 6. a white balloon-shaped gap with no lettering in it, "read" as
    #    それでも、 -> "(Laughter)": left alone; the same shape with letters in
    #    it is lettered
    def balloon_page(with_letters):
        pg = np.full((H, W, 3), 60, np.uint8)
        cv2.ellipse(pg, (300, 200), (150, 110), 0, 0, 360, (250, 250, 250), -1)
        cv2.ellipse(pg, (300, 200), (150, 110), 0, 0, 360, (0, 0, 0), 3)
        ink = np.zeros((H, W), np.uint8)
        if with_letters:
            for c in range(3):
                for r in range(4):
                    p = (250 + c * 36, 130 + r * 36)
                    cv2.rectangle(pg, p, (p[0] + 24, p[1] + 24), (0, 0, 0), -1)
                    cv2.rectangle(ink, p, (p[0] + 24, p[1] + 24), 255, -1)
        bm = np.zeros((H, W), np.uint8)
        cv2.ellipse(bm, (300, 200), (146, 106), 0, 0, 360, 255, -1)
        return pg, ink, bm

    for with_letters in (False, True):
        pg, ink, bm = balloon_page(with_letters)
        comp3 = Compositor(use_lama=False)

        class Seg:
            ok = True

            def mask(self, image, _ink=ink):
                return _ink.copy()
            text_mask = raw_mask = mask
        comp3.text_seg = Seg()
        drawn3 = []
        orig3 = comp3.renderer.draw_in_rect
        comp3.renderer.draw_in_rect = lambda image, rect, text, *a, _o=orig3, **k: (
            drawn3.append(text) or _o(image, rect, text, *a, **k))
        it = dict(id=1, bbox=[160, 100, 280, 200], in_bubble=True, type="dialogue",
                  original="それでも、", translation="(Laughter)")
        comp3.compose(pg.copy(), [it], masks={1: bm})
        if with_letters:
            assert drawn3, "a balloon with lettering in it was skipped"
        else:
            assert not drawn3, f"English lettered on a reading with no text under it: {drawn3}"
    print("reading with no lettering under it left alone; real one lettered OK")

    # 6b. a white gap enclosed by a huge sound effect's strokes (which the
    #     text model marks as lettering), "read" as この: the strokes are
    #     around the gap, not in it — nothing is lettered there
    gap = np.full((H, W, 3), 250, np.uint8)
    sfx = np.zeros((H, W), np.uint8)
    cv2.ellipse(sfx, (300, 200), (150, 110), 0, 0, 360, 255, 40)   # heavy stroke ring
    gap[sfx > 0] = 0
    gm = np.zeros((H, W), np.uint8)
    cv2.ellipse(gm, (300, 200), (128, 88), 0, 0, 360, 255, -1)
    comp4 = Compositor(use_lama=False)

    class SfxSeg:
        ok = True

        def mask(self, image):
            return sfx.copy()
        text_mask = raw_mask = mask
    comp4.text_seg = SfxSeg()
    drawn4 = []
    orig4 = comp4.renderer.draw_in_rect
    comp4.renderer.draw_in_rect = lambda image, rect, text, *a, **k: (
        drawn4.append(text) or orig4(image, rect, text, *a, **k))
    comp4.compose(gap.copy(), [dict(id=1, bbox=[150, 90, 300, 220], in_bubble=True,
                                    type="dialogue", original="この", translation="This.")],
                  masks={1: gm})
    assert not drawn4, f"English lettered in a gap of a sound effect: {drawn4}"
    print("gap inside a sound effect left alone OK")

    # 7. solid black letters on a clean digital page (pure 0 / 255) measure
    pg, ink, _ = balloon_page(True)
    gp = Compositor._glyph_px(cv2.cvtColor(pg, cv2.COLOR_BGR2GRAY), ink, (160, 100, 280, 200))
    assert 20 <= gp <= 30, f"flat black lettering measured {gp}"
    print(f"flat digital lettering measured ({gp:.0f} px) OK")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
