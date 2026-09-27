"""The credit line and the watermark never land on a translation.

Reported from a real page: the credit ("TRANSLATIONS BY WONPE4CE") was
stamped straight over the English line lettered in the bottom-right corner.

Two ways that happened:

  1. The credit is a compositor item, dropped at a random spot along a page
     edge (or wherever the user dragged it) and drawn with no look at what
     else the page lettered — so it went down on top of whatever was there.
  2. The corner watermark kept clear only of lettering it could RE-DETECT
     on the finished page. English the detector does not see (light
     lettering on a dark slab, big display lettering, a line over art) was
     not in the keep-out, and the mark went on top of it.

Both now keep clear of where the page's lines were actually DRAWN. Checked
pixel by pixel: the credit's / watermark's own pixels (finished page minus
the page without it) never touch a translation's pixels (the page with the
lettering minus the page before it).

Run:  python tests/test_watermark_no_overlap.py
"""
import inspect
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)          # the app's folders are relative

from core.compositor import Compositor  # noqa: E402

LINE = "HE WAS DEFINITELY USING IT EARLIER...!! COULD IT BE..."
CREDIT = "TRANSLATIONS BY WONPE4CE"
W, H = 900, 1300
GAP = 2                 # px the two sets of pixels must stay apart


def changed(a, b, thr=40):
    return cv2.absdiff(a, b).max(axis=2) > thr


def touching(a, b, gap=GAP):
    k = np.ones((2 * gap + 1, 2 * gap + 1), np.uint8)
    return int((cv2.dilate(a.astype(np.uint8), k) > 0)[b].sum())


def blank_page():
    page = np.full((H, W, 3), 246, np.uint8)
    # a little art so it is not a blank sheet
    cv2.rectangle(page, (20, 20), (W - 20, H // 2 - 10), (0, 0, 0), 3)
    cv2.circle(page, (300, 300), 120, (40, 40, 40), 3)
    return page


def line_item(i, box, text=LINE):
    # A drawn box (fixed layout, no detection), the way a line over art or a
    # box the user placed is lettered.
    return dict(id=i, bbox=list(box), original="", translation=text,
                type="manual", in_bubble=False, manual=True, rotation=0)


def credit_item(i, box):
    return dict(id=i, bbox=list(box), original="", translation=CREDIT,
                type="credit", in_bubble=False, credit=True, rotation=0)


def check_credit(comp, name, lines, cbox, offsets=None, expect_drawn=True):
    page = blank_page()
    plain = comp.compose(page.copy(), [dict(it) for it in lines])
    items = [dict(it) for it in lines] + [credit_item(99, cbox)]
    out = comp.compose(page.copy(), items, offsets=offsets)
    text_px = changed(plain, page)
    credit_px = changed(out, plain)
    assert text_px.sum() > 500, f"{name}: the translation drew nothing"
    hit = touching(credit_px, text_px)
    assert hit == 0, (f"{name}: the credit was drawn over the translation "
                      f"({hit} px touching)")
    if expect_drawn:
        assert credit_px.sum() > 150, f"{name}: the credit was not drawn at all"
        ys, xs = np.nonzero(credit_px)
        print(f"{name}: credit at x {xs.min()}-{xs.max()}, y {ys.min()}-"
              f"{ys.max()}, clear of the translation OK")
    else:
        assert credit_px.sum() == 0, f"{name}: credit drawn with no room for it"
        assert not items[-1].get("placed"), f"{name}: skipped credit marked placed"
        print(f"{name}: no free spot anywhere, credit skipped OK")
    return items


def part_credit_item():
    comp = Compositor(use_lama=False)
    corner = line_item(1, (500, 1150, 360, 120))
    # 1. The reported case: credit dropped right on the bottom-right line.
    items = check_credit(comp, "credit on the line", [corner],
                         (520, 1170, 300, 43))
    # The stored box follows the credit, so the editor shows it where it is.
    moved = items[-1]
    assert moved.get("placed") and moved.get("drawn"), "credit footprint missing"
    # ...and it stays near where it was put, not flung across the page.
    ys, xs = moved["bbox"][1], moved["bbox"][0]
    assert ys > H // 2, f"credit moved to the top half (y={ys})"
    # 2. Half on the line.
    check_credit(comp, "credit half on the line", [corner],
                 (380, 1200, 300, 43))
    # 3. Free where it was put, then dragged onto the line (editor Move).
    check_credit(comp, "credit dragged onto the line", [corner],
                 (60, 1230, 300, 43), offsets={"99": [460, -60]})
    # 4. Every lettered line on the page counts, not just the first.
    lines = [line_item(1, (40, 1150, 360, 120), "WHAT ARE YOU DOING HERE?!"),
             corner, line_item(3, (60, 700, 780, 90), "A LONG LINE ACROSS")]
    check_credit(comp, "credit among several lines", lines,
                 (60, 1180, 300, 43))
    # 5. Nowhere free: the page is lettered edge to edge — skipped, never
    # drawn over a line.
    full = [line_item(10 + k, (0, k * 130, W, 130), "WORDS " * 12)
            for k in range(H // 130)]
    check_credit(comp, "credit on a full page", full, (500, 1200, 300, 43),
                 expect_drawn=False)


def part_corner_stamp():
    """The watermark and credit stamped by the app after the page is done."""
    import app
    comp = Compositor(use_lama=False)
    comp.text_seg = None        # the lettering is all this test's own
    page = blank_page()
    # Dark slabs in both bottom corners with light lettering on them, lettered
    # right down into the corners where the marks go.
    cv2.rectangle(page, (440, 1060), (W - 1, H - 1), (18, 18, 18), -1)
    cv2.rectangle(page, (0, 1060), (430, H - 1), (18, 18, 18), -1)
    lines = [line_item(1, (470, 1130, 425, 168)),
             line_item(2, (5, 1130, 420, 168), "WAIT... WHERE DID IT GO?!")]
    out = comp.compose(page.copy(), lines)
    text_px = changed(out, page)
    assert text_px.sum() > 500, "the translations drew nothing"
    takes = "items" in inspect.signature(app._stamp_output).parameters
    if takes:
        assert all(it.get("drawn") for it in lines), "no drawn footprint"
    real_reader = app._page_lettering
    # How the finished page's lettering is read:
    #  - "missed": the text model read the page and found nothing where the
    #    English is (it is trained on Japanese pages; English lettered over
    #    art, or light on dark, is what its block head misses);
    #  - "heuristic": no model installed, the shape-of-the-ink fallback.
    readers = {
        "missed": lambda img: {"blocks": [],
                               "strokes": np.zeros(img.shape[:2], np.uint8)},
        "heuristic": lambda img: None,
    }
    tmpdir = tempfile.mkdtemp()
    try:
        for rname, reader in readers.items():
            app._page_lettering = reader
            for style in ("clean", "bold", "pill", "ribbon", "clean+tile"):
                for place in ("br", "bl", "tr", "random"):
                    path = os.path.join(tmpdir, f"{rname}_{style}_{place}.png")
                    cv2.imwrite(path, out)
                    kw = {"items": lines} if takes else {}
                    app._stamp_output(path, "@MyScanGroup", place, 100, "m",
                                      CREDIT, style, **kw)
                    stamped = cv2.imread(path)
                    mark_px = changed(stamped, out, 30)
                    assert mark_px.sum() > 100, (
                        f"{rname}/{style}/{place}: nothing stamped")
                    hit = touching(mark_px, text_px)
                    assert hit == 0, (
                        f"{rname}/{style}/{place}: watermark/credit drawn "
                        f"over a translation ({hit} px touching)")
            print(f"lettering {rname}: watermark + credit clear of every "
                  "translation, all styles and corners OK")
        # Nowhere clear at all: the marks are left off, never put on a line.
        app._page_lettering = readers["missed"]
        full = [line_item(10 + k, (0, k * 130, W, 130), "WORDS " * 12)
                for k in range(H // 130)]
        page2 = comp.compose(blank_page(), full)
        path = os.path.join(tmpdir, "full.png")
        cv2.imwrite(path, page2)
        kw = {"items": full} if takes else {}
        app._stamp_output(path, "@MyScanGroup", "br", 100, "m", CREDIT,
                          "clean", **kw)
        left = int(changed(cv2.imread(path), page2, 30).sum())
        assert left == 0, f"marks drawn on a fully lettered page ({left} px)"
        print("fully lettered page: marks left off OK")
    finally:
        app._page_lettering = real_reader
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    part_credit_item()
    part_corner_stamp()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
