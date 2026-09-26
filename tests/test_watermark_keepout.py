"""Watermark / credit keep-out on real manga pages.

The corner stamp and the credit line are placed clear of the page's
lettering. The keep-out used to be guessed from the shape of the ink, and on a
busy page hatching, halftone and whole bright panels passed for lettering: on
a One Piece chapter it marked 54-83% of every page, so no corner was ever free.
It now comes from the manga-trained text model (core/text_seg.py).

Checks, on real pages (the raw, and the same page as released in English,
aligned to the raw):
  1. the keep-out stays well below what the shape heuristic marks;
  2. every spot the English release lettered is inside it — the Japanese on
     the raw, the English on the release;
  3. on a page whose bottom-right corner is clear art, a "Bottom right"
     stamp lands in that corner and the credit line in the bottom-left one,
     neither on lettering nor on each other — through the real _stamp_all();
  4. without the model the shape heuristic still produces a keep-out.

Needs the pages (MT_KEEPOUT_PAGES, default /tmp/claude-0/ch: raw/p-00NN.jpg,
al/NN.png, al/NN_edit.png, al/tcb_regions.json) and the text model
(models/comictextdetector.pt.onnx). Skips cleanly without either. Reads four
pages with the model, ~5-10 s each on an idle CPU.

Run: python tests/test_watermark_keepout.py
"""
import json
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)          # the model path and the app's folders are relative

DATA = os.environ.get("MT_KEEPOUT_PAGES", "/tmp/claude-0/ch")

# Page 75: the busiest (13 lettered spots; the heuristic marked 77% of it).
# Page 78: the heuristic marked 83%; its bottom-right corner is a sea
# creature with no lettering on it.
PAGES = (75, 78)
CLEAR_BR = 78
# Where the ground truth is clean enough to demand every pixel. On the
# English 78 the release also redrew hatching where the Japanese used to be,
# and that retouched art is indistinguishable from lettering by pixel diff.
RECALL = {("raw", 75), ("en", 75), ("raw", 78)}
MAX_COVER = 45.0          # % of the page (measured 31-41; heuristic 74-83)
MAX_OF_HEURISTIC = 0.65   # ...and well under what the heuristic marks
MIN_RECALL = 0.97         # of each lettered spot's pixels inside the keep-out
PAD_M = 16                # the "m" watermark's padding on a 1403 px page


def skip(why):
    print(f"SKIPPED: {why}")
    sys.exit(0)


def page_paths(n):
    return {"raw": os.path.join(DATA, "raw", f"p-{n:04d}.jpg"),
            "en": os.path.join(DATA, "al", f"{n}.png")}


def lettered(n, img, other, regions):
    """This page's lettering in each spot the English release lettered:
    pixels its editors changed that are ink here and paper on the other
    version — the Japanese strokes they removed (on the raw), the English
    strokes they added (on the release)."""
    edit = cv2.imread(os.path.join(DATA, "al", f"{n}_edit.png"), 0) > 0
    ink = (edit & (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 128)
           & (cv2.cvtColor(other, cv2.COLOR_BGR2GRAY) >= 128))
    out = []
    for x, y, w, h, _ in regions:
        if w < 45 or h < 45:
            continue        # slivers of re-levelled art at a page edge
        m = np.zeros_like(ink)
        m[y:y + h, x:x + w] = ink[y:y + h, x:x + w]
        if m.sum() >= 50:
            out.append(((x, y, w, h), m))
    return ink, out


def changed_box(before, after):
    d = cv2.absdiff(after, before).max(axis=2) > 25
    ys, xs = np.nonzero(d)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def main():
    need = [os.path.join(DATA, "al", "tcb_regions.json")]
    for n in PAGES:
        need += list(page_paths(n).values())
        need.append(os.path.join(DATA, "al", f"{n}_edit.png"))
    missing = [p for p in need if not os.path.exists(p)]
    if missing:
        skip(f"test pages not found ({missing[0]})")
    from core import text_seg
    if not os.path.exists(text_seg._weights_path()):
        skip(f"text model not installed ({text_seg._weights_path()})")
    if not text_seg.TextSegmenter().ok:
        skip("text model will not load")

    import app

    gt = json.load(open(os.path.join(DATA, "al", "tcb_regions.json")))
    tmpdir = tempfile.mkdtemp()
    try:
        for n in PAGES:
            paths = page_paths(n)
            for kind, path in paths.items():
                img = cv2.imread(path)
                other = cv2.imread(paths["en" if kind == "raw" else "raw"])
                h, w = img.shape[:2]
                lett = app._page_lettering(img)
                assert lett is not None, "model loaded but read nothing"
                keep = app._text_keepout(img, PAD_M, lettering=lett)
                cover = 100.0 * cv2.countNonZero(keep) / (h * w)
                old = app._text_keepout(img, PAD_M, lettering=None)
                old_cover = 100.0 * cv2.countNonZero(old) / (h * w)
                ink, spots = lettered(n, img, other, gt[str(n)]["regions"])

                # 1. Coverage.
                assert cover < MAX_COVER, (
                    f"{kind} {n}: keep-out covers {cover:.1f}% of the page "
                    f"(limit {MAX_COVER}%) — marking more than the lettering")
                assert cover < MAX_OF_HEURISTIC * old_cover, (
                    f"{kind} {n}: keep-out {cover:.1f}% is not well under the "
                    f"shape heuristic's {old_cover:.1f}%")

                # 2. Every lettered spot is inside it.
                if (kind, n) in RECALL:
                    for (x, y, bw, bh), m in spots:
                        got = float((m & (keep > 0)).sum()) / m.sum()
                        assert got >= MIN_RECALL, (
                            f"{kind} {n}: lettering at {(x, y, bw, bh)} only "
                            f"{100 * got:.1f}% inside the keep-out")
                print(f"{kind} {n}: keep-out {cover:.1f}% of the page "
                      f"(heuristic {old_cover:.1f}%)"
                      + (f", all {len(spots)} lettered spots inside"
                         if (kind, n) in RECALL else ""))

                # 3. Where the stamps land, through the real code: the
                # watermark alone, then watermark + credit the way a finished
                # page gets them — the difference is the credit.
                if n != CLEAR_BR:
                    continue
                tmp = os.path.join(tmpdir, f"{kind}{n}.png")
                cv2.imwrite(tmp, img)
                app._stamp_all(tmp, "@MyScanGroup", "br", 100, "m",
                               "", "clean")
                after_wm = cv2.imread(tmp)
                wm = changed_box(img, after_wm)
                cv2.imwrite(tmp, img)
                app._stamp_all(tmp, "@MyScanGroup", "br", 100, "m",
                               "Translated by MyScanGroup", "clean")
                cr = changed_box(after_wm, cv2.imread(tmp))
                assert wm and cr, f"{kind} {n}: a stamp drew nothing"
                near = cv2.dilate(ink.astype(np.uint8), np.ones((7, 7), np.uint8))
                for name, (x0, y0, x1, y1), want in (("watermark", wm, "br"),
                                                     ("credit", cr, "bl")):
                    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                    quad = ("b" if cy >= h / 2 else "t") + \
                           ("r" if cx >= w / 2 else "l")
                    assert quad == want, (
                        f"{kind} {n}: {name} landed in the {quad} quarter "
                        f"at {(x0, y0, x1, y1)}, asked for {want}")
                    # Near the corner, not merely somewhere in the quarter (the
                    # old keep-out left it 864 px up the page). It may step
                    # aside a little: the model also marks the creature's
                    # dripping teeth near this corner.
                    dx = (w - 1 - x1) if want[1] == "r" else x0
                    dy = h - 1 - y1
                    assert dx <= 0.3 * w and dy <= 0.12 * h, (
                        f"{kind} {n}: {name} {dx} px from the side and {dy} px "
                        f"from the bottom — pushed out of its corner")
                    over = int(near[y0:y1 + 1, x0:x1 + 1].sum())
                    assert over == 0, (f"{kind} {n}: {name} covers {over} "
                                       f"px of lettering")
                    print(f"{kind} {n}: {name} at {(x0, y0, x1, y1)} "
                          f"({quad}, clear of lettering)")
                assert wm[0] > cr[2] or cr[0] > wm[2] or \
                    wm[1] > cr[3] or cr[1] > wm[3], "credit on the watermark"

        # 4. Without the model the heuristic still runs (the fallback).
        img = cv2.imread(page_paths(PAGES[0])["raw"])
        k = app._text_keepout(img, PAD_M, lettering=None)
        assert k.shape == img.shape[:2] and cv2.countNonZero(k) > 0
        k = app._dialogue_keepout(img, 25, lettering=None)
        assert k.shape == img.shape[:2]
        print("heuristic fallback OK")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
