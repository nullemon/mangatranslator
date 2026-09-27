"""The editor's brushes, replayed on the server: pen, spot heal, clone.

Each drag is ONE stroke ({"paint"|"heal": {"pts", "r", ...}} or
{"clone": {"src", "dst", "r", "pts"}}). The pen paints its colour along the
path and nothing else; spot heal rebuilds what it went over from the art
around it; a clone stroke copies the art at its offset — read BEFORE the
stroke is painted, so a drag whose source runs into its own fresh paint
copies the art, not the paint (dab after dab used to smear it along).

Run:  python tests/test_brush_tools.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.compositor import Compositor  # noqa: E402


def main():
    comp = Compositor(use_lama=False)

    # ── pen: paints its colour along the path, nothing else ──
    page = np.full((300, 400, 3), 240, np.uint8)
    out = comp.compose(page.copy(), [], covers=[
        {"paint": {"pts": [[100, 150], [300, 150]], "r": 6, "color": "#ff0000"}}])
    mid = out[148:153, 110:290]
    assert (mid[..., 2] > 230).all() and (mid[..., 0] < 20).all(), "pen didn't paint red"
    changed = np.any(out != page, axis=2)
    ys, xs = np.nonzero(changed)
    assert ys.min() >= 150 - 8 and ys.max() <= 150 + 8, "pen painted outside the brush"
    assert xs.min() >= 100 - 8 and xs.max() <= 300 + 8, "pen painted outside the brush"
    print("pen paints its stroke and nothing else OK")

    # ── spot heal: a speck on flat grey is rebuilt as the grey ──
    page = np.full((300, 400, 3), 180, np.uint8)
    cv2.circle(page, (200, 150), 6, (0, 0, 0), -1)
    out = comp.compose(page.copy(), [], covers=[
        {"heal": {"pts": [[200, 150]], "r": 12}}])
    spot = out[140:161, 190:211].astype(int)
    assert np.abs(spot - 180).max() < 25, f"speck not healed (max off {np.abs(spot - 180).max()})"
    assert (out[:, :150] == page[:, :150]).all(), "heal touched far-away art"
    print("spot heal removes a speck OK")

    # ── clone stroke: copies the art at its offset, never its own paint ──
    # every column its own value, so any smear shows as a wrong value
    page = np.zeros((300, 600, 3), np.uint8)
    page[:] = (np.arange(600) % 251).astype(np.uint8)[None, :, None]
    src_x, dst_x = 100, 140              # source 40 px to the LEFT of the brush
    stroke = {"clone": {"src": [src_x, 150], "dst": [dst_x, 150], "r": 20,
                        "pts": [[x, 150] for x in range(dst_x, 461, 5)]}}
    out = comp.compose(page.copy(), [], covers=[stroke])
    got = out[145:156, dst_x + 10:450].astype(int)
    want = page[145:156, dst_x + 10 - 40:450 - 40].astype(int)
    err = np.abs(got - want).max()
    assert err <= 1, f"clone stroke smeared (max off {err})"
    assert (out[:100] == page[:100]).all(), "clone painted outside the stroke"
    print("clone stroke copies the art, no smear OK")

    # an old single dab (saved before strokes) still replays
    out = comp.compose(page.copy(), [], covers=[
        {"clone": {"src": [100, 150], "dst": [300, 150], "r": 20}}])
    assert abs(int(out[150, 300, 0]) - int(page[150, 100, 0])) <= 1, "old clone dab broke"
    print("old clone dabs still work OK")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
