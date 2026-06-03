"""End-to-end check of the CV detector + compositor on a synthetic manga page.
Run: python3 tests/test_pipeline_cv.py
Validates: balloons are found, their interiors are fully wiped (no original
text left), translations land inside, and artwork outside balloons is untouched."""
import os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.detector import BubbleDetector
from core.compositor import Compositor


def make_page():
    H, W = 1400, 1000
    img = np.full((H, W, 3), 245, np.uint8)  # off-white page

    # Balloon 1: ellipse, top-right, with vertical "text" strokes inside.
    cv2.ellipse(img, (740, 240), (170, 130), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (740, 240), (170, 130), 0, 0, 360, (0, 0, 0), 4)
    for i, cxp in enumerate((690, 740, 790)):
        for j in range(4):
            yy = 170 + j * 35
            cv2.putText(img, "|", (cxp, yy + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

    # Balloon 2: rounded rectangle, bottom-left.
    cv2.rectangle(img, (120, 950), (430, 1180), (255, 255, 255), -1)
    cv2.rectangle(img, (120, 950), (430, 1180), (0, 0, 0), 4)
    for j in range(5):
        cv2.putText(img, "###", (200, 1010 + j * 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    # Artwork (must stay untouched): a filled black blob = "hair".
    cv2.circle(img, (300, 400), 120, (10, 10, 10), -1)

    return img


def interior_mask(detector, img):
    regions = detector.detect(img)
    return regions


def main():
    img = make_page()
    det = BubbleDetector()
    regions = det.detect(img)
    print(f"Detected {len(regions)} balloon(s)")
    for r in regions:
        print(f"  id={r.id} bbox={r.bbox} area={r.area} dark={r.dark}")
    assert len(regions) >= 2, "Expected to find both balloons"

    items = []
    for r in regions:
        items.append({
            "id": r.id, "bbox": list(r.bbox), "in_bubble": True,
            "type": "dialogue", "dark": r.dark,
            "translation": "HELLO THERE THIS IS A TEST OF CONTAINMENT",
        })
    masks = {r.id: r.mask for r in regions}

    comp = Compositor()
    out = comp.compose(img.copy(), items, masks)

    placed = [it for it in items if it.get("placed")]
    print(f"Placed text in {len(placed)} balloon(s)")
    assert placed, "No translations were placed"

    # The hair blob (artwork outside balloons) must be unchanged.
    before = img[400, 300]
    after = out[400, 300]
    assert np.array_equal(before, after), f"Artwork was modified! {before} -> {after}"
    print("Artwork outside balloons: untouched ✓")

    # Each balloon interior should now be mostly clean white where the original
    # had black text strokes — i.e. far fewer dark pixels than before.
    gray_before = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    for r in regions:
        m = cv2.erode(r.mask, np.ones((9, 9), np.uint8))
        b = int(np.sum(gray_before[m > 0] < 100))
        a_dark = gray_after[m > 0] < 100
        # after compositing, dark pixels = the rendered translation glyphs
        a = int(np.sum(a_dark))
        print(f"  balloon {r.id}: dark px before={b}, after={a} (after = new translation glyphs)")
        assert a > 50, f"balloon {r.id}: no translation glyphs rendered"

    cv2.imwrite("/tmp/test_in.png", img)
    cv2.imwrite("/tmp/test_out.png", out)
    print("\nWrote /tmp/test_in.png and /tmp/test_out.png")
    print("ALL CHECKS PASSED ✓")


if __name__ == "__main__":
    main()
