import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class TextRegion:
    id: int
    bbox: Tuple[int, int, int, int]      # x, y, w, h
    mask: np.ndarray                      # full-page uint8, 255 = bubble interior
    center: Tuple[int, int]
    area: int
    dark: bool = False                   # True = dark bubble (white text)
    region_type: str = "bubble"


class BubbleDetector:
    """Finds speech balloons by their defining property: a bright (or dark)
    region fully *enclosed* by an inked outline.

    The page background and open panel areas are bright too, but they reach the
    page border, so a flood/label pass separates them from the enclosed balloon
    interiors. Interior text only punches holes in that bright region, which we
    fill back in to recover the true balloon shape as a solid mask.
    """

    def __init__(
        self,
        white_thresh: int = 188,
        dark_thresh: int = 90,
        min_area_ratio: float = 0.0006,
        max_area_ratio: float = 0.22,
    ):
        self.white_thresh = white_thresh
        self.dark_thresh = dark_thresh
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

    # ── public ──
    def detect(self, image: np.ndarray) -> List[TextRegion]:
        h, w = image.shape[:2]
        page_area = h * w
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        regions: List[TextRegion] = []
        # Bright balloons (the common case): interior brighter than the outline.
        for mask in self._enclosed_blobs(gray, bright=True):
            r = self._make_region(gray, mask, page_area, dark=False)
            if r is not None:
                regions.append(r)
        # Dark balloons (white text on black): interior darker than the page.
        for mask in self._enclosed_blobs(gray, bright=False):
            r = self._make_region(gray, mask, page_area, dark=True)
            if r is not None:
                regions.append(r)

        regions = self._merge_overlapping(regions)
        self._sort_reading_order(regions, h)
        for idx, r in enumerate(regions):
            r.id = idx + 1
        return regions

    # ── core: enclosed bright/dark blobs ──
    def _enclosed_blobs(self, gray: np.ndarray, bright: bool) -> List[np.ndarray]:
        h, w = gray.shape[:2]
        page_area = h * w

        if bright:
            _, region = cv2.threshold(gray, self.white_thresh, 255, cv2.THRESH_BINARY)
        else:
            _, region = cv2.threshold(gray, self.dark_thresh, 255, cv2.THRESH_BINARY_INV)

        # Seal hairline gaps in the outline so the interior stays separated from
        # the page background (a leak would merge balloon + page into one blob).
        seal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        if bright:
            ink = cv2.morphologyEx(cv2.bitwise_not(region), cv2.MORPH_CLOSE, seal)
            region = cv2.bitwise_not(ink)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(region, 8)

        # Labels that touch the page border belong to the background / open areas.
        border = set(labels[0, :]) | set(labels[h - 1, :]) | set(labels[:, 0]) | set(labels[:, w - 1])

        min_pocket = max(int(page_area * 0.0002), 40)
        enclosed = np.zeros((h, w), np.uint8)
        for i in range(1, num):
            if i in border:
                continue
            if stats[i, cv2.CC_STAT_AREA] < min_pocket:
                continue
            enclosed[labels == i] = 255

        if not enclosed.any():
            return []

        # Each enclosed pocket ≈ one balloon interior (text just punches holes).
        # Fill those holes per-component to recover the solid balloon shape.
        num2, labels2, stats2, _ = cv2.connectedComponentsWithStats(enclosed, 8)
        masks: List[np.ndarray] = []
        for i in range(1, num2):
            if stats2[i, cv2.CC_STAT_AREA] < min_pocket:
                continue
            comp = (labels2 == i).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            filled = np.zeros((h, w), np.uint8)
            cv2.drawContours(filled, [cnt], -1, 255, -1)
            masks.append(filled)
        return masks

    def _make_region(self, gray, mask, page_area, dark: bool) -> Optional[TextRegion]:
        area = int(cv2.countNonZero(mask))
        if area < page_area * self.min_area_ratio or area > page_area * self.max_area_ratio:
            return None

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        cnt = max(cnts, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw == 0 or bh == 0:
            return None

        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if aspect > 6.0:
            return None

        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        if hull_area <= 0 or area / hull_area < 0.62:
            return None

        # Interior must look like a balloon, not inked art: a bright balloon is
        # mostly bright with sparse dark text; a dark balloon is the inverse.
        inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        vals = gray[inner > 0]
        if vals.size == 0:
            return None
        if dark:
            # A dark balloon is solidly dark AND holds white *lettering* —
            # several separated character blobs. This is what rejects eyeballs
            # (a single glint) and open mouths (teeth/highlights): they are dark
            # enclosed regions too, but they are NOT speech bubbles.
            fill_ratio = float(np.mean(vals < self.dark_thresh + 40))
            if fill_ratio < 0.55:
                return None
            if not self._dark_has_lettering(gray, inner):
                return None
        else:
            fill_ratio = float(np.mean(vals > self.white_thresh - 30))
            text_ratio = float(np.mean(vals < 110))
            if fill_ratio < 0.45:
                return None
            if text_ratio < 0.002 or text_ratio > 0.55:
                return None

        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] else x + bw // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] else y + bh // 2
        return TextRegion(
            id=0, bbox=(x, y, bw, bh), mask=mask, center=(cx, cy),
            area=area, dark=dark, region_type="bubble",
        )

    def _dark_has_lettering(self, gray: np.ndarray, inner_mask: np.ndarray) -> bool:
        """True only if a dark region contains white lettering: multiple
        separated character-like blobs covering a text-like fraction of the
        area. An eyeball has a single glint; an open mouth has teeth or a
        highlight — neither forms several distinct character blobs, so both
        are rejected here."""
        area = int(cv2.countNonZero(inner_mask))
        if area == 0:
            return False
        white = ((inner_mask > 0) & (gray > 160)).astype(np.uint8) * 255
        white_frac = float(cv2.countNonZero(white)) / area
        # Too little white = a glint (eye); too much = not real dark text.
        if white_frac < 0.04 or white_frac > 0.55:
            return False
        n, _, stats, _ = cv2.connectedComponentsWithStats(white, 8)
        min_blob = max(area * 0.0015, 6)
        max_blob = area * 0.45  # a single huge white patch isn't lettering
        chars = 0
        for i in range(1, n):
            a = stats[i, cv2.CC_STAT_AREA]
            if min_blob <= a <= max_blob:
                chars += 1
        return chars >= 2

    # ── dedupe / order ──
    def _merge_overlapping(self, regions: List[TextRegion]) -> List[TextRegion]:
        if len(regions) <= 1:
            return regions
        regions = sorted(regions, key=lambda r: r.area, reverse=True)
        kept: List[TextRegion] = []
        for r in regions:
            if any(self._iou(r.bbox, k.bbox) > 0.3 for k in kept):
                continue
            kept.append(r)
        return kept

    def _iou(self, b1, b2) -> float:
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        xi, yi = max(x1, x2), max(y1, y2)
        xf, yf = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        if xi >= xf or yi >= yf:
            return 0.0
        inter = (xf - xi) * (yf - yi)
        return inter / max(w1 * h1 + w2 * h2 - inter, 1)

    def _sort_reading_order(self, regions: List[TextRegion], img_h: int):
        # Manga reads right-to-left, top-to-bottom.
        row_h = max(img_h / max(4, len(regions) // 2 + 1), 1)
        regions.sort(key=lambda r: (int(r.center[1] / row_h), -r.center[0]))

    # ── annotation for the vision model ──
    def create_annotated_image(self, image: np.ndarray, regions: List[TextRegion]) -> np.ndarray:
        return draw_annotations(image, regions)


def draw_annotations(image: np.ndarray, regions: List[TextRegion]) -> np.ndarray:
    """Numbered red boxes over each detected region, for the vision model."""
    ann = image.copy()
    for region in regions:
        x, y, w, h = region.bbox
        cv2.rectangle(ann, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cx = x + w // 2
        cy = y - 25 if y > 40 else y + h + 25
        cv2.circle(ann, (cx, cy), 22, (0, 0, 255), -1)
        cv2.circle(ann, (cx, cy), 22, (255, 255, 255), 2)
        label = str(region.id)
        scale = 0.8 if region.id < 10 else 0.6
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.putText(ann, label, (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
    return ann
