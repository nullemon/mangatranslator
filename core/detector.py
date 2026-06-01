import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class TextRegion:
    id: int
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    contour: np.ndarray
    mask: np.ndarray
    center: Tuple[int, int]
    area: int
    text_mask: Optional[np.ndarray] = None
    region_type: str = "bubble"


class BubbleDetector:
    def __init__(
        self,
        white_thresh: int = 200,
        text_thresh: int = 120,
        min_area_ratio: float = 0.0015,
        max_area_ratio: float = 0.25,
        min_text_ratio: float = 0.005,
    ):
        self.white_thresh = white_thresh
        self.text_thresh = text_thresh
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.min_text_ratio = min_text_ratio

    def detect(self, image: np.ndarray) -> List[TextRegion]:
        h, w = image.shape[:2]
        img_area = h * w
        min_area = int(img_area * self.min_area_ratio)
        max_area = int(img_area * self.max_area_ratio)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(gray, self.white_thresh, 255, cv2.THRESH_BINARY)

        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close, iterations=3)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open, iterations=2)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

        regions: List[TextRegion] = []

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area or area > max_area:
                continue

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]

            component_mask = (labels == i).astype(np.uint8) * 255

            roi_gray = gray[y : y + bh, x : x + bw]
            roi_mask = component_mask[y : y + bh, x : x + bw]

            dark_pixels = np.sum((roi_gray < self.text_thresh) & (roi_mask > 0))
            text_ratio = dark_pixels / max(area, 1)

            if text_ratio < self.min_text_ratio:
                continue

            contours_found, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours_found:
                continue

            contour = max(contours_found, key=cv2.contourArea)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue

            solidity = area / hull_area
            if solidity < 0.35:
                continue

            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if aspect > 8:
                continue

            text_mask = self._build_text_mask(gray, component_mask, x, y, bw, bh)

            center = (int(centroids[i][0]), int(centroids[i][1]))
            regions.append(
                TextRegion(
                    id=0,
                    bbox=(x, y, bw, bh),
                    contour=contour,
                    mask=component_mask,
                    center=center,
                    area=area,
                    text_mask=text_mask,
                    region_type=self._classify_region(solidity, aspect, bw, bh),
                )
            )

        regions = self._merge_overlapping(regions, h, w)
        self._sort_reading_order(regions, h)

        for idx, r in enumerate(regions):
            r.id = idx + 1

        return regions

    def _build_text_mask(
        self,
        gray: np.ndarray,
        bubble_mask: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> np.ndarray:
        roi_gray = gray[y : y + h, x : x + w]
        roi_bubble = bubble_mask[y : y + h, x : x + w]

        _, text_bin = cv2.threshold(roi_gray, self.text_thresh, 255, cv2.THRESH_BINARY_INV)
        text_bin = cv2.bitwise_and(text_bin, roi_bubble)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        text_bin = cv2.dilate(text_bin, k, iterations=2)
        text_bin = cv2.bitwise_and(text_bin, roi_bubble)

        full_mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        full_mask[y : y + h, x : x + w] = text_bin
        return full_mask

    def _classify_region(
        self, solidity: float, aspect: float, w: int, h: int
    ) -> str:
        if solidity > 0.85 and aspect < 2.0:
            return "bubble"
        if aspect > 2.5:
            return "box"
        return "bubble"

    def _merge_overlapping(
        self, regions: List[TextRegion], img_h: int, img_w: int
    ) -> List[TextRegion]:
        if len(regions) <= 1:
            return regions

        merged = []
        used = set()

        for i, r1 in enumerate(regions):
            if i in used:
                continue
            for j, r2 in enumerate(regions):
                if j <= i or j in used:
                    continue
                if self._iou(r1.bbox, r2.bbox) > 0.3:
                    if r1.area >= r2.area:
                        used.add(j)
                    else:
                        used.add(i)
                        break
            if i not in used:
                merged.append(r1)

        return merged

    def _iou(self, b1: tuple, b2: tuple) -> float:
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2

        xi = max(x1, x2)
        yi = max(y1, y2)
        xf = min(x1 + w1, x2 + w2)
        yf = min(y1 + h1, y2 + h2)

        if xi >= xf or yi >= yf:
            return 0.0

        inter = (xf - xi) * (yf - yi)
        union = w1 * h1 + w2 * h2 - inter
        return inter / max(union, 1)

    def _sort_reading_order(self, regions: List[TextRegion], img_h: int):
        row_h = img_h / max(4, len(regions) // 2)
        regions.sort(key=lambda r: (int(r.center[1] / row_h), -r.center[0]))

    def create_annotated_image(
        self, image: np.ndarray, regions: List[TextRegion]
    ) -> np.ndarray:
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
            cv2.putText(
                ann,
                label,
                (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (255, 255, 255),
                2,
            )

        return ann
