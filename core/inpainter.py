import cv2
import numpy as np
from typing import List


class TextInpainter:
    def __init__(self, inpaint_radius: int = 7):
        self.inpaint_radius = inpaint_radius

    def remove_text(self, image: np.ndarray, regions: list) -> np.ndarray:
        result = image.copy()

        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for region in regions:
            if region.text_mask is None:
                continue

            x, y, w, h = region.bbox
            bubble_roi = region.mask[y : y + h, x : x + w]
            text_roi = region.text_mask[y : y + h, x : x + w]

            non_text = cv2.bitwise_and(bubble_roi, cv2.bitwise_not(text_roi))
            gray_roi = cv2.cvtColor(result[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
            bg_pixels = gray_roi[non_text > 0]

            if len(bg_pixels) > 0 and np.mean(bg_pixels > 200) > 0.7:
                mask_full = region.text_mask > 0
                result[mask_full] = [255, 255, 255]
            else:
                combined_mask = cv2.bitwise_or(combined_mask, region.text_mask)

        if np.any(combined_mask > 0):
            result = cv2.inpaint(
                result, combined_mask, self.inpaint_radius, cv2.INPAINT_NS
            )

        return result

    def clean_bubble_interior(
        self, image: np.ndarray, region, padding: int = 3
    ) -> np.ndarray:
        result = image.copy()
        x, y, w, h = region.bbox

        bubble_mask_roi = region.mask[y : y + h, x : x + w]
        k = np.ones((padding * 2 + 1, padding * 2 + 1), np.uint8)
        eroded = cv2.erode(bubble_mask_roi, k, iterations=1)

        gray_roi = cv2.cvtColor(result[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
        bg_pixels = gray_roi[eroded > 0]

        if len(bg_pixels) > 0:
            median_val = int(np.median(bg_pixels))
            fill_color = [median_val, median_val, median_val]

            text_area = region.text_mask[y : y + h, x : x + w]
            fill_mask = cv2.bitwise_and(text_area, eroded)

            roi = result[y : y + h, x : x + w]
            roi[fill_mask > 0] = fill_color
            result[y : y + h, x : x + w] = roi

        return result
