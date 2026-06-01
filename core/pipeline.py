import cv2
import numpy as np
import os
from typing import Callable, Optional, Dict, Any

from .detector import BubbleDetector, TextRegion
from .translator import make_translator
from .inpainter import TextInpainter
from .renderer import TextRenderer


class TranslationPipeline:
    def __init__(
        self,
        api_key: str,
        target_lang: str = "English",
        model: str = "",
        font_path: Optional[str] = None,
        use_smart_detection: bool = False,
        provider: str = "claude",
    ):
        self.detector = BubbleDetector()
        self.translator = make_translator(provider, api_key, model)
        self.inpainter = TextInpainter()
        self.renderer = TextRenderer(font_path)
        self.target_lang = target_lang
        self.use_smart_detection = use_smart_detection

    def process(
        self,
        image_path: str,
        output_path: str,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        def update(step: int, msg: str, pct: int):
            if progress_cb:
                progress_cb({"step": step, "message": msg, "progress": pct})

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        h, w = image.shape[:2]
        max_dim = 4000
        scale = 1.0
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        if self.use_smart_detection:
            return self._smart_pipeline(image, output_path, update)
        return self._standard_pipeline(image, output_path, update)

    def _standard_pipeline(
        self,
        image: np.ndarray,
        output_path: str,
        update: Callable,
    ) -> Dict[str, Any]:
        update(1, "Detecting text regions...", 10)
        regions = self.detector.detect(image)

        if not regions:
            cv2.imwrite(output_path, image)
            update(5, "No text regions found.", 100)
            return self._result(output_path, [], {})

        update(1, f"Found {len(regions)} text regions", 20)

        annotated = self.detector.create_annotated_image(image, regions)
        ann_path = output_path.replace(".", "_annotated.", 1)
        cv2.imwrite(ann_path, annotated)

        update(2, "Translating with Claude...", 30)
        translations = self.translator.translate_regions(
            image, annotated, len(regions), self.target_lang
        )
        update(2, f"Translated {len(translations)} regions", 55)

        update(3, "Removing original text...", 60)
        cleaned = self.inpainter.remove_text(image, regions)
        update(3, "Text removed", 75)

        update(4, "Rendering translations...", 80)
        result = self.renderer.render(cleaned, regions, translations)
        update(4, "Rendered", 90)

        cv2.imwrite(output_path, result)
        update(5, "Complete!", 100)

        return self._result(output_path, regions, translations, ann_path)

    def _smart_pipeline(
        self,
        image: np.ndarray,
        output_path: str,
        update: Callable,
    ) -> Dict[str, Any]:
        update(1, "Claude is analyzing the page...", 10)
        detections = self.translator.smart_detect_and_translate(
            image, self.target_lang
        )
        update(2, f"Found {len(detections)} text regions", 40)

        if not detections:
            cv2.imwrite(output_path, image)
            update(5, "No text found.", 100)
            return self._result(output_path, [], {})

        h, w = image.shape[:2]
        regions = []
        translations = {}

        for i, det in enumerate(detections):
            x = int(det["x_pct"] / 100 * w)
            y = int(det["y_pct"] / 100 * h)
            bw = int(det["width_pct"] / 100 * w)
            bh = int(det["height_pct"] / 100 * h)

            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            bw = max(20, min(bw, w - x))
            bh = max(15, min(bh, h - y))

            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y : y + bh, x : x + bw] = 255

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            roi_gray = gray[y : y + bh, x : x + bw]
            _, text_bin = cv2.threshold(roi_gray, 120, 255, cv2.THRESH_BINARY_INV)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            text_bin = cv2.dilate(text_bin, k, iterations=2)

            text_mask = np.zeros((h, w), dtype=np.uint8)
            text_mask[y : y + bh, x : x + bw] = text_bin

            rid = i + 1
            contour = np.array(
                [[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]]
            ).reshape(-1, 1, 2)

            regions.append(
                TextRegion(
                    id=rid,
                    bbox=(x, y, bw, bh),
                    contour=contour,
                    mask=mask,
                    center=(x + bw // 2, y + bh // 2),
                    area=bw * bh,
                    text_mask=text_mask,
                    region_type=det.get("type", "bubble"),
                )
            )
            translations[rid] = {
                "original": det.get("original", ""),
                "translation": det.get("translation", ""),
                "type": det.get("type", "dialogue"),
            }

        update(3, "Removing original text...", 60)
        cleaned = self.inpainter.remove_text(image, regions)
        update(3, "Text removed", 75)

        update(4, "Rendering translations...", 80)
        result = self.renderer.render(cleaned, regions, translations)
        update(4, "Rendered", 90)

        cv2.imwrite(output_path, result)
        update(5, "Complete!", 100)

        return self._result(output_path, regions, translations)

    def _result(
        self,
        output_path: str,
        regions: list,
        translations: dict,
        annotated_path: str = "",
    ) -> Dict[str, Any]:
        return {
            "output_path": output_path,
            "annotated_path": annotated_path,
            "regions": [
                {"id": r.id, "bbox": list(r.bbox), "type": r.region_type}
                for r in regions
            ],
            "translations": {
                str(k): {
                    "original": v.get("original", ""),
                    "translation": v.get("translation", ""),
                    "type": v.get("type", ""),
                }
                for k, v in translations.items()
            },
            "num_regions": len(regions),
            "num_translated": len(translations),
        }
