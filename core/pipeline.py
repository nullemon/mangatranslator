import cv2
import numpy as np
import os
from typing import Callable, Optional, Dict, Any, List

from .detector import BubbleDetector, TextRegion
from .translator import make_translator
from .compositor import Compositor


def make_detector(use_seg: bool = True):
    """Prefer the GPU segmentation model; fall back to CV when it's unavailable."""
    if use_seg:
        try:
            from .bubble_seg import BubbleSegDetector
            d = BubbleSegDetector()
            if d.ok:
                return d, "segmentation model (GPU)"
        except Exception as e:
            print(f"[pipeline] seg detector unavailable: {e}")
    return BubbleDetector(), "CV detector"


class TranslationPipeline:
    def __init__(
        self,
        api_key: str,
        target_lang: str = "English",
        model: str = "",
        font_path: Optional[str] = None,
        use_smart_detection: bool = False,
        provider: str = "claude",
        use_seg: bool = True,
    ):
        self.detector, self.detector_name = make_detector(use_seg)
        self.translator = make_translator(provider, api_key, model)
        self.compositor = Compositor(font_path)
        self.target_lang = target_lang
        self.use_smart_detection = use_smart_detection
        self.last_masks: Dict[int, np.ndarray] = {}

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
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # Persist the exact image we operate on so the comparison view aligns
        # and re-renders start from a pristine (un-composited) base.
        base_path = self._base_path(output_path)
        cv2.imwrite(base_path, image)

        self.last_masks = {}
        if self.use_smart_detection:
            items, ann_path, masks = self._smart_detect(image, output_path, update)
        else:
            items, ann_path, masks = self._standard_detect(image, output_path, update)
        self.last_masks = masks

        if not items:
            cv2.imwrite(output_path, image)
            update(5, "No text regions found.", 100)
            return self._result(output_path, base_path, [], ann_path)

        update(3, "Erasing original text...", 60)
        update(4, "Fitting translations into balloons...", 80)
        result = self.compositor.compose(image, items, masks)
        cv2.imwrite(output_path, result)
        update(5, "Complete!", 100)

        return self._result(output_path, base_path, items, ann_path)

    # ── Detection strategies → (items, annotated_path, masks) ──
    def _standard_detect(self, image, output_path, update):
        update(1, f"Detecting balloons with {self.detector_name}...", 10)
        regions: List[TextRegion] = self.detector.detect(image)
        if not regions:
            return [], "", {}
        update(1, f"Found {len(regions)} balloons", 22)

        annotated = self.detector.create_annotated_image(image, regions)
        ann_path = self._suffix_path(output_path, "annotated")
        cv2.imwrite(ann_path, annotated)

        update(2, "Translating...", 32)
        translations = self.translator.translate_regions(
            image, annotated, len(regions), self.target_lang
        )
        update(2, f"Translated {len(translations)} regions", 55)

        items, masks = [], {}
        for r in regions:
            tr = translations.get(r.id, {})
            items.append({
                "id": r.id,
                "bbox": [int(v) for v in r.bbox],
                "original": tr.get("original", ""),
                "translation": tr.get("translation", ""),
                "type": tr.get("type", "dialogue"),
                "in_bubble": True,
                "dark": bool(getattr(r, "dark", False)),
            })
            masks[r.id] = r.mask
        return items, ann_path, masks

    def _smart_detect(self, image, output_path, update):
        update(1, "AI is analyzing the page...", 10)
        detections = self.translator.smart_detect_and_translate(image, self.target_lang)
        update(2, f"Found {len(detections)} text regions", 45)
        if not detections:
            return [], "", {}

        h, w = image.shape[:2]
        items = []
        for i, det in enumerate(detections):
            x = max(0, min(int(det.get("x_pct", 0) / 100 * w), w - 1))
            y = max(0, min(int(det.get("y_pct", 0) / 100 * h), h - 1))
            bw = max(10, min(int(det.get("width_pct", 0) / 100 * w), w - x))
            bh = max(10, min(int(det.get("height_pct", 0) / 100 * h), h - y))
            items.append({
                "id": i + 1,
                "bbox": [x, y, bw, bh],
                "original": det.get("original", ""),
                "translation": det.get("translation", ""),
                "type": det.get("type", "dialogue"),
                "in_bubble": det.get("in_bubble", True),
                "dark": False,
            })
        return items, "", {}

    # ── Re-render with an edited / filtered item set ──
    def recompose(self, base_path: str, items: List[dict], output_path: str,
                  masks: Optional[Dict] = None) -> np.ndarray:
        image = cv2.imread(base_path)
        if image is None:
            raise ValueError(f"Cannot load base image: {base_path}")
        result = self.compositor.compose(image, items, masks)
        cv2.imwrite(output_path, result)
        return result

    # ── Helpers ──
    def _base_path(self, output_path: str) -> str:
        return self._suffix_path(output_path, "base")

    def _suffix_path(self, output_path: str, suffix: str) -> str:
        root, _ = os.path.splitext(output_path)
        return f"{root}_{suffix}.png"

    def _result(self, output_path, base_path, items, annotated_path=""):
        return {
            "output_path": output_path,
            "base_path": base_path,
            "annotated_path": annotated_path,
            "detector": self.detector_name,
            "items": [
                {
                    "id": it["id"],
                    "bbox": it["bbox"],
                    "original": it.get("original", ""),
                    "translation": it.get("translation", ""),
                    "type": it.get("type", ""),
                    "in_bubble": it.get("in_bubble", True),
                    "dark": it.get("dark", False),
                    "placed": it.get("placed", False),
                }
                for it in items
            ],
            "translations": {
                str(it["id"]): {
                    "original": it.get("original", ""),
                    "translation": it.get("translation", ""),
                    "type": it.get("type", ""),
                }
                for it in items
            },
            "num_regions": len(items),
            "num_translated": sum(1 for it in items if it.get("placed")),
        }
