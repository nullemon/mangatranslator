import cv2
import numpy as np
import os
from typing import Callable, Optional, Dict, Any, List

from .detector import BubbleDetector, TextRegion
from .translator import make_translator
from .compositor import Compositor


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left  (smallest x+y)
    rect[2] = pts[np.argmax(s)]      # bottom-right (largest x+y)
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]      # top-right (smallest y-x)
    rect[3] = pts[np.argmax(d)]      # bottom-left (largest y-x)
    return rect


def auto_crop_page(image: np.ndarray) -> np.ndarray:
    """Detect the manga page in a photo and warp it flat (deskew + crop).

    Finds the page's 4 corners and applies a perspective transform so an
    angled phone photo becomes a clean, rectangular, front-on page. Falls
    back to an axis-aligned crop, then to the original, if that fails."""
    h, w = image.shape[:2]
    page_area = h * w
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < page_area * 0.25:
        return image
    # Already fills the frame → nothing to crop.
    bx, by, bw, bh = cv2.boundingRect(largest)
    if bw >= w * 0.98 and bh >= h * 0.98:
        return image

    # Try to approximate the page outline as a 4-corner quad for a perspective
    # warp (fixes rotation + keystone). Loosen epsilon until we get a quad.
    peri = cv2.arcLength(largest, True)
    quad = None
    for eps in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08):
        approx = cv2.approxPolyDP(largest, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2).astype(np.float32)
            break

    if quad is not None:
        rect = _order_corners(quad)
        (tl, tr, br, bl) = rect
        wA = np.linalg.norm(br - bl)
        wB = np.linalg.norm(tr - tl)
        hA = np.linalg.norm(tr - br)
        hB = np.linalg.norm(tl - bl)
        out_w = int(max(wA, wB))
        out_h = int(max(hA, hB))
        if out_w >= w * 0.35 and out_h >= h * 0.35:
            dst = np.array([[0, 0], [out_w - 1, 0],
                            [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (out_w, out_h))
            print(f"[pipeline] auto-crop+deskew: {w}x{h} -> {out_w}x{out_h}")
            return warped

    # Fallback: axis-aligned crop.
    if bw < w * 0.35 or bh < h * 0.35:
        return image
    pad = 3
    bx, by = max(0, bx - pad), max(0, by - pad)
    bw = min(w - bx, bw + 2 * pad)
    bh = min(h - by, bh + 2 * pad)
    print(f"[pipeline] auto-crop: {w}x{h} -> {bw}x{bh}")
    return image[by:by + bh, bx:bx + bw].copy()


def scan_cleanup(image: np.ndarray) -> np.ndarray:
    """Turn a phone photo into a clean 'scanned' page, locally and reliably:
    deskew + crop away the background, then normalize lighting so the paper
    goes pure white — while preserving solid blacks, screentones and ink."""
    img = auto_crop_page(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape[:2]

    # 1. Flatten uneven lighting / shadows by dividing out a large-kernel
    #    background estimate → paper becomes pure white. (This step alone
    #    washes out big black regions, so we repair them in step 3.)
    k = max(31, (min(h, w) // 8) | 1)
    bg = cv2.morphologyEx(
        gray.astype(np.uint8), cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
    ).astype(np.float32)
    flat = np.clip(gray / np.maximum(bg, 1.0) * 255.0, 0, 255)

    # 2. Global levels stretch of the ORIGINAL → keeps solid blacks black.
    bp = float(np.percentile(gray, 2))
    wp = float(np.percentile(gray, 95))
    if wp - bp < 20:
        return cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    stretched = np.clip((gray - bp) * (255.0 / (wp - bp)), 0, 255)

    # 3. Combine: paper stays white (from flat), inks and large black areas
    #    are restored (from stretched) via per-pixel minimum.
    out = np.minimum(flat, stretched)
    # 4. Snap near-white paper to pure white for a clean scanned look.
    out[out > 225] = 255
    out = out.astype(np.uint8)
    # 5. Light denoise to remove paper grain without smearing line art.
    out = cv2.fastNlMeansDenoising(out, None, 5, 7, 21)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


_SFX_KINDS = {"sfx", "sound", "sound_effect", "soundeffect", "onomatopoeia"}


def _boxes_overlap(a, b, thresh=0.3) -> bool:
    """True if box a (x,y,w,h) overlaps b enough to be the same region."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    xi, yi = max(ax, bx), max(ay, by)
    xf, yf = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if xi >= xf or yi >= yf:
        return False
    inter = (xf - xi) * (yf - yi)
    smaller = max(min(aw * ah, bw * bh), 1)
    # Also treat "center of one inside the other" as overlap.
    acx, acy = ax + aw / 2, ay + ah / 2
    bcx, bcy = bx + bw / 2, by + bh / 2
    center_in = (bx <= acx <= bx + bw and by <= acy <= by + bh) or \
                (ax <= bcx <= ax + aw and ay <= bcy <= ay + ah)
    return inter / smaller > thresh or center_in


def _det_to_bbox(det, w, h):
    x = max(0, min(int(det.get("x_pct", 0) / 100 * w), w - 1))
    y = max(0, min(int(det.get("y_pct", 0) / 100 * h), h - 1))
    bw = max(8, min(int(det.get("width_pct", 0) / 100 * w), w - x))
    bh = max(8, min(int(det.get("height_pct", 0) / 100 * h), h - y))
    return [x, y, bw, bh]


def _has_text_strokes(gray, bbox, lo=0.005, hi=0.85) -> bool:
    """True if the box plausibly contains text: some — but not overwhelming —
    dark ink. Filters out blank areas (AI hallucinations) and solid-black art."""
    x, y, bw, bh = bbox
    roi = gray[y:y + bh, x:x + bw]
    if roi.size == 0:
        return False
    _, th = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    frac = float(cv2.countNonZero(th)) / roi.size
    return lo < frac < hi


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

        update(0, "Preprocessing image...", 2)
        image = auto_crop_page(image)

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
        bubble_count = len(regions) if regions else 0
        if bubble_count:
            update(1, f"Found {bubble_count} balloons", 22)

        ann_path = ""
        items, masks = [], {}

        if regions:
            annotated = self.detector.create_annotated_image(image, regions)
            ann_path = self._suffix_path(output_path, "annotated")
            cv2.imwrite(ann_path, annotated)

            update(2, "Translating bubbles...", 32)
            translations = self.translator.translate_regions(
                image, annotated, len(regions), self.target_lang
            )
            update(2, f"Translated {len(translations)} bubble regions", 48)

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

        # NOTE: the AI "completeness pass" was removed — its imprecise
        # coordinates dumped duplicate/free text into the gutters. Catching
        # missed text is now handled precisely by the local GPU detector
        # (comic-text-detector) when available.
        return items, ann_path, masks

    def _add_missed_text(self, image, items, update) -> int:
        """Run AI full-page detection and append any text region that the
        bubble detector didn't already cover. Returns how many were added."""
        h, w = image.shape[:2]
        try:
            dets = self.translator.smart_detect_and_translate(image, self.target_lang)
            print(f"[pipeline] completeness scan found {len(dets)} total text regions")
        except Exception as e:
            print(f"[pipeline] completeness scan failed: {e}")
            return 0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        existing = [it["bbox"] for it in items]
        next_id = max((it["id"] for it in items), default=0) + 1
        added = 0
        for det in dets:
            kind = (det.get("type") or "").lower().replace(" ", "_")
            if kind in _SFX_KINDS:
                continue
            text = (det.get("translation") or "").strip()
            if not text:
                continue
            bbox = _det_to_bbox(det, w, h)
            if bbox[2] < 8 or bbox[3] < 8:
                continue
            if any(_boxes_overlap(bbox, e) for e in existing):
                continue  # already handled by a detected bubble
            if not _has_text_strokes(gray, bbox):
                continue  # AI imagined text on a blank/empty area — skip it
            in_bubble = det.get("in_bubble", True)
            items.append({
                "id": next_id,
                "bbox": bbox,
                "original": det.get("original", ""),
                "translation": text,
                "type": kind or ("dialogue" if in_bubble else "caption"),
                "in_bubble": bool(in_bubble),
                "dark": False,
            })
            existing.append(bbox)
            next_id += 1
            added += 1
        return added

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
