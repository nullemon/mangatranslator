import cv2
import numpy as np
import os
from typing import Callable, Optional, Dict, Any, List

from .detector import BubbleDetector, TextRegion
from .translator import make_translator
from .compositor import Compositor


def _is_sfx(text: str) -> bool:
    """True if text looks like a sound effect (SFX / onomatopoeia).
    Manga SFX are short, mostly-katakana text: ドン, ガッ, ゴゴゴ, etc."""
    t = text.strip().replace(" ", "").replace("\n", "")
    if not t:
        return False
    n = len(t)
    kata = sum(1 for c in t if '゠' <= c <= 'ヿ' or '･' <= c <= 'ﾟ')
    if n <= 5 and kata / max(n, 1) > 0.6:
        return True
    if n <= 3 and kata > 0:
        return True
    return False


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
    back to an axis-aligned crop, then to the original, if that fails.

    The page is found as everything that differs from the photo's border tone,
    so DARK page content — black panels, or a score/timer strip along the very
    bottom — counts as page and is never sliced off. A plain bright threshold
    would treat that dark strip as background and crop the content away."""
    h, w = image.shape[:2]
    page_area = h * w
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Page mask = pixels that differ from the surrounding background (sampled
    # from the image border), so both bright paper and dark inked content count.
    border = np.concatenate([blurred[0, :], blurred[-1, :], blurred[:, 0], blurred[:, -1]])
    bg = int(round(float(np.median(border))))
    diff = cv2.absdiff(blurred, np.full_like(blurred, bg))
    _, mask = cv2.threshold(diff, 24, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    k = max(9, (min(h, w) // 40) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    # Full extent of the page content = union of every non-trivial blob, so a
    # detached corner or the bottom UI strip is included, not cropped away.
    sig = [c for c in contours if cv2.contourArea(c) > page_area * 0.004]
    if not sig:
        return image
    allpts = np.vstack(sig)
    bx, by, bw, bh = cv2.boundingRect(allpts)
    if bw * bh < page_area * 0.25:
        return image
    # Already fills the frame → nothing to crop (don't risk shaving content).
    if bw >= w * 0.95 and bh >= h * 0.95:
        return image

    # Try to approximate the page outline as a 4-corner quad for a perspective
    # warp (fixes rotation + keystone). Loosen epsilon until we get a quad.
    largest = max(sig, key=cv2.contourArea)
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
        # Only accept the warp when it actually spans the detected content — a
        # quad smaller than the content box would slice text off an edge.
        if (out_w >= w * 0.35 and out_h >= h * 0.35
                and out_w >= bw * 0.92 and out_h >= bh * 0.92):
            dst = np.array([[0, 0], [out_w - 1, 0],
                            [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (out_w, out_h))
            print(f"[pipeline] auto-crop+deskew: {w}x{h} -> {out_w}x{out_h}")
            return warped

    # Fallback: axis-aligned crop of the content box, with a small safety pad.
    if bw < w * 0.35 or bh < h * 0.35:
        return image
    pad = max(3, int(min(w, h) * 0.005))
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


def compress_upload(data: bytes, max_dim: int = 2600, target_kb: int = 1024) -> bytes:
    """Shrink an oversized upload so processing stays fast and AI calls don't
    choke on huge payloads. Caps the long side at `max_dim`, then re-encodes as
    JPEG — first lowering quality, then stepping the resolution down further if
    needed — until it fits under `target_kb`. Text stays crisp enough for OCR
    and detection. Images already under the target pass through untouched."""
    if len(data) <= target_kb * 1024:
        return data
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return data  # not a decodable image — leave it for the caller to handle
    h0, w0 = img.shape[:2]

    base = img
    if max(h0, w0) > max_dim:
        scale = max_dim / max(h0, w0)
        base = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    best = None
    target = target_kb * 1024
    for dim_scale in (1.0, 0.85, 0.72, 0.6, 0.5):
        stage = base if dim_scale == 1.0 else cv2.resize(
            base, None, fx=dim_scale, fy=dim_scale, interpolation=cv2.INTER_AREA)
        for q in (90, 84, 78, 72):
            ok, enc = cv2.imencode(".jpg", stage, [cv2.IMWRITE_JPEG_QUALITY, q])
            if not ok:
                continue
            best = (enc, stage.shape[1], stage.shape[0])
            if enc.nbytes <= target:
                out = enc.tobytes()
                print(f"[upload] compressed {len(data)//1024}KB -> {len(out)//1024}KB "
                      f"({w0}x{h0} -> {stage.shape[1]}x{stage.shape[0]})")
                return out
    if best is None:
        return data
    enc, bw, bh = best
    out = enc.tobytes()
    print(f"[upload] compressed {len(data)//1024}KB -> {len(out)//1024}KB "
          f"({w0}x{h0} -> {bw}x{bh}, best effort)")
    return out


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
        # Optional local OCR: read each bubble's own text so translations can
        # never be matched to the wrong bubble. Lazily loaded; no-op if absent.
        self.ocr = None
        self.text_detector = None
        if use_seg:
            try:
                from .ocr import MangaOCR
                self.ocr = MangaOCR()
            except Exception as e:
                print(f"[pipeline] OCR unavailable: {e}")
            try:
                from .text_detect import FreeTextDetector
                self.text_detector = FreeTextDetector()
            except Exception as e:
                print(f"[pipeline] free-text detector unavailable: {e}")

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

        # The GPU model handles white/light bubbles well but misses dark /
        # inverted ones. Supplement with ONLY the CV detector's dark-bubble
        # results — adding all CV results causes false positives on eyes,
        # highlights, and small artwork gaps.
        if self.detector_name != "CV detector":
            try:
                cv_det = BubbleDetector()
                cv_regions = cv_det.detect(image)
                dark_extras = [r for r in cv_regions if r.dark]
                existing_boxes = [r.bbox for r in regions]
                added = 0
                for dr in dark_extras:
                    if not any(_boxes_overlap(list(dr.bbox), list(eb)) for eb in existing_boxes):
                        regions.append(dr)
                        existing_boxes.append(dr.bbox)
                        added += 1
                if added:
                    for idx, r in enumerate(regions):
                        r.id = idx + 1
            except Exception as e:
                print(f"[pipeline] dark bubble supplement failed: {e}")

        bubble_count = len(regions) if regions else 0
        if bubble_count:
            update(1, f"Found {bubble_count} balloons", 22)

        ann_path = ""
        items, masks = [], {}

        if regions:
            annotated = self.detector.create_annotated_image(image, regions)
            ann_path = self._suffix_path(output_path, "annotated")
            cv2.imwrite(ann_path, annotated)

            translations = self._translate_regions(image, regions, annotated, update)

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

        # Free text pass: find narration, dramatic text, labels that aren't
        # inside any detected bubble. Uses CRAFT + manga-ocr + SFX filter.
        free = self._detect_free_text(image, regions, update)
        items.extend(free)

        return items, ann_path, masks

    def _translate_regions(self, image, regions, annotated, update) -> Dict[int, dict]:
        """Translate each detected bubble. Prefers local OCR (reads each
        bubble's OWN text → no cross-bubble mismatch); falls back to the
        vision-LLM number-matching path when OCR isn't available."""
        if self.ocr is not None and self.ocr.ok:
            update(2, "Reading bubbles with manga-ocr...", 30)
            id_to_text = {}
            for r in regions:
                jp = self.ocr.read_region(image, r.bbox, getattr(r, "mask", None))
                if jp:
                    id_to_text[r.id] = jp
            update(2, f"Read {len(id_to_text)} bubbles, translating...", 42)
            if id_to_text:
                try:
                    out = self.translator.translate_texts(id_to_text, self.target_lang)
                    # keep the OCR'd original text for the editor view
                    for rid, jp in id_to_text.items():
                        out.setdefault(rid, {})
                        out[rid].setdefault("original", jp)
                        out[rid]["original"] = out[rid].get("original") or jp
                    update(2, f"Translated {len(out)} bubbles (OCR)", 50)
                    return out
                except Exception as e:
                    print(f"[pipeline] text translation failed, using vision path: {e}")

        update(2, "Translating bubbles...", 32)
        out = self.translator.translate_regions(
            image, annotated, len(regions), self.target_lang
        )
        update(2, f"Translated {len(out)} bubble regions", 50)
        return out

    def _detect_free_text(self, image, bubble_regions, update) -> List[dict]:
        """Second pass: find text not in any bubble (narration, titles, labels).

        Primary path is the vision LLM, which reads vertical Japanese columns,
        large stylized titles, and narration boxes that the CV morphology
        detector can't. Falls back to the CV detector (+ local OCR) when the
        LLM path returns nothing or is unavailable."""
        items = self._free_text_llm(image, bubble_regions, update)
        if items:
            return items
        return self._free_text_cv(image, bubble_regions, update)

    def _free_text_llm(self, image, bubble_regions, update) -> List[dict]:
        """Vision-LLM free-text detection: returns box + original + translation
        in a single call. Reads the vertical / dramatic text the CV pass misses."""
        update(2, "Scanning for free text (narration / titles)...", 52)
        h, w = image.shape[:2]
        bubble_ids = [r.id for r in bubble_regions]
        try:
            dets = self.translator.detect_free_text(image, self.target_lang, bubble_ids)
        except Exception as e:
            print(f"[pipeline] LLM free-text detection failed: {e}")
            return []
        if not dets:
            return []

        from .ocr import _has_japanese
        bubble_boxes = [list(r.bbox) for r in bubble_regions]
        next_id = max((r.id for r in bubble_regions), default=0) + 1
        used: List[list] = []
        items: List[dict] = []

        for det in dets:
            try:
                bx = int(float(det["x_pct"]) / 100.0 * w)
                by = int(float(det["y_pct"]) / 100.0 * h)
                bw = int(float(det["width_pct"]) / 100.0 * w)
                bh = int(float(det["height_pct"]) / 100.0 * h)
            except (KeyError, ValueError, TypeError):
                continue
            bx = max(0, min(bx, w - 1))
            by = max(0, min(by, h - 1))
            bw = min(bw, w - bx)
            bh = min(bh, h - by)
            if bw < 8 or bh < 8:
                continue

            jp = (det.get("original") or "").strip()
            tr = (det.get("translation") or "").strip()
            typ = (det.get("type") or "narration").strip().lower()

            # Only keep regions the model actually read as Japanese — guards
            # against boxes dropped on already-English text or bare artwork.
            if typ in ("sfx", "sound", "onomatopoeia"):
                continue
            if not jp or not _has_japanese(jp) or _is_sfx(jp):
                continue
            if not tr:
                continue

            box = [bx, by, bw, bh]
            if any(_boxes_overlap(box, bb) for bb in bubble_boxes):
                continue
            if any(_boxes_overlap(box, u) for u in used):
                continue
            used.append(box)

            rotation = 0.0
            try:
                rotation = float(det.get("rotation_deg", 0))
            except (ValueError, TypeError):
                pass

            items.append({
                "id": next_id,
                "bbox": box,
                "original": jp,
                "translation": tr,
                "type": typ if typ in ("title", "credit", "narration", "caption") else "narration",
                "in_bubble": False,
                "dark": False,
                "rotation": rotation,
            })
            next_id += 1

        if items:
            update(2, f"Found {len(items)} free text regions", 58)
        return items

    def _free_text_cv(self, image, bubble_regions, update) -> List[dict]:
        """CV-morphology free-text detection + local OCR (fallback path)."""
        if self.text_detector is None or not self.text_detector.ok:
            return []
        if self.ocr is None or not self.ocr.ok:
            return []

        update(2, "Scanning for free text (narration / labels)...", 52)
        bubble_boxes = [tuple(r.bbox) for r in bubble_regions]
        free_boxes = self.text_detector.detect(image, bubble_boxes)
        if not free_boxes:
            return []

        next_id = max((r.id for r in bubble_regions), default=0) + 1
        id_to_text: Dict[int, str] = {}
        box_map: Dict[int, tuple] = {}

        for box in free_boxes:
            jp = self.ocr.read_region(image, box, None)
            if not jp:
                continue
            if _is_sfx(jp):
                continue
            fid = next_id
            next_id += 1
            id_to_text[fid] = jp
            box_map[fid] = box

        if not id_to_text:
            return []

        update(2, f"Translating {len(id_to_text)} free text regions...", 55)
        try:
            translations = self.translator.translate_texts(id_to_text, self.target_lang)
        except Exception as e:
            print(f"[pipeline] free-text translation failed: {e}")
            return []

        items = []
        for fid, jp in id_to_text.items():
            tr = translations.get(fid, {})
            items.append({
                "id": fid,
                "bbox": [int(v) for v in box_map[fid]],
                "original": jp,
                "translation": tr.get("translation", ""),
                "type": tr.get("type", "narration"),
                "in_bubble": False,
                "dark": False,
            })
        if items:
            update(2, f"Found {len(items)} free text regions", 58)
        return items

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
            rotation = 0.0
            try:
                rotation = float(det.get("rotation_deg", 0))
            except (ValueError, TypeError):
                pass
            items.append({
                "id": i + 1,
                "bbox": [x, y, bw, bh],
                "original": det.get("original", ""),
                "translation": det.get("translation", ""),
                "type": det.get("type", "dialogue"),
                "in_bubble": det.get("in_bubble", True),
                "dark": False,
                "rotation": rotation,
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
                    "rotation": it.get("rotation", 0),
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
