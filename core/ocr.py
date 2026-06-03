"""Local Japanese manga OCR (kha-white/manga-ocr) — reads the text *inside*
each detected bubble so a bubble's translation can never end up in another
bubble. Runs on the GPU when torch+CUDA are present. Entirely optional: if
manga-ocr isn't installed, `ok` is False and the pipeline keeps using the
vision-LLM path."""
import cv2
import numpy as np
from PIL import Image
from typing import Optional


def _has_japanese(text: str) -> bool:
    """True if the string contains at least one hiragana, katakana, or kanji.
    Filters OCR hallucinations on non-text regions (eyes, mouths, art), which
    come back as latin garbage or bare punctuation rather than real Japanese."""
    for ch in text:
        o = ord(ch)
        if (0x3040 <= o <= 0x30FF      # hiragana + katakana
                or 0x4E00 <= o <= 0x9FFF  # CJK unified ideographs (kanji)
                or 0xFF66 <= o <= 0xFF9D):  # half-width katakana
            return True
    return False


class MangaOCR:
    _shared = None  # process-wide cache of the loaded model

    def __init__(self):
        self._mocr = None
        self._failed = False

    @property
    def ok(self) -> bool:
        """True if the model is loadable. Lazily loads on first check."""
        if self._failed:
            return False
        if self._mocr is not None:
            return True
        return self._load()

    def _load(self) -> bool:
        if MangaOCR._shared is not None:
            self._mocr = MangaOCR._shared
            return True
        try:
            from manga_ocr import MangaOcr
            print("[ocr] loading manga-ocr model (first run downloads ~400MB)...")
            self._mocr = MangaOcr()
            MangaOCR._shared = self._mocr
            print("[ocr] manga-ocr ready")
            return True
        except Exception as e:
            print(f"[ocr] manga-ocr unavailable ({e}); using vision-LLM reading")
            self._failed = True
            return False

    def read(self, bgr: np.ndarray) -> str:
        """OCR a BGR image crop → Japanese text (empty string on failure).
        Returns "" when the result has no real Japanese characters, so text
        is never placed into a non-bubble region (eye, mouth, stray art)."""
        if not self.ok or bgr is None or bgr.size == 0:
            return ""
        try:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            text = (self._mocr(Image.fromarray(rgb)) or "").strip()
            if not _has_japanese(text):
                return ""
            return text
        except Exception as e:
            print(f"[ocr] read failed: {e}")
            return ""

    def read_region(self, image: np.ndarray, bbox, mask: Optional[np.ndarray] = None) -> str:
        """Crop `bbox` from `image` (optionally limited to `mask`) and OCR it.
        Adds a little white margin so edge glyphs read cleanly."""
        H, W = image.shape[:2]
        x, y, w, h = [int(v) for v in bbox]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return ""
        crop = image[y0:y1, x0:x1].copy()
        if mask is not None:
            m = mask[y0:y1, x0:x1]
            if m.shape[:2] == crop.shape[:2]:
                white = np.full_like(crop, 255)
                crop = np.where(m[..., None] > 0, crop, white)
        crop = cv2.copyMakeBorder(crop, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        return self.read(crop)
