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


def _has_arabic(text: str) -> bool:
    """True if the string contains at least one Arabic-script character (covers
    the main block plus the Supplement and Presentation Forms used for
    ligatures). Used to keep real Arabic and drop latin/art hallucinations."""
    for ch in text:
        o = ord(ch)
        if (0x0600 <= o <= 0x06FF      # Arabic
                or 0x0750 <= o <= 0x077F  # Arabic Supplement
                or 0x08A0 <= o <= 0x08FF  # Arabic Extended-A
                or 0xFB50 <= o <= 0xFDFF  # Presentation Forms-A
                or 0xFE70 <= o <= 0xFEFF):  # Presentation Forms-B
            return True
    return False


def _has_korean(text: str) -> bool:
    """True if the string contains at least one Hangul character — syllable
    blocks (가-힣), the individual jamo, and the half-width forms. Used for
    Korean webtoons/manhwa exactly as _has_japanese is used for manga."""
    for ch in text:
        o = ord(ch)
        if (0xAC00 <= o <= 0xD7A3        # Hangul syllables (가 … 힣)
                or 0x1100 <= o <= 0x11FF  # Jamo
                or 0x3130 <= o <= 0x318F  # Compatibility jamo
                or 0xA960 <= o <= 0xA97F  # Jamo Extended-A
                or 0xD7B0 <= o <= 0xD7FF  # Jamo Extended-B
                or 0xFFA0 <= o <= 0xFFDC):  # half-width jamo
            return True
    return False


def _has_source_text(text: str, source_lang: str = "Japanese") -> bool:
    """Whether `text` looks like real source-language text for the chosen
    source. Japanese, Korean and Arabic get script-specific checks; 'auto' (or
    anything else) just requires at least one letter so the vision-LLM's
    reading isn't thrown away over a non-Japanese page."""
    sl = (source_lang or "Japanese").strip().lower()
    if sl in ("japanese", "ja", "jp"):
        return _has_japanese(text)
    if sl in ("korean", "ko", "kr", "hangul"):
        # Korean pages carry occasional Hanja and stylised Japanese-style
        # SFX, so accept either script rather than rejecting a good read.
        return _has_korean(text) or _has_japanese(text)
    if sl in ("arabic", "ar"):
        return _has_arabic(text)
    return any(ch.isalpha() for ch in text)


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
            from .gpu_throttle import limit as _gpu_limit
            with _gpu_limit():
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
