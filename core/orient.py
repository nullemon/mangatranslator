"""Is this page upside down?

Scans arrive rotated 180° often enough to be a nuisance — a batch gets fed
through a scanner the wrong way round, or a phone photo of a tankoubon comes
out inverted. Every stage downstream then fails quietly: the balloons are
found, the OCR returns nothing usable, and the page comes back with garbage
in it.

A person spots it instantly, and so can the app, because there is a test that
settles it: read the text both ways up and see which way produces real
language. Upside-down Japanese is not slightly worse to an OCR model — it is
noise.

The balloons are found with the plain OpenCV detector (no model, no download),
and only the biggest few are read, so the whole check is a fraction of a
second per page and costs nothing in API calls.
"""
import cv2
import numpy as np
from typing import Dict, List, Optional

from .detector import BubbleDetector
from .ocr import _has_source_text


def _crops(image: np.ndarray, want: int = 4) -> List[np.ndarray]:
    """The most promising text areas on the page, biggest first.

    Big balloons are picked deliberately: they hold the most characters, so a
    reading either succeeds convincingly or fails convincingly. A tiny one can
    return one plausible-looking character either way up and settle nothing.
    """
    h, w = image.shape[:2]
    try:
        regions = BubbleDetector().detect(image)
    except Exception as e:
        print(f"[orient] balloon detection failed: {e}")
        return []
    out = []
    for r in sorted(regions, key=lambda r: -r.area):
        x, y, bw, bh = [int(v) for v in r.bbox]
        if bw < 40 or bh < 40:
            continue
        m = 6
        crop = image[max(0, y - m):min(h, y + bh + m),
                     max(0, x - m):min(w, x + bw + m)]
        if crop.size:
            out.append(crop)
        if len(out) >= want:
            break
    return out


def _score(ocr, crop: np.ndarray, source_lang: str) -> int:
    """How much real source-language text does this crop yield? Characters,
    not a yes/no — a stray character read out of a pattern should not weigh
    the same as a full line of dialogue."""
    try:
        text = ocr.read(crop) or ""
    except Exception:
        return 0
    if not text or not _has_source_text(text, source_lang):
        return 0
    return len(text.strip())


def check(image: np.ndarray, source_lang: str = "Japanese",
          ocr=None) -> Dict[str, object]:
    """Decide whether `image` is upside down.

    Returns {"upside_down", "sure", "why"}. `sure` is False whenever the page
    gives no clear answer — a page with no readable text, or one that reads
    about as badly both ways — and in that case NOTHING should be flipped
    automatically. Turning a page the user scanned correctly is a worse
    failure than leaving an inverted one for them to fix by hand, so this
    errs firmly towards leaving it alone.
    """
    verdict = {"upside_down": False, "sure": False, "why": "", "up": 0,
               "down": 0}

    if ocr is None:
        from .ocr import MangaOCR
        ocr = MangaOCR()
    if not getattr(ocr, "ok", False):
        verdict["why"] = ("the offline reader isn't available, so the page "
                          "can't be checked automatically")
        return verdict

    # manga-ocr reads Japanese. On a Korean or Chinese page it returns nothing
    # either way up, which would look exactly like "no text found" — so say so
    # rather than pretending to a verdict.
    if (source_lang or "").strip().lower() not in ("japanese", "auto", ""):
        verdict["why"] = (f"automatic checking only works on Japanese pages "
                          f"(this one is set to {source_lang})")
        return verdict

    crops = _crops(image)
    if not crops:
        verdict["why"] = "no speech balloons found to read"
        return verdict

    up = down = 0
    for crop in crops:
        up += _score(ocr, crop, source_lang)
        down += _score(ocr, cv2.rotate(crop, cv2.ROTATE_180), source_lang)
    verdict["up"], verdict["down"] = up, down

    if max(up, down) < 3:
        verdict["why"] = "the balloons didn't read either way up"
        return verdict
    # A clear margin, not a bare majority. Rotated text occasionally reads as
    # a character or two of something plausible, and a page must not be turned
    # over on that.
    if down > up * 1.6 and down - up >= 4:
        verdict.update(upside_down=True, sure=True,
                       why=f"reads as {down} characters upside down "
                           f"against {up} the right way up")
    elif up >= down:
        verdict.update(upside_down=False, sure=True,
                       why=f"reads fine as it is ({up} characters)")
    else:
        verdict["why"] = (f"unclear — {up} characters one way, {down} the "
                          f"other; left alone")
    return verdict


def check_file(path: str, source_lang: str = "Japanese",
               ocr=None) -> Dict[str, object]:
    img = cv2.imread(path)
    if img is None:
        return {"upside_down": False, "sure": False,
                "why": "the image could not be read", "up": 0, "down": 0}
    # Reading is done on a working copy: a huge scan is slow to OCR and gains
    # nothing, the text is legible either way.
    h, w = img.shape[:2]
    if max(h, w) > 2000:
        s = 2000.0 / max(h, w)
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                         interpolation=cv2.INTER_AREA)
    return check(img, source_lang, ocr)
