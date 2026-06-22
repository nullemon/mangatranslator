import cv2
import numpy as np
import os
import random
import re
from typing import Callable, Optional, Dict, Any, List

from .detector import BubbleDetector, TextRegion
from .translator import make_translator
from .compositor import Compositor


_WATERMARK_RE = re.compile(
    r"(?:https?://|www\.)\S+"                       # an explicit URL
    r"|\b[a-z0-9][a-z0-9\-]{1,30}\."                # a domain label + dot +
    r"(?:net|com|org|io|co|info|me|tv|xyz|biz|online|site|fan|fans|sh|to|cc|"
    r"ru|in|us|uk|live|app|art|gg|club|world|space|web|moe|scan|scans)\b",
    re.IGNORECASE,
)


def _is_watermark(text: str) -> bool:
    """Looks like a scanlation-site stamp / URL (eshadow.net, www.x.com, …)
    rather than story text — those get erased, not translated. Kept short so a
    line of dialogue that merely mentions a word isn't mistaken for one."""
    t = (text or "").strip()
    if not t or len(t) > 60:
        return False
    return bool(_WATERMARK_RE.search(t))


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


def _texts_match(a: str, b: str) -> bool:
    """Loose match: do two readings of the same region share most of their
    Japanese characters? Robust to OCR quirks, furigana and ordering — used
    to spot the SAME text found twice (dedupe) and a vision-LLM box that
    doesn't contain the text it claims (misplacement)."""
    def chars(s):
        return {c for c in s
                if "ぁ" <= c <= "ん" or "ァ" <= c <= "ヶ"
                or "一" <= c <= "鿿" or c == "ー"}
    aa, bb = chars(a), chars(b)
    if not aa or not bb:
        return False
    inter = len(aa & bb)
    return inter / max(min(len(aa), len(bb)), 1) >= 0.5


def _merge_column_boxes(boxes):
    """The block detector splits giant vertical lettering (one huge glyph per
    box) — merge boxes that line up into a single column/run so OCR reads the
    whole phrase instead of one syllable at a time."""
    boxes = [list(map(int, b)) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            if boxes[i] is None:
                continue
            for j in range(i + 1, len(boxes)):
                if boxes[j] is None:
                    continue
                a, b = boxes[i], boxes[j]
                ox = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
                oy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
                # vertical column: strong horizontal overlap, small v-gap
                col = ox > 0.5 * min(a[2], b[2]) and -oy < 0.8 * min(a[2], b[2])
                # horizontal run: strong vertical overlap, small h-gap
                run = oy > 0.5 * min(a[3], b[3]) and -ox < 0.8 * min(a[3], b[3])
                if not (col or run):
                    continue
                x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                x1 = max(a[0] + a[2], b[0] + b[2])
                y1 = max(a[1] + a[3], b[1] + b[3])
                boxes[i] = [x0, y0, x1 - x0, y1 - y0]
                boxes[j] = None
                changed = True
    return [tuple(b) for b in boxes if b is not None]


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


def boost_for_detection(image: np.ndarray) -> np.ndarray:
    """Local-contrast boost (CLAHE on L) used ONLY to help the detectors read a
    faint, washed-out raw — the erase/clean is still applied to the original
    pixels, so this never alters the output, just what the models can see."""
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    except Exception:
        return image


def _fill_largest(binmask: np.ndarray, h: int, w: int, page_area: float,
                  close_div: int = 30):
    """Clean a candidate page mask, keep its LARGEST blob and fill its holes
    (so dark panels inside the page count as page). Returns (filled_mask, bbox,
    area) following the page's true — possibly tilted/irregular — outline, or
    None if it isn't a believable page (too small, or fills the whole frame)."""
    binmask = cv2.morphologyEx(binmask, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    k = max(9, (min(h, w) // close_div) | 1)
    binmask = cv2.morphologyEx(binmask, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)), iterations=2)
    contours, _ = cv2.findContours(binmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sig = [c for c in contours if cv2.contourArea(c) > page_area * 0.01]
    if not sig:
        return None
    big = max(sig, key=cv2.contourArea)
    filled = np.zeros((h, w), np.uint8)
    cv2.drawContours(filled, [big], -1, 255, cv2.FILLED)
    area = int(cv2.countNonZero(filled))
    if area < page_area * 0.20 or area > page_area * 0.985:
        return None
    return filled, cv2.boundingRect(big), area


def _isolate_grabcut(image: np.ndarray, h: int, w: int, page_area: float):
    """Segment the page from the background with GrabCut: seed the centre as
    definite foreground (the page) and a thin outer ring as definite background
    (the surface), and let GrabCut learn each one's colour. Handles a coloured /
    shadowed / gradient carpet that fixed thresholds can't. Runs on a downscaled
    copy for speed; returns (filled_mask, bbox, area) or None."""
    longest = max(h, w)
    scale = 768.0 / longest if longest > 768 else 1.0
    sh, sw = max(1, int(h * scale)), max(1, int(w * scale))
    small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA) if scale < 1 else image.copy()

    gc = np.full((sh, sw), cv2.GC_PR_FGD, np.uint8)
    ring = max(2, int(min(sh, sw) * 0.02))      # outer ~2% = definite background
    gc[:ring, :] = cv2.GC_BGD; gc[-ring:, :] = cv2.GC_BGD
    gc[:, :ring] = cv2.GC_BGD; gc[:, -ring:] = cv2.GC_BGD
    gc[int(sh * 0.30):int(sh * 0.70),
       int(sw * 0.30):int(sw * 0.70)] = cv2.GC_FGD   # centre = definite page
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(small, gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    fg = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    if scale < 1:
        fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST)
    return _fill_largest(fg, h, w, page_area)


def isolate_page(image: np.ndarray) -> np.ndarray:
    """Beta: remove the background around a photographed page (table, floor,
    carpet, the adjacent page/spine). Finds the page outline, paints everything
    OUTSIDE it white, and crops to it.

    GrabCut is the primary segmenter — it learns the page's and the surface's
    colours, so it carves out a tilted/irregular page from a coloured or
    shadowed carpet. If GrabCut isn't confident, three simple cues are tried as
    a fallback (colour distance from the border, brightness, grey tone). Returns
    the image unchanged only if none find a confident page (never wrecks a clean
    scan)."""
    h, w = image.shape[:2]
    page_area = h * w

    res, method = None, None
    try:
        res = _isolate_grabcut(image, h, w, page_area)
        method = "grabcut"
    except Exception as e:
        print(f"[isolate] grabcut failed ({e}); trying heuristics")

    if res is None:
        blur = cv2.GaussianBlur(image, (7, 7), 0)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        border_px = np.concatenate([blur[0, :, :], blur[-1, :, :],
                                    blur[:, 0, :], blur[:, -1, :]]).reshape(-1, 3)
        bg_col = np.median(border_px, axis=0)
        dist = np.linalg.norm(blur.astype(np.float32) - bg_col, axis=2)
        _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bg = int(round(float(np.median(gray[[0, -1], :]))))
        candidates = [
            ("colour", (dist > 30).astype(np.uint8) * 255),
            ("bright", bright),
            ("tone", (cv2.absdiff(gray, np.full_like(gray, bg)) > 24).astype(np.uint8) * 255),
        ]
        for name, mask in candidates:
            r = _fill_largest(mask, h, w, page_area)
            if r is not None and (res is None or r[2] > res[2]):
                res, method = r, name

    if res is None:
        print("[isolate] no confident page found — leaving page unchanged")
        return image
    mask, (bx, by, bw, bh), area = res
    print(f"[isolate] page via '{method}' — {bw}x{bh} ({area/page_area:.0%} of frame)")

    # feather the edge a touch so the cut isn't a hard jagged line
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    out = image.copy()
    out[mask == 0] = (255, 255, 255)          # background → white
    return out[by:by + bh, bx:bx + bw]        # crop to the page


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

    # How rotated is the page overall? minAreaRect's angle is robust against a
    # footer/credit nub poking out of one corner (which would otherwise fool the
    # 4-corner fit into thinking the page is skewed). A clean digital scan reads
    # ~0°, so we never warp/tilt it.
    rang = cv2.minAreaRect(largest)[2]
    skew = min(abs(rang), abs(abs(rang) - 90))

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
        # quad smaller than the content box would slice text off an edge — AND
        # the page is meaningfully skewed (a real photo, not a flat scan).
        if (skew >= 3.0
                and out_w >= w * 0.35 and out_h >= h * 0.35
                and out_w >= bw * 0.92 and out_h >= bh * 0.92):
            dst = np.array([[0, 0], [out_w - 1, 0],
                            [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (out_w, out_h))
            print(f"[pipeline] auto-crop+deskew ({skew:.1f}°): {w}x{h} -> {out_w}x{out_h}")
            return warped

    # Fallback: axis-aligned crop of the content box — but ONLY when the margin
    # around it is a clean, uniform background (a photographed page sitting on a
    # desk), never the page's own white gutter. Without this guard a flat digital
    # raw gets its margins (and the footer/credit strip) shaved off.
    if bw < w * 0.35 or bh < h * 0.35:
        return image
    if bw >= w * 0.9 and bh >= h * 0.9:
        return image  # content already fills the frame; nothing to crop
    if float(np.std(border)) > 18.0:
        return image  # busy/inked border = digital page edge, not a backdrop
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


def scan_finish(image: np.ndarray) -> np.ndarray:
    """Final 'clean scan' pass (TCB-release look): melt scanner/photo grain,
    snap the paper to pure white and the ink to solid black — while leaving
    every midtone (screentones, gradients, pencil shading) untouched.

    Unlike a global levels stretch, this uses knee curves anchored to the
    page's own paper and ink histogram peaks, so only the tails move:
    a grainy 200-gray paper becomes white, a 40-gray ink becomes black,
    and a 120-gray screentone stays exactly 120."""
    den = cv2.fastNlMeansDenoisingColored(image, None, 7, 7, 7, 21)
    lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]

    hist = cv2.calcHist([L], [0], None, [256], [0, 256]).ravel()
    # Paper tone = brightest strong peak; ink tone = darkest strong peak.
    paper = int(hist[140:].argmax()) + 140 if hist[140:].sum() > 0 else 235
    ink = int(hist[:100].argmax()) if hist[:100].sum() > 0 else 12

    wp = max(180, paper - 6)        # everything at/above paper -> pure white
    kw = wp - 34                    # ramp starts just below the paper tone
    bp = min(ink + 14, 64)          # everything at/below ink -> solid black
    kb = bp + 30

    lut = np.arange(256, dtype=np.float32)
    lut[:bp + 1] = 0
    if kb > bp:
        lut[bp:kb + 1] = np.linspace(0, kb, kb - bp + 1)
    if wp > kw:
        lut[kw:wp + 1] = np.linspace(kw, 255, wp - kw + 1)
    lut[wp:] = 255

    lab[:, :, 0] = cv2.LUT(L, lut.astype(np.uint8))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preserve_dark_regions(enhanced: np.ndarray, original: np.ndarray) -> np.ndarray:
    """A generative "scan" (Grok/Gemini) bleaches solid-black art — black
    panels, gutters, white-on-black title text — to white, inverting the page.
    Claw those back: wherever the ORIGINAL has a LARGE solid-dark region, use
    the locally clean-scanned original there instead of the AI output, so black
    manga areas (and any white lettering inside them) stay exactly as drawn.

    Scoped to big blobs only (panels/gutters/title slabs), never individual ink
    strokes, so the AI's clean lineart everywhere else is untouched."""
    if enhanced.shape[:2] != original.shape[:2]:
        original = cv2.resize(original, (enhanced.shape[1], enhanced.shape[0]),
                              interpolation=cv2.INTER_AREA)
    clean = scan_finish(original)          # black stays black, white text stays
    g = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    h, w = g.shape[:2]
    dark = (g < 90).astype(np.uint8) * 255
    # Close over white-on-black lettering so a titled black slab counts as ONE
    # solid region (text included), then drop thin/ink-sized bits.
    k = max(9, (min(h, w) // 80) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, ker)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, ker)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    keep = np.zeros((h, w), np.uint8)
    min_area = 0.004 * h * w               # ≥0.4% of the page = a real black area
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    if cv2.countNonZero(keep) == 0:
        return enhanced
    keep = cv2.GaussianBlur(keep, (0, 0), 2)        # feather the swap boundary
    a = (keep.astype(np.float32) / 255.0)[..., None]
    out = clean.astype(np.float32) * a + enhanced.astype(np.float32) * (1 - a)
    return out.clip(0, 255).astype(np.uint8)


def compress_upload(data: bytes, max_dim: int = 3200, target_kb: int = 3072) -> bytes:
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


def _overlap_frac(a, b) -> float:
    """Intersection area as a fraction of the smaller box — how strongly two
    boxes coincide (used to match an LLM translation to a precise GPU bubble)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    xi, yi = max(ax, bx), max(ay, by)
    xf, yf = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if xi >= xf or yi >= yf:
        return 0.0
    inter = (xf - xi) * (yf - yi)
    smaller = max(min(aw * ah, bw * bh), 1)
    return inter / smaller


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


def probe_components() -> Dict[str, Any]:
    """Fast availability snapshot for /api/health — checks imports, GPU and
    weights WITHOUT loading heavy models, so it's cheap to call. Tells you at a
    glance whether the full 'AI finds all + GPU fixes all' stack is in place."""
    out: Dict[str, Any] = {}

    def _imp(mod):
        import importlib
        try:
            importlib.import_module(mod)
            return True
        except Exception:
            return False

    # GPU / CUDA
    try:
        import torch
        out["torch"] = torch.__version__
        out["cuda"] = torch.cuda.is_available()
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        out["torch"] = None
        out["cuda"] = False
        out["gpu"] = None
    try:
        import onnxruntime as ort
        out["onnxruntime_cuda"] = "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        out["onnxruntime_cuda"] = False

    out["balloon_seg_yolo"] = _imp("ultralytics")
    out["manga_ocr"] = _imp("manga_ocr")
    out["lama_inpaint"] = _imp("simple_lama_inpainting")
    out["free_text_craft"] = _imp("craft_text_detector")
    out["upscale_spandrel"] = _imp("spandrel")

    # Weights on disk
    out["weights"] = {
        "comic_text_detector": os.path.exists("models/comictextdetector.pt.onnx"),
        "real_esrgan": os.path.exists("models/RealESRGAN_x4plus_anime_6B.pth"),
        "mangajanai": bool(__import__("glob").glob("models/mangajanai/*.pth")),
    }

    # RTL Arabic typesetting readiness
    try:
        from PIL import features as _pf
        out["raqm_rtl_shaping"] = bool(_pf.check("raqm"))
    except Exception:
        out["raqm_rtl_shaping"] = False
    out["arabic_reshaper_fallback"] = _imp("arabic_reshaper") and _imp("bidi")
    try:
        from .renderer import TextRenderer
        r = TextRenderer()
        out["arabic_font"] = bool(
            r.font_path and r._effective_font_path("العربية") is not None)
    except Exception:
        out["arabic_font"] = False

    out["ready_full_stack"] = all([
        out["cuda"], out["balloon_seg_yolo"], out["manga_ocr"],
        out["lama_inpaint"], out["free_text_craft"],
    ])
    return out


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
        style_prompt: str = "",
        text_case: str = "upper",
        finish: str = "clean",
        upscale: Optional[bool] = None,
        source_lang: str = "Japanese",
        translate_sfx: bool = False,
        max_quality: bool = False,
        remove_watermark: bool = True,
        replace_watermark: bool = False,
        watermark_text: str = "",
        clean_only: bool = False,
        isolate_page: bool = False,
        credit: str = "",
    ):
        self.finish = finish
        self.source_lang = source_lang or "Japanese"
        self.translate_sfx = bool(translate_sfx)
        # Clean-only: just erase ALL text (no translation), for a usable raw.
        self.clean_only = bool(clean_only)
        # Isolate page (beta): white-out + crop the background around a photo.
        self.isolate_page = bool(isolate_page)
        # Credit / TL name dropped in the margin (movable per page in the editor).
        self.credit = (credit or "").strip()
        # Maximum quality: process at full resolution (no working-size downscale)
        # and lean on the whole GPU stack. OFF by default — opt-in per run.
        self.max_quality = bool(max_quality)
        # Site watermarks (eshadow.net, …): erase by default; optionally drop the
        # user's own watermark in their place.
        self.remove_watermark = bool(remove_watermark)
        self.replace_watermark = bool(replace_watermark)
        self.watermark_text = (watermark_text or "").strip()
        # HD upscale: explicit per-run choice wins; otherwise fall back to the
        # MANGA_UPSCALE env default. OFF unless asked for.
        self.upscale_on = (upscale if upscale is not None
                           else os.environ.get("MANGA_UPSCALE", "0") == "1")
        self.detector, self.detector_name = make_detector(use_seg)
        self.translator = make_translator(
            provider, api_key, model, style_prompt,
            source_lang=self.source_lang, translate_sfx=self.translate_sfx)
        self.compositor = Compositor(font_path, uppercase=(text_case != "keep"),
                                     translate_sfx=self.translate_sfx,
                                     replace_watermark=self.replace_watermark,
                                     watermark_text=self.watermark_text)
        self.target_lang = target_lang
        self.use_smart_detection = use_smart_detection
        self.last_masks: Dict[int, np.ndarray] = {}
        # Optional local OCR: read each bubble's own text so translations can
        # never be matched to the wrong bubble. Lazily loaded; no-op if absent.
        self.ocr = None
        self.text_detector = None
        self.text_seg = None
        self.upscaler = None
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
            try:
                from .text_seg import TextSegmenter
                self.text_seg = TextSegmenter()
            except Exception as e:
                print(f"[pipeline] text segmenter unavailable: {e}")
            try:
                from .upscale import Upscaler
                self.upscaler = Upscaler()
            except Exception as e:
                print(f"[pipeline] upscaler unavailable: {e}")

        self.components = self._component_status()
        self._log_component_banner()

    def _credit_item(self, w: int, h: int, next_id: int) -> dict:
        """A TL/scanlation credit drawn as an overlay (no erase), movable/editable
        per page. Position is RANDOMIZED per page (biased to the edges/gutters,
        away from the centre where faces/art sit) so thieves can't batch-crop it
        off a fixed corner. The chosen spot is saved with the page, so re-render
        is stable and you can still drag it if it lands awkwardly."""
        cw, ch = int(w * 0.30), max(int(h * 0.033), 18)
        mx, my = int(w * 0.02), int(h * 0.02)            # edge margin
        # Anchor to a random edge band, then jitter ALONG that edge so it never
        # repeats from page to page. Centre is deliberately excluded.
        edge = random.choice(("top", "bottom", "left", "right"))
        if edge in ("top", "bottom"):
            cy = my if edge == "top" else int(h - ch - my)
            cx = random.randint(mx, max(mx, w - cw - mx))
        else:  # left / right: ride vertically up the gutter
            cx = mx if edge == "left" else int(w - cw - mx)
            cy = random.randint(my, max(my, h - ch - my))
        return {"id": next_id, "bbox": [cx, cy, cw, ch], "original": "",
                "translation": self.credit, "type": "credit", "in_bubble": False,
                "credit": True, "dark": False, "rotation": 0}

    def _component_status(self) -> Dict[str, Any]:
        """A snapshot of which detection / cleanup / upscale stages actually
        loaded, so the logs (and /api/health) show exactly what will run."""
        def ok(obj):
            return bool(obj is not None and getattr(obj, "ok", True))
        try:
            from PIL import features as _pf
            raqm = bool(_pf.check("raqm"))
        except Exception:
            raqm = False
        return {
            "detector": self.detector_name,
            "gpu_balloon_seg": "GPU" in self.detector_name,
            "manga_ocr": ok(self.ocr),
            "free_text_detector_craft": ok(self.text_detector),
            "text_pixel_seg": ok(self.text_seg),
            "upscaler": ok(self.upscaler),
            "lama_inpaint": getattr(self.compositor, "lama", None) is not None,
            "translator": type(self.translator).__name__,
            "raqm_rtl_shaping": raqm,
        }

    def _log_component_banner(self):
        c = self.components
        def mark(v):
            return "GPU/ON " if v else "off    "
        print("[pipeline] ===== component stack =====")
        print(f"[pipeline]   balloon detect : {c['detector']}")
        print(f"[pipeline]   manga-ocr      : {mark(c['manga_ocr'])}")
        print(f"[pipeline]   free-text CRAFT: {mark(c['free_text_detector_craft'])}")
        print(f"[pipeline]   text-pixel seg : {mark(c['text_pixel_seg'])}")
        print(f"[pipeline]   LaMa inpaint   : {mark(c['lama_inpaint'])}")
        print(f"[pipeline]   upscaler       : {mark(c['upscaler'])}")
        print(f"[pipeline]   RTL shaping    : {mark(c['raqm_rtl_shaping'])} (raqm)")
        print(f"[pipeline]   translator     : {c['translator']} | "
              f"src={self.source_lang} -> {self.target_lang} | "
              f"max_quality={self.max_quality} | sfx={self.translate_sfx}")
        missing = [k for k in ("manga_ocr", "free_text_detector_craft",
                               "text_pixel_seg", "lama_inpaint")
                   if not c[k]]
        if missing:
            print(f"[pipeline]   ⚠ inactive (run ./setup_gpu.sh): {', '.join(missing)}")
        print("[pipeline] ============================")

    def process(
        self,
        image_path: str,
        output_path: str,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
        render_base_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Translate a page. Normally detection AND typesetting happen on the
        same image. For the Scan → Translate flow, pass `render_base_path` (the
        AI-scanned page): text is then DETECTED/OCR'd on the crisp RAW
        (`image_path`) but ERASED and typeset onto the clean scanned base — so
        the generative scan can't corrupt detection. Both must be the same page;
        the base is resized to the raw's working size to keep boxes aligned."""
        def update(step: int, msg: str, pct: int):
            if progress_cb:
                progress_cb({"step": step, "message": msg, "progress": pct})

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        h, w = image.shape[:2]
        # Max-quality: keep the page at full resolution so detection, OCR and
        # erasure work on every pixel (cleaner masks, sharper lettering). The
        # default caps the working size to keep CPU-only runs responsive.
        max_dim = 10000 if self.max_quality else 4000
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if self.max_quality:
            print(f"[pipeline] MAX QUALITY run — full-resolution {image.shape[1]}x{image.shape[0]}")

        # Detect/OCR on `image`; erase + typeset on `base`. They're the same
        # except in Scan → Translate, where base is the AI-scanned page.
        base = None
        if render_base_path:
            base = cv2.imread(render_base_path)
            if base is None:
                print(f"[pipeline] render base missing ({render_base_path}); using raw")

        update(0, "Preprocessing image...", 2)
        if base is None:
            # Standard path: detection and rendering share one image.
            image = auto_crop_page(image)
            if self.isolate_page:
                update(0, "Isolating page (removing background)...", 3)
                try:
                    image = isolate_page(image)
                except Exception as e:
                    print(f"[pipeline] isolate-page failed, continuing: {e}")

            # HD upscale: OFF by default, opt-in per run (UI toggle / MANGA_UPSCALE).
            # With MangaJaNai installed this is FAITHFUL — sharpens and de-artifacts
            # without redrawing, so the art is preserved; Real-ESRGAN is the more
            # aggressive fallback. Only runs on pages that aren't already large.
            if (self.upscaler is not None
                    and self.upscale_on
                    and max(image.shape[:2]) < 2000
                    and self.upscaler.ok):
                if self.max_quality:
                    print("[pipeline] NOTE: HD Upscale + Maximum Quality together is "
                          "very heavy — on an 8GB laptop GPU this can take many "
                          "minutes. For fast translation, leave HD Upscale OFF.")
                update(0, "Upscaling page to HD (MangaJaNai)...", 4)
                try:
                    image = self.upscaler.upscale(image)
                    print(f"[pipeline] upscaled to {image.shape[1]}x{image.shape[0]}")
                except Exception as e:
                    print(f"[pipeline] upscale failed, continuing as-is: {e}")
            base = image
        else:
            # Scan → Translate: keep the raw at its working size for detection and
            # match the scanned base to it so erase masks / boxes line up.
            if base.shape[:2] != image.shape[:2]:
                base = cv2.resize(base, (image.shape[1], image.shape[0]),
                                  interpolation=cv2.INTER_AREA)
            print(f"[pipeline] Scan→Translate: detecting on RAW, typesetting on scanned base")

        base_path = self._base_path(output_path)
        cv2.imwrite(base_path, base)

        # Clean-only: remove ALL text and stop — no detection-LLM, no
        # translation (so no API key needed). Page Finish still decides raw
        # (untouched) vs scan-like (clean).
        if self.clean_only:
            update(3, "Cleaning all text from the page...", 55)
            # Detect on a contrast-boosted copy (helps faint/low-contrast raws);
            # erase from the original. Also wipe detected speech-bubble interiors
            # so in-bubble text the stroke detector misses is cleared too.
            det = boost_for_detection(image)
            bubble_masks = []
            try:
                for r in self.detector.detect(det):
                    m = getattr(r, "mask", None)
                    if m is not None:
                        bubble_masks.append(m)
            except Exception as e:
                print(f"[pipeline] clean: bubble detect failed: {e}")
            cleaned = self.compositor.clean(image, bubble_masks, det_image=det)
            # Re-render (the Erase tool) must build on the CLEANED page, else
            # erasing one leftover rebuilds from the original and all the removed
            # text comes back. Save cleaned (pre-finish) to a SEPARATE base — and
            # leave base_path as the real original-with-text so the 'Original'
            # view still shows the page you uploaded.
            clean_base = self._suffix_path(output_path, "cleanbase")
            cv2.imwrite(clean_base, cleaned)
            # Stamp the TL/credit name on the cleaned page (if requested). Keep it
            # OUT of clean_base so the Erase tool re-renders from a credit-free
            # plate; the credit item rides along in the result so it stays
            # movable/editable and gets re-drawn at its (possibly moved) spot.
            items = []
            if self.credit:
                items.append(self._credit_item(image.shape[1], image.shape[0], 1))
            out = self.compositor.compose(cleaned, items, {}) if items else cleaned
            if self.finish in ("clean", "api"):
                update(4, "Applying clean-scan finish...", 90)
                out = scan_finish(out)
            cv2.imwrite(output_path, out)
            update(5, "Cleaned!", 100)
            res = self._result(output_path, base_path, items, "")
            res["clean_base_path"] = clean_base
            return res

        self.last_masks = {}
        if self.use_smart_detection:
            items, ann_path, masks = self._smart_detect(image, output_path, update)
        else:
            items, ann_path, masks = self._standard_detect(image, output_path, update)
        self.last_masks = masks

        if self.credit:
            nid = max([it["id"] for it in items] + [0]) + 1
            items.append(self._credit_item(image.shape[1], image.shape[0], nid))

        if not items:
            out = scan_finish(base) if self.finish in ("clean", "api") else base
            cv2.imwrite(output_path, out)
            update(5, "No text regions found.", 100)
            return self._result(output_path, base_path, [], ann_path)

        update(3, "Erasing original text...", 60)
        update(4, "Fitting translations into balloons...", 80)
        result = self.compositor.compose(base, items, masks)
        if self.finish in ("clean", "api"):
            update(4, "Applying clean-scan finish...", 92)
            result = scan_finish(result)
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
        """Translate each detected bubble. For Japanese, prefer local OCR (reads
        each bubble's OWN text → no cross-bubble mismatch). For any OTHER source
        language the local manga-ocr can't read it (it's Japanese-only), so go
        straight to the vision model, which reads Arabic/Korean/etc. directly."""
        src = (self.source_lang or "Japanese").strip().lower()
        ocr_usable = (self.ocr is not None and self.ocr.ok
                      and src in ("japanese", "ja", "jp"))
        if not ocr_usable and self.ocr is not None and self.ocr.ok:
            print(f"[pipeline] source={self.source_lang}: skipping Japanese "
                  f"manga-ocr, reading bubbles with the vision model")
        if ocr_usable:
            from .ocr import _has_japanese
            update(2, "Reading bubbles with manga-ocr...", 30)
            id_to_text = {}
            for r in regions:
                jp = self.ocr.read_region(image, r.bbox, getattr(r, "mask", None))
                # Keep only genuinely Japanese reads. If the page is actually
                # another language, manga-ocr returns latin/garbage — drop it so
                # the whole page falls through to the vision model below.
                if jp and _has_japanese(jp):
                    id_to_text[r.id] = jp
            update(2, f"Read {len(id_to_text)} bubbles, translating...", 42)
            if id_to_text:
                try:
                    out = self.translator.translate_texts(id_to_text, self.target_lang, image=image)
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
        LLM path returns nothing or is unavailable. Finally the manga-trained
        text-block detector supplements whatever was found with any block the
        other passes missed — each candidate is verified by OCR so a stray
        detection just reads as no Japanese and is dropped.

        Every pass is isolated: a failure in one detector degrades gracefully
        (we keep what the others found) instead of crashing the whole page."""
        items = []
        try:
            items = self._free_text_llm(image, bubble_regions, update)
        except Exception as e:
            print(f"[pipeline] LLM free-text pass failed: {e}")
        if not items:
            try:
                items = self._free_text_cv(image, bubble_regions, update)
            except Exception as e:
                print(f"[pipeline] CV free-text pass failed: {e}")
                items = []
        try:
            items += self._free_text_seg(image, bubble_regions, items, update)
        except Exception as e:
            print(f"[pipeline] text-block free-text pass failed: {e}")
        return items

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

        from .ocr import _has_source_text
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

            # Site watermark / URL stamped on the art (eshadow.net, www.x.com):
            # erase it instead of translating. Checked BEFORE the source-script
            # filter, since a watermark is latin and would otherwise be dropped.
            if self.remove_watermark and (typ in ("watermark", "url", "logo", "credit_url")
                                          or _is_watermark(jp) or _is_watermark(tr)):
                box = [bx, by, bw, bh]
                if (not any(_boxes_overlap(box, bb) for bb in bubble_boxes)
                        and not any(_boxes_overlap(box, u) for u in used)):
                    used.append(box)
                    items.append({
                        "id": next_id, "bbox": box, "original": (jp or tr),
                        "translation": "", "type": "watermark", "erase": True,
                        "in_bubble": False, "dark": False, "rotation": 0.0,
                    })
                    next_id += 1
                continue

            # Only keep regions the model actually read as Japanese — guards
            # against boxes dropped on already-English text or bare artwork.
            # The "sfx" label alone isn't trusted: the LLM tags big dramatic
            # display lines (ここから…, 一人でこの戦場を…) as sfx, but those
            # carry meaning and official releases translate them. Skip only
            # when the text itself reads like onomatopoeia — unless the user
            # opted to translate SFX, in which case keep it as a placed sfx item.
            if not jp or not _has_source_text(jp, self.source_lang):
                continue
            is_sfx = _is_sfx(jp) or typ in ("sfx", "sound", "onomatopoeia")
            if is_sfx and not self.translate_sfx:
                continue
            if typ in ("sfx", "sound", "onomatopoeia") and not self.translate_sfx:
                typ = "narration"
            if not tr:
                continue

            box = [bx, by, bw, bh]
            if any(_boxes_overlap(box, bb) for bb in bubble_boxes):
                continue
            if any(_boxes_overlap(box, u) for u in used):
                continue

            # Vision-LLM coordinates can be sloppy — a det can carry the RIGHT
            # text with the WRONG box, which cleans and typesets a different
            # spot entirely. When local OCR reads something substantial in the
            # claimed box that shares nothing with the det's text, drop it:
            # the GPU catch-all pass finds the real block at real coordinates.
            if self.ocr is not None and self.ocr.ok:
                seen = (self.ocr.read_region(image, box, None) or "").strip()
                if len(seen) >= 2 and not _texts_match(seen, jp):
                    print(f"[pipeline] LLM box mismatch (claims {jp[:12]!r}, "
                          f"box reads {seen[:12]!r}) — dropped")
                    continue
            used.append(box)

            rotation = 0.0
            try:
                rotation = float(det.get("rotation_deg", 0))
            except (ValueError, TypeError):
                pass

            allowed = ("title", "credit", "narration", "caption")
            if self.translate_sfx and is_sfx:
                out_type = "sfx"
            else:
                out_type = typ if typ in allowed else "narration"
            items.append({
                "id": next_id,
                "bbox": box,
                "original": jp,
                "translation": tr,
                "type": out_type,
                "in_bubble": False,
                "dark": False,
                "rotation": rotation,
            })
            next_id += 1

        if items:
            update(2, f"Found {len(items)} free text regions", 58)
        return items

    def _free_text_seg(self, image, bubble_regions, existing_items, update) -> List[dict]:
        """Catch-all pass: the manga-trained text-block detector finds every
        block of lettering on the page. Any block not already covered by a
        bubble or a found free-text region is OCR'd; blocks that actually read
        as Japanese get translated and added."""
        if self.text_seg is None or not self.text_seg.ok:
            return []
        if self.ocr is None or not self.ocr.ok:
            return []
        try:
            boxes = self.text_seg.detect_blocks(image)
        except Exception as e:
            print(f"[pipeline] text-block detection failed: {e}")
            return []
        if not boxes:
            return []

        from .ocr import _has_source_text
        # Giant display lettering arrives as one box per glyph — merge the
        # aligned boxes into whole columns/runs so OCR reads full phrases.
        boxes = _merge_column_boxes(boxes)
        taken = [list(r.bbox) for r in bubble_regions]
        taken += [list(it["bbox"]) for it in existing_items]
        known_texts = [it.get("original", "") for it in existing_items]
        all_ids = [r.id for r in bubble_regions] + [it["id"] for it in existing_items]
        next_id = max(all_ids, default=0) + 1

        id_to_text: Dict[int, str] = {}
        box_map: Dict[int, tuple] = {}
        for box in boxes:
            if any(_boxes_overlap(list(box), tb) for tb in taken):
                continue
            jp = self.ocr.read_region(image, box, None)
            if not jp or not _has_source_text(jp, self.source_lang):
                continue
            if _is_sfx(jp) and not self.translate_sfx:
                continue
            # Same text already found by another pass (LLM box was off but
            # close enough that both versions would be placed = doubled text).
            if any(_texts_match(jp, t) for t in known_texts):
                continue
            taken.append(list(box))
            known_texts.append(jp)
            id_to_text[next_id] = jp
            box_map[next_id] = box
            next_id += 1

        if not id_to_text:
            return []

        update(2, f"Translating {len(id_to_text)} extra text blocks...", 56)
        try:
            translations = self.translator.translate_texts(id_to_text, self.target_lang, image=image)
        except Exception as e:
            print(f"[pipeline] extra block translation failed: {e}")
            return []

        items = []
        for fid, jp in id_to_text.items():
            tr = translations.get(fid, {})
            text = (tr.get("translation") or "").strip()
            if not text:
                continue
            items.append({
                "id": fid,
                "bbox": [int(v) for v in box_map[fid]],
                "original": jp,
                "translation": text,
                "type": tr.get("type", "narration"),
                "in_bubble": False,
                "dark": False,
            })
        if items:
            update(2, f"Caught {len(items)} extra text blocks (GPU detector)", 58)
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
            if _is_sfx(jp) and not self.translate_sfx:
                continue
            fid = next_id
            next_id += 1
            id_to_text[fid] = jp
            box_map[fid] = box

        if not id_to_text:
            return []

        update(2, f"Translating {len(id_to_text)} free text regions...", 55)
        try:
            translations = self.translator.translate_texts(id_to_text, self.target_lang, image=image)
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
        """Hybrid detection: the vision-LLM reads + translates the page (good
        translations, reading order, free text), while the GPU segmentation
        model supplies the PRECISE balloon shape for every bubble. The LLM box
        alone is a loose rectangle with no mask — that's what let text sprawl
        across the art and skipped bubbles. Here each LLM translation is snapped
        onto the bubble mask that overlaps it (so text is contained), and any
        balloon the LLM missed is OCR'd + translated so nothing is left behind."""
        from .ocr import _has_source_text
        update(1, "AI is analyzing the page...", 10)
        try:
            detections = self.translator.smart_detect_and_translate(image, self.target_lang)
        except Exception as e:
            print(f"[pipeline] smart detect failed, using standard detect: {e}")
            return self._standard_detect(image, output_path, update)
        update(2, f"Found {len(detections)} text regions", 35)

        h, w = image.shape[:2]
        llm_items = []
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
            d_orig = det.get("original", "")
            d_type = (det.get("type") or "dialogue").strip().lower()
            # Site watermark / URL → erase (no translation), not a bubble.
            if self.remove_watermark and (d_type in ("watermark", "url", "logo")
                                          or _is_watermark(d_orig)
                                          or _is_watermark(det.get("translation", ""))):
                llm_items.append({
                    "id": i + 1, "bbox": [x, y, bw, bh], "original": d_orig,
                    "translation": "", "type": "watermark", "erase": True,
                    "in_bubble": False, "dark": False, "rotation": 0.0,
                })
                continue
            llm_items.append({
                "id": i + 1,
                "bbox": [x, y, bw, bh],
                "original": d_orig,
                "translation": det.get("translation", ""),
                "type": det.get("type", "dialogue"),
                "in_bubble": det.get("in_bubble", True),
                "dark": False,
                "rotation": rotation,
            })

        # GPU balloon masks: precise containment + recall for missed bubbles.
        seg_regions = []
        try:
            update(2, "Locating balloons precisely (GPU)...", 42)
            seg_regions = self.detector.detect(image)
        except Exception as e:
            print(f"[pipeline] seg detect in smart mode failed: {e}")

        # No precise masks available (CPU-only / model absent): fall back to the
        # LLM boxes alone, but route bubble text through the free-text path so
        # it's contained to its strokes rather than a loose oversized box.
        if not seg_regions:
            for it in llm_items:
                it["in_bubble"] = False
            return llm_items, "", {}

        masks: Dict[int, Any] = {}
        items: List[dict] = []
        matched = set()

        # 1) Snap each precise bubble to the LLM translation that overlaps it.
        for reg in seg_regions:
            rb = [int(v) for v in reg.bbox]
            best, best_ov = None, 0.0
            for it in llm_items:
                if id(it) in matched or it.get("in_bubble") is False:
                    continue
                ov = _overlap_frac(rb, it["bbox"])
                if ov > best_ov:
                    best, best_ov = it, ov
            if best is not None and best_ov >= 0.2:
                matched.add(id(best))
                items.append({
                    "id": reg.id, "bbox": rb,
                    "original": best.get("original", ""),
                    "translation": best.get("translation", ""),
                    "type": best.get("type", "dialogue"),
                    "in_bubble": True,
                    "dark": bool(getattr(reg, "dark", False)),
                    "rotation": 0.0,
                })
                masks[reg.id] = reg.mask
                continue
            # Bubble the LLM missed entirely → read it locally and translate.
            jp = ""
            if self.ocr is not None and self.ocr.ok:
                jp = (self.ocr.read_region(image, rb, getattr(reg, "mask", None)) or "").strip()
            if not jp or not _has_source_text(jp, self.source_lang):
                continue
            tr = ""
            try:
                out = self.translator.translate_texts({reg.id: jp}, self.target_lang, image=image)
                tr = ((out.get(reg.id) or {}).get("translation") or "").strip()
            except Exception as e:
                print(f"[pipeline] missed-bubble translate failed: {e}")
            if not tr:
                continue
            items.append({
                "id": reg.id, "bbox": rb, "original": jp, "translation": tr,
                "type": "dialogue", "in_bubble": True,
                "dark": bool(getattr(reg, "dark", False)), "rotation": 0.0,
            })
            masks[reg.id] = reg.mask

        # 2) LLM detections with no matching balloon = free text over artwork.
        #    Keep them, but as free text so the compositor contains them to the
        #    actual lettering strokes instead of a loose rectangle.
        next_id = max([r.id for r in seg_regions] + [0]) + 1
        for it in llm_items:
            if id(it) in matched:
                continue
            it["id"] = next_id
            next_id += 1
            it["in_bubble"] = False
            items.append(it)

        # Also run the dedicated free-text / text-block detectors (the
        # manga-trained block model is far better at giant vertical title & SFX
        # lettering than the LLM's loose box). Merge anything that doesn't
        # overlap a region we already have — this is what stops a big title from
        # being missed or mistaken for a bubble in smart mode.
        try:
            extra = self._detect_free_text(image, seg_regions, update)
        except Exception as e:
            print(f"[pipeline] smart-mode free-text pass failed: {e}")
            extra = []
        if extra:
            next_id = max([it["id"] for it in items]
                          + [r.id for r in seg_regions] + [0]) + 1
            added = 0
            for it in extra:
                nb = list(it["bbox"])
                a_new = nb[2] * nb[3]
                hit = next((ex for ex in items
                            if _boxes_overlap(nb, list(ex["bbox"]))), None)
                if hit is not None:
                    a_ex = hit["bbox"][2] * hit["bbox"][3]
                    # The block detector found a MUCH bigger region here — a giant
                    # title/SFX the LLM under-boxed or mistook for a bubble. Adopt
                    # the big box and erase it as free text (keep the better
                    # wording). Normal bubbles (block box is smaller) are kept.
                    if a_new > 1.6 * a_ex:
                        hit["bbox"] = [int(v) for v in it["bbox"]]
                        hit["in_bubble"] = False
                        hit["rotation"] = it.get("rotation", hit.get("rotation", 0))
                        if not (hit.get("translation") or "").strip():
                            hit["translation"] = it.get("translation", "")
                        masks.pop(hit["id"], None)
                        added += 1
                    continue
                it["id"] = next_id
                next_id += 1
                items.append(it)
                added += 1
            if added:
                update(2, f"Added/expanded {added} text regions (block detector)", 56)

        update(2, f"Placed {len(items)} regions ({len(masks)} masked)", 50)
        return items, "", masks

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
