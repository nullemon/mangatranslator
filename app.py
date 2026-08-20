import asyncio
import io
import os
import time
import re
import uuid
import random
import zipfile
from pathlib import Path


def _load_env(path: str = ".env"):
    """Load KEY=VALUE lines from a local .env (gitignored) into the
    environment — e.g. HF_TOKEN so HuggingFace model downloads are
    authenticated. Variables already set in the environment win."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    except OSError:
        pass


_load_env()
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.requests import Request

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.pipeline import (TranslationPipeline, scan_cleanup, compress_upload,
                           preserve_dark_regions, probe_components)
from core.compositor import Compositor
from core.effects import raw_scan
from core.enhancer import ImageEnhancer


def compress_output(path: str, target_kb: int = 3072) -> str:
    """Re-encode a finished page as a tuned JPEG so big outputs (a 20MB PNG)
    come down to ~2-4MB. Steps quality down until under target. Returns the new
    .jpg path (and removes the original) or the original path on failure."""
    img = cv2.imread(path)
    if img is None:
        return path
    jpg = os.path.splitext(path)[0] + ".jpg"
    q, last = 92, None
    while q >= 40:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if not ok:
            break
        last = buf
        if len(buf) <= target_kb * 1024:
            break
        q -= 8
    if last is None:
        return path
    with open(jpg, "wb") as f:
        f.write(last.tobytes())
    if jpg != path:
        try:
            os.remove(path)
        except OSError:
            pass
    return jpg


_WM_FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
_WM_FONT_CANDIDATES = [
    os.path.join(_WM_FONT_DIR, "ComicNeue-Bold.ttf"),   # bundled, keeps case
    os.path.join(_WM_FONT_DIR, "BebasNeue-Regular.ttf"),
    os.path.join(_WM_FONT_DIR, "Anton-Regular.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]


def _stamp_all(output_path, watermark, wm_place="br", wm_opacity=50,
               wm_size="m", credit="", wm_style="clean"):
    """Stamp the user's watermark and/or credit line on ANY finished output
    (translate, upscale, raw, enhance — same look everywhere). The credit
    goes small in the opposite corner so the two never collide."""
    if watermark:
        _stamp_watermark(output_path, watermark, wm_place, wm_opacity, wm_size,
                         wm_style)
    if credit:
        cplace = "bl" if wm_place != "bl" else "br"
        _stamp_watermark(output_path, credit, cplace, 85, "s", "clean")


def _text_keepout(img, pad_px: int):
    """Where the watermark must NOT go: the page's lettering, fattened.

    A watermark dropped on top of dialogue ruins both — the mark is unreadable
    and so is the line under it. Corner placement used to take whatever was in
    the corner, and a bottom-right balloon is extremely common.

    Lettering is picked out by the shape of its ink rather than by how dark it
    is, because manga artwork is just as black as its text. Glyphs are small,
    thin-stroked and tightly grouped; panel borders are long and straight,
    screentone is tiny and evenly spread, and figure art is large. Components
    that pass are merged into blocks and grown by `pad_px`, so the mark keeps
    clear of the text rather than just missing it.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 25, 12)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    keep = np.zeros((h, w), np.uint8)
    # A glyph is a small fraction of the page height. Below this it is
    # screentone or dust; above it, artwork.
    lo, hi = max(4, int(h * 0.006)), int(h * 0.09)
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if not (lo <= ch <= hi) or cw > hi * 3:
            continue
        if area < 12 or cw < 2:
            continue
        # Solid blobs (a filled eye, a spot of black) and hairline rules (a
        # panel border, a speed line) are both common and neither is text.
        fill = area / float(max(1, cw * ch))
        if fill > 0.92 or fill < 0.05:
            continue
        ar = cw / float(max(1, ch))
        if ar > 8 or ar < 0.06:
            continue
        keep[lab == i] = 255

    glyphs = keep.copy()          # before any growing — used to find balloons
    if cv2.countNonZero(keep):
        # Join the glyphs of a line, then a block, so the gaps inside a word
        # are not read as somewhere the mark could sit.
        gap = max(3, int(h * 0.012))
        keep = cv2.dilate(keep, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (gap * 2 + 1, gap + 1)))

    # Speech balloons, whole.
    #
    # Glyph-shape detection has one predictable blind spot: a character that
    # TOUCHES the balloon outline merges with it into a single component far
    # too big to be a letter, so it is dropped and the mark can sit on it. A
    # balloon is where dialogue lives, so the safer rule is that the mark never
    # goes inside one at all, whether or not every glyph was picked out.
    #
    # Balloons are found as bright blobs that CONTAIN lettering. That last part
    # is what makes it safe: the page background and the panel gutters are just
    # as bright, and they hold no text, so they are not mistaken for balloons
    # and the whole page does not become out of bounds. All plain OpenCV — no
    # model, no download, a few milliseconds.
    if cv2.countNonZero(glyphs):
        _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        # Outer contours, filled. The lettering punches holes in a balloon's
        # bright interior and taking the outer boundary closes them back up.
        # Deliberately NOT a morphological close: a kernel wide enough to span
        # the gaps between lines of dialogue also steps straight over the
        # balloon's own outline, merging it into the panel behind and making
        # the whole thing too big to recognise.
        cnts, _h = cv2.findContours(bright, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        page_area = float(h * w)
        for c in cnts:
            x, y, cw, ch = cv2.boundingRect(c)
            area = float(cv2.contourArea(c))
            if area < page_area * 0.004 or area > page_area * 0.30:
                continue
            # A blob spanning almost the whole page is the paper, not a bubble.
            if cw > w * 0.85 or ch > h * 0.85:
                continue
            # Balloons are compact; a sprawling bright background is not.
            if area / float(max(1, cw * ch)) < 0.5:
                continue
            filled = np.zeros((h, w), np.uint8)
            cv2.drawContours(filled, [c], -1, 255, -1)
            if not (glyphs[filled > 0] > 0).any():
                continue                     # bright but empty — not a bubble
            cv2.rectangle(keep, (x, y), (x + cw, y + ch), 255, -1)

    if pad_px > 0 and cv2.countNonZero(keep):
        keep = cv2.dilate(keep, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (pad_px * 2 + 1, pad_px * 2 + 1)))
    return keep


def _clear_spot(img, tw, th, place, keepout):
    """The best top-left corner for a `tw`×`th` mark.

    The chosen corner is honoured whenever it is actually free — "bottom
    right" should stay bottom right, and quietly relocating a mark the user
    positioned is its own kind of wrong. Only when the preferred spot lands on
    text does this look elsewhere, trying the same corner nudged along the
    edge first and the other corners after that, so the mark moves as little
    as it can get away with.
    """
    h, w = img.shape[:2]
    m = max(12, int(w * 0.015))
    tw, th = max(1, int(tw)), max(1, int(th))
    if tw >= w or th >= h:
        return m, m
    # Summed-area table, so scoring a candidate box is four lookups however
    # many candidates the sweep below tries.
    integral = cv2.integral((keepout > 0).astype(np.uint8), sdepth=cv2.CV_32S)

    def hits(x, y):
        x0, y0 = int(np.clip(x, 0, w - tw)), int(np.clip(y, 0, h - th))
        x1, y1 = x0 + tw, y0 + th
        return int(integral[y1, x1] - integral[y0, x1]
                   - integral[y1, x0] + integral[y0, x0])

    corners = {"br": (w - tw - m, h - th - m), "bl": (m, h - th - m),
               "tr": (w - tw - m, m), "tl": (m, m)}
    order = ([place] if place in corners else []) + \
            [c for c in ("br", "bl", "tr", "tl") if c != place]

    cands = []
    for c in order:
        cx, cy = corners[c]
        cands.append((cx, cy))
        # Same corner, nudged: along its edge, then inward. A mark that shifts
        # a little still reads as "in the corner".
        for d in (1, 2, 3):
            step = int(th * 0.9 * d)
            cands.append((cx, cy - step if c in ("br", "bl") else cy + step))
            side = int(tw * 0.35 * d)
            cands.append((cx - side if c in ("br", "tr") else cx + side, cy))

    for x, y in cands:
        if hits(x, y) == 0:
            return int(np.clip(x, 0, w - tw)), int(np.clip(y, 0, h - th))

    # Nowhere clear in a corner — sweep the page and take the emptiest spot.
    best, score = corners.get(place, (m, m)), None
    for y in range(m, max(m + 1, h - th - m), max(8, th // 2)):
        for x in range(m, max(m + 1, w - tw - m), max(8, tw // 3)):
            s = hits(x, y)
            if score is None or s < score:
                score, best = s, (x, y)
                if s == 0:
                    return best
    return int(np.clip(best[0], 0, w - tw)), int(np.clip(best[1], 0, h - th))


def _stamp_watermark(image_path: str, text: str, place: str = "br",
                     opacity: int = 50, size: str = "m", style: str = "clean"):
    """Watermark the finished page. Six styles, all sized off the PAGE WIDTH
    so they stay readable at any resolution (the old min-side divisors made
    marks near-invisible on tall scans):

      clean  — text with a thin contrasting outline (auto light/dark)
      bold   — heavy display text with a thick contrast stroke
      pill   — white text on a rounded dark badge
      ribbon — full-width translucent band along the top or bottom edge
      ghost  — big translucent diagonal text across the page
      tile   — small text repeated diagonally over everything

    `place` = tl/tr/bl/br/random for point styles; for ribbon it picks the
    top or bottom edge. `opacity` 0-100."""
    img = cv2.imread(image_path)
    if img is None:
        return
    h, w = img.shape[:2]
    style = (style or "clean").lower()
    if place == "tile":                       # legacy value from old configs
        style, place = "tile", "br"
    alpha = int(np.clip(int(opacity or 50), 5, 100) * 255 / 100)
    sf = {"s": 0.024, "m": 0.034, "l": 0.048}.get(size, 0.034)
    fs = max(16, int(w * sf))
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    def _font(size_px):
        for p in _WM_FONT_CANDIDATES:
            try:
                if os.path.exists(p):
                    return ImageFont.truetype(p, size_px)
            except Exception:
                continue
        try:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", size_px)
        except Exception:
            return ImageFont.load_default()

    def _measure(font):
        bb = draw.textbbox((0, 0), text, font=font)
        return bb, bb[2] - bb[0], bb[3] - bb[1]

    # Worked out once, only if a placed style actually needs it — the
    # full-page styles below use it differently and the ribbon spans the width
    # regardless.
    _ko = {}

    def _keepout():
        if "m" not in _ko:
            _ko["m"] = _text_keepout(img, max(4, int(fs * 0.35)))
        return _ko["m"]

    def _corner_xy(tw, th, pad):
        m = max(12, int(w * 0.015))
        if place == "random":
            # "Random quiet spot" used to mean twelve random guesses scored by
            # how flat they were. Flat is exactly what the inside of a speech
            # balloon is, so it would happily drop the mark next to a line of
            # dialogue. It now means the quietest spot that is genuinely clear
            # of lettering.
            keep = _keepout()
            gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            x_lo, y_lo = int(0.06 * w), int(0.06 * h)
            x_hi = max(int(0.94 * w) - tw, x_lo + 1)
            y_hi = max(int(0.94 * h) - th, y_lo + 1)
            best, score = None, 1e18
            for _ in range(60):
                cx = random.randint(x_lo, x_hi)
                cy = random.randint(y_lo, y_hi)
                if keep[cy:cy + th, cx:cx + tw].any():
                    continue
                reg = gray_full[cy:cy + th, cx:cx + tw]
                if reg.size and float(reg.std()) < score:
                    score, best = float(reg.std()), (cx, cy)
            return best if best else _clear_spot(img, tw, th, "br", keep)
        # Keep the chosen corner when it is free; step aside when it is not.
        return _clear_spot(img, tw, th, place, _keepout())

    def _auto_colors(x, y, tw, th):
        region = img[max(0, y - 4):min(h, y + th + 4),
                     max(0, x - 4):min(w, x + tw + 4)]
        lum = float(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).mean()) if region.size else 255.0
        if lum < 128:
            return (255, 255, 255, alpha), (15, 15, 15, alpha)
        return (25, 25, 25, alpha), (255, 255, 255, alpha)

    if style == "tile":
        font = _font(max(14, int(w * 0.022)))
        bb, tw, th = _measure(font)
        for yy in range(-h, h * 2, th * 4 + 40):
            for xx in range(-w, w * 2, tw + max(80, tw)):
                draw.text((xx, yy), text, fill=(128, 128, 128, alpha), font=font)
        overlay = overlay.rotate(30, expand=False, center=(w // 2, h // 2))

    elif style == "ghost":
        font = _font(max(40, int(w * 0.10)))
        bb, tw, th = _measure(font)
        ga = max(18, int(alpha * 0.30))
        gl = Image.new("RGBA", (tw + 40, th + 40), (0, 0, 0, 0))
        ImageDraw.Draw(gl).text((20 - bb[0], 20 - bb[1]), text,
                                fill=(90, 90, 90, ga), font=font)
        gl = gl.rotate(-24, expand=True, resample=Image.BICUBIC)
        overlay.alpha_composite(gl, (max(0, (w - gl.width) // 2),
                                     max(0, (h - gl.height) // 2)))

    elif style == "ribbon":
        font = _font(fs)
        bb, tw, th = _measure(font)
        band_h = int(th * 1.9)
        y0 = 0 if place in ("tl", "tr") else h - band_h
        # A ribbon is anchored to an edge and spans the width, so it cannot be
        # nudged out of the way — but it can take the other edge. Pick the one
        # with less lettering under it.
        keep = _text_keepout(img, max(2, int(fs * 0.15)))
        top_hit = int((keep[0:band_h, :] > 0).sum())
        bot_hit = int((keep[h - band_h:h, :] > 0).sum())
        if y0 == 0 and top_hit > bot_hit:
            y0 = h - band_h
        elif y0 != 0 and bot_hit > top_hit:
            y0 = 0
        draw.rectangle([0, y0, w, y0 + band_h],
                       fill=(12, 12, 12, min(235, int(alpha * 1.1))))
        draw.text(((w - tw) // 2 - bb[0],
                   y0 + (band_h - th) // 2 - bb[1]), text,
                  fill=(255, 255, 255, 255), font=font)

    elif style == "pill":
        font = _font(fs)
        bb, tw, th = _measure(font)
        px_, py_ = int(fs * 0.75), int(fs * 0.42)
        pw, ph = tw + 2 * px_, th + 2 * py_
        x, y = _corner_xy(pw, ph, 0)
        draw.rounded_rectangle([x, y, x + pw, y + ph], radius=ph // 2,
                               fill=(14, 14, 14, min(235, int(alpha * 1.15))))
        draw.text((x + px_ - bb[0], y + py_ - bb[1]), text,
                  fill=(255, 255, 255, 255), font=font)

    elif style == "bold":
        font = _font(int(fs * 1.25))
        bb, tw, th = _measure(font)
        x, y = _corner_xy(tw, th, 0)
        fill, stroke = _auto_colors(x, y, tw, th)
        draw.text((x - bb[0], y - bb[1]), text, font=font, fill=fill,
                  stroke_width=max(2, font.size // 9), stroke_fill=stroke)

    else:  # clean
        font = _font(fs)
        bb, tw, th = _measure(font)
        x, y = _corner_xy(tw, th, 0)
        fill, stroke = _auto_colors(x, y, tw, th)
        draw.text((x - bb[0], y - bb[1]), text, font=font, fill=fill,
                  stroke_width=max(1, font.size // 16), stroke_fill=stroke)

    if style in ("tile", "ghost"):
        # These two cover the whole page by design, so there is nowhere to
        # move them to. Instead the mark is cut away wherever it would cross
        # lettering, which reads as the watermark passing BEHIND the text —
        # the page stays legible and the mark still covers the art.
        # A generous margin here is deliberate. A mark cut flush to the glyph
        # edges still crowds them, and it also covers the one case the glyph
        # pass can miss: a character touching a balloon outline merges with it
        # and is dropped, so the halo around its neighbours has to reach far
        # enough to cover it. The margin reads as intentional either way.
        keep = _text_keepout(img, max(6, int(w * 0.022)))
        if cv2.countNonZero(keep):
            a = np.array(overlay.split()[-1])
            a[keep > 0] = 0
            overlay.putalpha(Image.fromarray(a))

    pil = Image.alpha_composite(pil, overlay)
    result = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, result)

app = FastAPI(title="MangaTranslator")

for d in ("uploads", "output", "fonts"):
    os.makedirs(d, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
# Serve the typeset fonts so the Font dropdown can PREVIEW each one.
app.mount("/fonts", StaticFiles(directory="fonts"), name="fonts")
templates = Jinja2Templates(directory="templates")

tasks: dict = {}
# Balloon interior masks per task (numpy arrays — kept out of `tasks` so the
# JSON status endpoint stays serializable). Reused for clean re-renders.
MASKS: dict = {}

# ---- Memory housekeeping ------------------------------------------------
# Between jobs the loaded models sit on VRAM and torch/Python never hand
# freed memory back on their own — on WSL2 that reads as "vmmem keeps
# growing / GPU stays hogged" long after the last page. The server now
# cleans up after itself: a light sweep after every job, a trim after 10
# idle minutes, and a full model release after 30 (models lazy-reload on
# the next job in a few seconds).
from core import memory as memsweep

# One re-render at a time per page. The editor fires an Apply per edit, so
# several can land together; without this they all write the SAME output file
# concurrently and each other's readers see a half-written PNG ("libpng error:
# IDAT: CRC error", and a briefly garbled page in the browser).
_TASK_LOCKS = {}


def _task_lock(task_id: str):
    lk = _TASK_LOCKS.get(task_id)
    if lk is None:
        lk = asyncio.Lock()
        _TASK_LOCKS[task_id] = lk
        while len(_TASK_LOCKS) > 64:          # bound the bookkeeping
            _TASK_LOCKS.pop(next(iter(_TASK_LOCKS)), None)
    return lk


def _write_atomic(path: str, img) -> bool:
    """Write an image so readers never see it half-finished: render to a temp
    file beside the target, then rename (atomic on the same filesystem)."""
    # Keep the real extension on the temp name: OpenCV picks its encoder from
    # it, and a bare ".tmp1234" makes imwrite fail outright.
    root, ext = os.path.splitext(path)
    tmp = f"{root}.tmp{os.getpid()}{ext or '.png'}"
    try:
        if not cv2.imwrite(tmp, img):
            return False
        os.replace(tmp, path)
        return True
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


LAST_JOB_TS = time.time()
_IDLE_TRIMMED = False
_MODELS_DROPPED = False


def _note_job_done():
    """A full page job finished — reset the idle clock and do a light sweep."""
    global LAST_JOB_TS, _IDLE_TRIMMED, _MODELS_DROPPED
    LAST_JOB_TS = time.time()
    _IDLE_TRIMMED = False
    _MODELS_DROPPED = False
    memsweep.light_sweep()


def _note_activity():
    """Editing counts as being in use.

    Every editor action — re-render, OCR a drawn box, re-translate — runs the
    same heavy models as a full job, but only the batch runners used to reset
    the idle clock. So a long editing session looked "idle": the housekeeper
    unloaded the models EVERY minute, and the next Apply paid to reload
    text-seg and LaMa. No sweep here (it runs many times per minute) — just
    the timestamp."""
    global LAST_JOB_TS, _IDLE_TRIMMED, _MODELS_DROPPED
    LAST_JOB_TS = time.time()
    _IDLE_TRIMMED = False
    _MODELS_DROPPED = False


async def _housekeeper():
    global LAST_JOB_TS, _IDLE_TRIMMED, _MODELS_DROPPED
    loop = asyncio.get_event_loop()
    while True:
        await asyncio.sleep(60)
        try:
            if any(t.get("status") == "processing" for t in tasks.values()):
                LAST_JOB_TS = time.time()
                continue
            idle = time.time() - LAST_JOB_TS
            if idle >= 1800:
                if not _MODELS_DROPPED:
                    released = await loop.run_in_executor(
                        None, memsweep.unload_models)
                    _MODELS_DROPPED = True
                    if released:
                        print("[mem] idle 30 min — models released, VRAM freed "
                              "(they reload automatically on the next job)")
            elif idle >= 600 and not _IDLE_TRIMMED:
                await loop.run_in_executor(None, memsweep.deep_sweep)
                _IDLE_TRIMMED = True
                print("[mem] idle 10 min — trimmed cached RAM/VRAM back to the OS")
        except Exception as e:
            print(f"[mem] housekeeper: {e}")


@app.on_event("startup")
async def _start_housekeeper():
    asyncio.create_task(_housekeeper())


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html",
        context={"version": _asset_version(), "build": _SERVER_COMMIT})


def _asset_version() -> str:
    """Cache-buster appended to the CSS/JS URLs. Uses the newest mtime of the
    static assets so a browser always fetches fresh code after an update,
    instead of silently running a stale app.js."""
    newest = 0.0
    for p in ("static/js/app.js", "static/css/style.css", "templates/index.html"):
        try:
            newest = max(newest, os.path.getmtime(p))
        except OSError:
            pass
    return str(int(newest)) or "1"


_HEALTH_CACHE = {}


def _git_commit() -> str:
    """Short commit the server is running, so the UI/logs can prove whether the
    backend was actually restarted on the latest code (a frequent source of
    'the fix didn't work' — the browser updated but app.py wasn't restarted)."""
    try:
        import subprocess
        c = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
        if c:
            return c
    except Exception:
        pass
    # Standalone install (no repo): report the shipped release version instead.
    try:
        vp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VERSION")
        with open(vp, encoding="utf-8") as f:
            v = f.read().strip()
        if v:
            return v
    except Exception:
        pass
    return "standalone"


_SERVER_COMMIT = _git_commit()
print(f"[app] ===== MangaTranslator server starting — commit {_SERVER_COMMIT} =====")

LAST_TRANSLATE_TASK = None  # most recent /api/translate task, for /api/debug


def _debug_dump(tid):
    """Human-readable dump of what was detected on a page — text + boxes — so a
    detection problem can be diagnosed by pasting the output, no screenshots."""
    t = tasks.get(tid) or {}
    r = t.get("result") or {}
    items = r.get("items", [])
    head = [
        f"build {_SERVER_COMMIT}   task {tid}",
        f"src={t.get('source_lang')} -> {t.get('target_lang')}   "
        f"smart={t.get('smart_mode')}   regions={len(items)}",
        "-" * 70,
    ]
    rows = []
    for it in items:
        o = (it.get("original") or "").replace("\n", " ")[:22]
        tr = (it.get("translation") or "").replace("\n", " ")[:38]
        rows.append(
            f"#{it.get('id'):>3} {str(it.get('type','')):9} "
            f"bubble={str(it.get('in_bubble'))[0]} placed={str(it.get('placed'))[0]} "
            f"bbox={it.get('bbox')}  {o!r} -> {tr!r}")
    if not items:
        rows.append("(no regions — nothing was detected/translated on this page)")
    return Response("\n".join(head + rows), media_type="text/plain")


@app.get("/api/debug")
async def debug_last():
    return _debug_dump(LAST_TRANSLATE_TASK)


@app.get("/api/debug/{task_id}")
async def debug_task(task_id: str):
    return _debug_dump(task_id)


@app.get("/api/health")
async def health(refresh: bool = False):
    """Which detection / cleanup / GPU / RTL components are available, plus the
    server's git commit. Cheap (no heavy model loads) and cached. curl + share:
        curl -s localhost:8000/api/health | python3 -m json.tool"""
    if refresh or not _HEALTH_CACHE:
        loop = asyncio.get_event_loop()
        _HEALTH_CACHE.update(await loop.run_in_executor(None, probe_components))
    _HEALTH_CACHE["server_commit"] = _SERVER_COMMIT
    return _HEALTH_CACHE


@app.post("/api/merge-strip")
async def merge_strip(files: list[UploadFile] = File(...),
                      trim_seams: str = Form("true")):
    """Webtoon mode: stack the uploaded episode slices into ONE long strip.

    Slices arrive in the order the browser sent them (the frontend sorts by
    filename first, so 01, 02, ... 10 stack correctly rather than 1, 10, 2).
    Returns a merged image the normal translate flow then treats as a single
    very tall page."""
    if not files:
        raise HTTPException(400, "Upload at least one image")
    from core.pipeline import stitch_vertical

    mid = str(uuid.uuid4())
    parts = []
    try:
        for i, f in enumerate(files):
            if not f.content_type or not f.content_type.startswith("image/"):
                continue
            p = f"uploads/{mid}_part{i:03d}{Path(f.filename or '.png').suffix or '.png'}"
            with open(p, "wb") as fh:
                fh.write(await f.read())
            parts.append(p)
        if not parts:
            raise HTTPException(400, "No readable images in the upload")

        merged_path = f"uploads/{mid}_strip.png"
        loop = asyncio.get_event_loop()
        h, w, n = await loop.run_in_executor(
            None,
            lambda: stitch_vertical(parts, merged_path,
                                    trim_seams=(trim_seams == "true")),
        )
    finally:
        for p in parts:                     # the slices are merged; drop them
            try:
                os.remove(p)
            except OSError:
                pass

    return {"strip_id": mid, "url": f"/api/strip/{mid}",
            "width": w, "height": h, "slices": n}


@app.get("/api/strip/{strip_id}")
async def get_strip(strip_id: str):
    """The merged webtoon strip, so the browser can preview it before running."""
    if not re.fullmatch(r"[a-f0-9\-]{36}", strip_id or ""):
        raise HTTPException(404, "Not found")
    p = f"uploads/{strip_id}_strip.png"
    if not os.path.exists(p):
        raise HTTPException(404, "Strip not found")
    return FileResponse(p, media_type="image/png")


@app.post("/api/translate")
async def translate(
    file: UploadFile = File(...),
    api_key: str = Form(""),
    target_lang: str = Form("English"),
    provider: str = Form("claude"),
    model: str = Form(""),
    smart_mode: str = Form("false"),
    font: str = Form(""),
    enhance: str = Form("false"),
    enhance_provider: str = Form("gemini"),
    enhance_key: str = Form(""),
    enhance_prompt: str = Form(""),
    enhance_model: str = Form(""),
    watermark: str = Form(""),
    wm_place: str = Form("br"),
    wm_opacity: str = Form("50"),
    wm_size: str = Form("m"),
    wm_style: str = Form("clean"),
    style_prompt: str = Form(""),
    text_case: str = Form("upper"),
    finish: str = Form("clean"),
    upscale: str = Form("false"),
    source_lang: str = Form("Japanese"),
    translate_sfx: str = Form("false"),
    max_quality: str = Form("false"),
    remove_watermark: str = Form("true"),
    replace_watermark: str = Form("false"),
    clean_only: str = Form("false"),
    isolate_page: str = Form("false"),
    compress: str = Form("false"),
    credit: str = Form(""),
    profile: str = Form(""),
    gpu_cap: str = Form("100"),
    cut_regions: str = Form(""),
    one_by_one: str = Form("false"),
    webtoon: str = Form("false"),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload an image file")
    from core.gpu_throttle import set_cap as _set_gpu_cap
    _set_gpu_cap(gpu_cap)
    is_clean = clean_only == "true"
    is_offline = provider.strip().lower() in ("local", "offline")
    if not is_clean and not is_offline and not api_key:
        raise HTTPException(400, "api_key is required to translate")
    if is_offline and smart_mode == "true":
        # Smart Detection is a vision-model feature; offline uses the local
        # balloon detector + manga-ocr instead. Fall back silently rather
        # than failing the run.
        print("[translate] offline engine: Smart Detection not available, "
              "using the local detectors")
        smart_mode = "false"

    # Trained series profile: fold the learned glossary + house style into the
    # style instructions so this chapter matches the team's established style.
    style_prompt = style_prompt or ""
    if profile.strip():
        from core import profiles as _profiles
        _prof = _profiles.load(profile.strip())
        if _prof:
            block = _profiles.prompt_block(_prof)
            style_prompt = (block + "\n\n" + style_prompt).strip() if style_prompt.strip() else block

    task_id = str(uuid.uuid4())
    global LAST_TRANSLATE_TASK
    LAST_TRANSLATE_TASK = task_id
    ext = Path(file.filename or "img.png").suffix or ".png"
    upload_path = f"uploads/{task_id}{ext}"
    output_path = f"output/{task_id}{ext}"

    # Maximum Quality keeps the upload uncompressed (full resolution end-to-end).
    # Webtoon strips must NOT be capped at the normal 4000px long side — that
    # would squash a 20000px episode down and destroy every letter. The band
    # processor keeps memory sane instead.
    content = compress_upload(
        await file.read(),
        full=(max_quality == "true" or webtoon == "true"))
    with open(upload_path, "wb") as f:
        f.write(content)

    font_path = f"fonts/{font}" if font and os.path.exists(f"fonts/{font}") else None

    tasks[task_id] = {
        "status": "processing",
        "step": 0,
        "message": "Queued",
        "progress": 0,
        "upload_path": upload_path,
        "output_path": output_path,
        "font_path": font_path,
        "watermark": watermark.strip(),
        "wm_place": wm_place.strip() or "br",
        "wm_style": wm_style.strip() or "clean",
        "wm_opacity": int(wm_opacity) if str(wm_opacity).strip().isdigit() else 50,
        "wm_size": wm_size.strip() or "m",
        "text_case": text_case,
        "finish": finish,
        "enhance_provider": enhance_provider,
        "enhance_key": enhance_key,
        "enhance_model": enhance_model,
        "enhance_prompt": enhance_prompt,
        "name": file.filename or "page.png",
        "mode": "translate",
        "source_lang": source_lang,
        "target_lang": target_lang,
        "smart_mode": smart_mode == "true",
        "webtoon": webtoon == "true",
        "translate_sfx": translate_sfx == "true",
        "max_quality": max_quality == "true",
        "remove_watermark": remove_watermark == "true",
        "replace_watermark": replace_watermark == "true",
    }

    asyncio.create_task(
        _run(
            task_id, upload_path, output_path, api_key, target_lang, provider, model,
            (smart_mode == "true") and one_by_one != "true", font_path,
            enhance == "true", enhance_provider, enhance_key, enhance_prompt, enhance_model,
            watermark=watermark.strip(),
            wm_place=wm_place.strip() or "br",
            wm_opacity=int(wm_opacity) if str(wm_opacity).strip().isdigit() else 50,
            wm_size=wm_size.strip() or "m",
            wm_style=wm_style.strip() or "clean",
            style_prompt=style_prompt.strip(),
            text_case=text_case,
            finish=finish,
            upscale=(upscale == "true"),
            source_lang=source_lang,
            translate_sfx=(translate_sfx == "true"),
            max_quality=(max_quality == "true"),
            remove_watermark=(remove_watermark == "true"),
            replace_watermark=(replace_watermark == "true"),
            clean_only=is_clean,
            isolate_page=(isolate_page == "true"),
            compress=(compress == "true"),
            credit=credit.strip(),
            cut_regions=cut_regions,
            one_by_one=(one_by_one == "true"),
            webtoon=(webtoon == "true"),
        )
    )

    return {"task_id": task_id}


async def _run(
    task_id: str,
    image_path: str,
    output_path: str,
    api_key: str,
    target_lang: str,
    provider: str,
    model: str,
    smart_mode: bool,
    font_path: str = None,
    enhance: bool = False,
    enhance_provider: str = "gemini",
    enhance_key: str = "",
    enhance_prompt: str = "",
    enhance_model: str = "",
    watermark: str = "",
    wm_place: str = "br",
    wm_opacity: int = 50,
    wm_size: str = "m",
    wm_style: str = "clean",
    style_prompt: str = "",
    text_case: str = "upper",
    finish: str = "clean",
    upscale: bool = False,
    source_lang: str = "Japanese",
    translate_sfx: bool = False,
    max_quality: bool = False,
    remove_watermark: bool = True,
    replace_watermark: bool = False,
    clean_only: bool = False,
    isolate_page: bool = False,
    compress: bool = False,
    credit: str = "",
    cut_regions: str = "",
    one_by_one: bool = False,
    webtoon: bool = False,
):
    try:
        loop = asyncio.get_event_loop()
        # SBS "cut into pieces": parse the drawn regions (normalised polygons).
        pieces = []
        if cut_regions:
            try:
                import json as _json
                parsed = _json.loads(cut_regions)
                if isinstance(parsed, list):
                    pieces = [r for r in parsed if isinstance(r, list) and len(r) >= 3]
            except Exception as e:
                print(f"[run] bad cut_regions ignored: {e}")

        # Default: translate on the exact uploaded pixels. The Scan workflows
        # below redirect this to the cleaned/enhanced page.
        translate_source = image_path

        if enhance:
            tasks[task_id].update(
                {"step": 0, "progress": 2,
                 "message": "Preprocessing image (crop + clean)..."}
            )
            enhanced_path = f"uploads/{task_id}_enhanced.png"
            enhancer = ImageEnhancer()

            def do_enhance():
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError(f"Cannot load image: {image_path}")
                tasks[task_id].update(
                    {"progress": 15,
                     "message": f"Sending to {enhance_provider.title()} (this can take 30-60s)..."}
                )
                ai_ok = False
                try:
                    if enhance_provider == "local":
                        # LOCAL accurate scan: deterministic cleanup + crisp B&W —
                        # the art is never redrawn, so it is always 1:1 faithful
                        # (no hallucination, no restyle, no seams). No API needed.
                        tasks[task_id].update(
                            {"progress": 25,
                             "message": "Cleaning locally (accurate — no AI redraw)..."})
                        from core.pipeline import scan_finish
                        out = scan_finish(scan_cleanup(img))
                        tasks[task_id].update({"progress": 35, "message": "Local scan complete!"})
                    else:
                        # Send the RAW page straight to the AI scanner. Pre-deskewing
                        # locally here and letting the pipeline deskew again warped
                        # the page twice and stretched it on the way back (visible
                        # distortion). One clean pass: AI scans, pipeline deskews once.
                        out = enhancer.enhance(img, enhance_prompt, enhance_provider, enhance_key, enhance_model)
                        ai_ok = True
                        tasks[task_id].update({"progress": 35, "message": "AI enhancement complete!"})
                except Exception as e:
                    print(f"[enhance] AI step failed, using local scan cleanup: {e}")
                    # Surface the REAL reason (bad key, quota, wrong model) so the
                    # user sees why the page wasn't AI-scanned — not a vague
                    # "failed" that looks like the local result is the AI one.
                    reason = str(e).strip() or type(e).__name__
                    tasks[task_id].update(
                        {"progress": 35,
                         "enhance_error": reason[:300],
                         "message": f"⚠ {enhance_provider.title()} scan FAILED — "
                                    f"{reason[:160]} — fell back to local cleanup "
                                    f"(not the AI scan you asked for)."}
                    )
                    out = scan_cleanup(img)
                # Snap the AI result back to the EXACT source geometry so nothing
                # is stretched and detection boxes stay aligned. (AI output only —
                # the local path may legitimately crop the photo's background, and
                # stretching that back would distort the page.)
                if ai_ok and out.shape[:2] != img.shape[:2]:
                    out = cv2.resize(out, (img.shape[1], img.shape[0]),
                                     interpolation=cv2.INTER_AREA)
                # Claw back solid-black art the generative scan bleached to white
                # (black panels, gutters, white-on-black titles stay as drawn).
                if ai_ok:
                    try:
                        out = preserve_dark_regions(out, img)
                    except Exception as e:
                        print(f"[enhance] dark-region preserve skipped: {e}")
                cv2.imwrite(enhanced_path, out)

            await loop.run_in_executor(None, do_enhance)
            tasks[task_id]["enhanced_path"] = enhanced_path
            tasks[task_id]["enhanced_url"] = f"/api/enhanced/{task_id}"
            # Scan workflows (Raw → Scan → Translate) translate ON the cleaned /
            # AI-scanned page — that IS the point of the scan step: the user
            # wants the clean TCB-style result with English typeset over it. So
            # the enhanced page becomes the translation base. (Plain Raw →
            # Translate doesn't enhance, so it stays on the original pixels.)
            translate_source = enhanced_path
            # The enhanced page already carries the desired clean-scan look — do
            # NOT run the local scan finish over it again; keep it as delivered.
            finish = "off"
            tasks[task_id]["finish"] = "off"

        pipeline = TranslationPipeline(
            api_key=api_key,
            target_lang=target_lang,
            provider=provider,
            model=model,
            use_smart_detection=smart_mode,
            font_path=font_path,
            style_prompt=style_prompt,
            text_case=text_case,
            finish=finish,
            upscale=upscale,
            source_lang=source_lang,
            translate_sfx=translate_sfx,
            max_quality=max_quality,
            remove_watermark=remove_watermark,
            replace_watermark=replace_watermark,
            watermark_text=watermark,
            clean_only=clean_only,
            isolate_page=isolate_page,
            credit=credit,
            one_by_one=one_by_one,
            webtoon=webtoon,
        )

        def on_progress(update):
            tasks[task_id].update(update)

        # "Cut into pieces" (SBS): translate each drawn region on its own and
        # merge back, when regions were provided and we're actually translating.
        if pieces and not clean_only:
            result = await loop.run_in_executor(
                None,
                lambda: pipeline.process_pieces(
                    translate_source, output_path, pieces, on_progress),
            )
        elif webtoon and not clean_only:
            # Long vertical strip: walk it in overlapping bands so the
            # lettering is never resized into mush, then compose one long page.
            result = await loop.run_in_executor(
                None,
                lambda: pipeline.process_webtoon(
                    translate_source, output_path, on_progress),
            )
        else:
            result = await loop.run_in_executor(
                None,
                lambda: pipeline.process(translate_source, output_path, on_progress),
            )
        MASKS[task_id] = getattr(pipeline, "last_masks", {}) or {}
        # Bound memory: full-page bubble masks are ~MBs each at high res and
        # accumulate every page — a long batch quietly eats gigabytes and lags
        # the machine. Keep only the most recent pages' masks; older pages
        # re-render mask-less (bubbles are recovered from their boxes instead).
        while len(MASKS) > 8:
            MASKS.pop(next(iter(MASKS)), None)

        # NOTE: the delivered translation is NEVER run through the generative
        # enhancer. A whole-page generative pass repaints the art — it redraws
        # hair, drops labels it doesn't understand, and bleaches screentone to
        # hard B&W — which violates the one rule: only the text is touched, the
        # art stays byte-for-byte as drawn. The "api" finish now just delivers
        # the clean-scanned surgical page (scan_finish, applied in the
        # pipeline). The generative model stays available as the explicit
        # "Enhance & Translate" workflow, where it produces a SEPARATE image.

        # User watermark (corner by default; tiled optional). Skipped when
        # "replace watermark" is on — there the user's mark is dropped in place
        # of the erased site watermark instead.
        if watermark and not replace_watermark:
            _stamp_watermark(output_path, watermark, wm_place, wm_opacity,
                             wm_size, wm_style)

        # Optional: shrink a heavy output (e.g. a 20MB PNG) to a ~3MB JPEG.
        if compress:
            try:
                newp = await loop.run_in_executor(None, lambda: compress_output(output_path))
                if newp != output_path and isinstance(result, dict):
                    result["output_path"] = newp
            except Exception as e:
                print(f"[run] output compress failed: {e}")

        update = {
            "status": "done",
            "progress": 100,
            "message": "Complete!",
            "result": result,
            "output_url": f"/api/result/{task_id}",
            "original_url": f"/api/original/{task_id}",
        }
        if tasks[task_id].get("enhanced_url"):
            update["enhanced_url"] = tasks[task_id]["enhanced_url"]
        tasks[task_id].update(update)

    except Exception as e:
        if task_id in tasks:
            tasks[task_id].update(
                {"status": "error", "message": str(e), "progress": 0}
            )
    finally:
        _note_job_done()


@app.post("/api/enhance")
async def enhance_only(
    file: UploadFile = File(...),
    provider: str = Form("gemini"),
    api_key: str = Form(""),
    prompt: str = Form(""),
    model: str = Form(""),
    upscale: str = Form("false"),
    tiles: str = Form("1"),
    protect_dark: str = Form("false"),
    gpu_cap: str = Form("100"),
    watermark: str = Form(""), wm_place: str = Form("br"),
    wm_opacity: str = Form("50"), wm_size: str = Form("m"),
    wm_style: str = Form("clean"),
    credit: str = Form(""),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload an image file")
    from core.gpu_throttle import set_cap as _set_gpu_cap
    _set_gpu_cap(gpu_cap)
    if provider != "local" and not api_key:
        raise HTTPException(400, "api_key is required for AI scan providers "
                                 "(pick 'Local — accurate' for no-key scanning)")

    task_id = str(uuid.uuid4())
    ext = Path(file.filename or "img.png").suffix or ".png"
    upload_path = f"uploads/{task_id}{ext}"
    output_path = f"output/{task_id}_scan.png"

    # Scan sends the page to the AI (which resizes it itself), so keep the upload
    # at full quality — don't pre-compress it.
    content = compress_upload(await file.read(), full=True)
    with open(upload_path, "wb") as f:
        f.write(content)

    tasks[task_id] = {
        "status": "processing",
        "step": 1,
        "message": "Queued",
        "progress": 0,
        "upload_path": upload_path,
        "name": file.filename or "page.png",
        "mode": "enhance",
    }

    try:
        n_tiles = int(tiles)
    except (TypeError, ValueError):
        n_tiles = 1
    asyncio.create_task(
        _run_enhance(task_id, upload_path, output_path, provider, api_key, prompt,
                     model, upscale=(upscale == "true"), tiles=n_tiles,
                     protect_dark=(protect_dark == "true"),
                     wm=dict(watermark=watermark.strip(),
                             wm_place=wm_place.strip() or "br",
                             wm_opacity=int(wm_opacity) if str(wm_opacity).strip().isdigit() else 50,
                             wm_size=wm_size.strip() or "m",
                             wm_style=wm_style.strip() or "clean",
                             credit=credit.strip()))
    )
    return {"task_id": task_id}


async def _run_enhance(
    task_id: str,
    image_path: str,
    output_path: str,
    provider: str,
    api_key: str,
    prompt: str,
    model: str,
    upscale: bool = False,
    tiles: int = 1,
    protect_dark: bool = False,
    wm: dict = None,
):
    try:
        tasks[task_id].update(
            {"step": 1, "progress": 5,
             "message": "Preprocessing image (crop + clean)..."}
        )
        enhancer = ImageEnhancer()

        def do_work():
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Cannot load image: {image_path}")
            tasks[task_id].update(
                {"progress": 30,
                 "message": f"Sending to {provider.title()} (this can take 30-60s)..."}
            )
            ai_ok = False
            try:
                if provider == "local":
                    # LOCAL accurate scan: 100% deterministic — deskew/crop,
                    # flatten lighting, denoise, white paper. The art is never
                    # redrawn: no hallucination, no restyle, no tile seams,
                    # always 1:1 faithful to the page. No API key needed.
                    tasks[task_id].update(
                        {"progress": 40,
                         "message": "Cleaning locally (accurate — no AI redraw)..."})
                    out = scan_cleanup(img)
                    ai_ok = True    # apply the crisp B&W finish below
                elif tiles >= 2:
                    # BETA tile mode: split the page, AI-scan each piece at full
                    # quality (beats the ~2K cap), then merge into a high-res page.
                    def _tp(n, total):
                        tasks[task_id].update(
                            {"progress": 30 + int(40 * n / max(total, 1)),
                             "message": f"AI scanning tile {n + 1}/{total} (Beta {tiles})..."})
                    out = enhancer.enhance_tiled(img, prompt, provider, api_key, model,
                                                 tiles=tiles, progress=_tp)
                    tasks[task_id].update({"progress": 70, "message": "AI enhancement complete!"})
                else:
                    # Send the RAW page to the AI scanner — like pasting it into Grok.
                    out = enhancer.enhance(img, prompt, provider, api_key, model)
                    tasks[task_id].update({"progress": 70, "message": "AI enhancement complete!"})
                ai_ok = True
            except Exception as e:
                print(f"[enhance] AI step failed, using local scan cleanup: {e}")
                tasks[task_id].update(
                    {"progress": 70,
                     "message": f"AI failed ({type(e).__name__}); used local clean scan"}
                )
                out = scan_cleanup(img)
            if ai_ok:
                # Grok caps at ~1-2K and returns generative grain / a colour tint,
                # so snap to a clean B&W scan: desaturate (manga is B&W) + paper→
                # white, ink→black, screentones kept grey. This is what removes the
                # grainy washed-out look.
                try:
                    from core.pipeline import scan_finish
                    out = scan_finish(cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY),
                                                   cv2.COLOR_GRAY2BGR))
                except Exception as e:
                    print(f"[enhance] crisp finish skipped: {e}")
                if protect_dark and provider != "local":
                    # OPT-IN: restore large panels that were dark in the source
                    # but came back white (inverted/splash panels the generative
                    # model re-drew in normal polarity).
                    try:
                        from core.pipeline import protect_dark_panels
                        out = protect_dark_panels(out, img)
                    except Exception as e:
                        print(f"[enhance] dark-panel guard skipped: {e}")
            # HD upscale (MangaJaNai) ONLY when the toggle is on — off by default.
            if upscale:
                tasks[task_id].update({"progress": 85,
                                       "message": "Upscaling to HD (MangaJaNai)..."})
                try:
                    from core.upscale import Upscaler
                    up = Upscaler()
                    if up.ok:
                        out = up.upscale(out, target_long=3600)
                    else:
                        print("[enhance] HD upscale requested but no model installed")
                except Exception as e:
                    print(f"[enhance] HD upscale step failed: {e}")
            cv2.imwrite(output_path, out)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, do_work)
        if wm and (wm.get("watermark") or wm.get("credit")):
            _stamp_all(output_path, wm["watermark"], wm["wm_place"],
                       wm["wm_opacity"], wm["wm_size"], wm["credit"],
                       wm.get("wm_style", "clean"))

        tasks[task_id].update(
            {
                "status": "done",
                "step": 2,
                "progress": 100,
                "message": "Manga scan ready!",
                "result": {"output_path": output_path, "translations": {}},
                "output_url": f"/api/result/{task_id}",
                "original_url": f"/api/original/{task_id}",
            }
        )
    except Exception as e:
        if task_id in tasks:
            tasks[task_id].update(
                {"status": "error", "message": str(e), "progress": 0}
            )
    finally:
        _note_job_done()


@app.post("/api/upscale")
async def upscale_only(file: UploadFile = File(...),
                       gpu_cap: str = Form("100"),
                       watermark: str = Form(""), wm_place: str = Form("br"),
                       wm_opacity: str = Form("50"), wm_size: str = Form("m"),
                       wm_style: str = Form("clean"),
                       credit: str = Form("")):
    """Faithful HD upscale only — no translation, no generative redraw. Runs
    the MangaJaNai (or Real-ESRGAN fallback) model and returns the bigger,
    sharper page with the art preserved exactly."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload an image file")
    from core.gpu_throttle import set_cap as _set_gpu_cap
    _set_gpu_cap(gpu_cap)

    task_id = str(uuid.uuid4())
    ext = Path(file.filename or "img.png").suffix or ".png"
    upload_path = f"uploads/{task_id}{ext}"
    output_path = f"output/{task_id}_hd.png"

    content = compress_upload(await file.read())
    with open(upload_path, "wb") as f:
        f.write(content)

    tasks[task_id] = {
        "status": "processing",
        "step": 1,
        "message": "Queued",
        "progress": 0,
        "upload_path": upload_path,
        "name": file.filename or "page.png",
        "mode": "upscale",
    }

    wm = dict(watermark=watermark.strip(), wm_place=wm_place.strip() or "br",
              wm_opacity=int(wm_opacity) if str(wm_opacity).strip().isdigit() else 50,
              wm_size=wm_size.strip() or "m", credit=credit.strip(),
              wm_style=wm_style.strip() or "clean")
    asyncio.create_task(_run_upscale(task_id, upload_path, output_path, wm))
    return {"task_id": task_id}


@app.post("/api/rawify")
async def rawify_only(file: UploadFile = File(...),
                      strength: str = Form("1.0"),
                      style: str = Form("photo"),
                      watermark: str = Form(""), wm_place: str = Form("br"),
                      wm_opacity: str = Form("50"), wm_size: str = Form("m"),
                      wm_style: str = Form("clean"),
                      credit: str = Form("")):
    """Scan → Raw: make a clean page look like a rough magazine raw (tan
    paper, grain, vignette, dust). Pure deterministic CV — no models, no
    translation, art and lettering untouched."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload an image file")
    task_id = str(uuid.uuid4())
    ext = Path(file.filename or "img.png").suffix or ".png"
    upload_path = f"uploads/{task_id}{ext}"
    output_path = f"output/{task_id}_raw.png"
    content = compress_upload(await file.read())
    with open(upload_path, "wb") as f:
        f.write(content)
    tasks[task_id] = {
        "status": "processing", "step": 1, "message": "Queued", "progress": 0,
        "upload_path": upload_path, "name": file.filename or "page.png",
        "mode": "rawify",
    }
    try:
        stren = float(strength)
    except (TypeError, ValueError):
        stren = 1.0
    wstyle = style if style in ("photo", "scan") else "photo"
    wm = dict(watermark=watermark.strip(), wm_place=wm_place.strip() or "br",
              wm_opacity=int(wm_opacity) if str(wm_opacity).strip().isdigit() else 50,
              wm_size=wm_size.strip() or "m", credit=credit.strip(),
              wm_style=wm_style.strip() or "clean")
    asyncio.create_task(_run_rawify(task_id, upload_path, output_path,
                                    stren, wstyle, wm))
    return {"task_id": task_id}


async def _run_rawify(task_id: str, image_path: str, output_path: str,
                      strength: float = 1.0, style: str = "photo",
                      wm: dict = None):
    try:
        tasks[task_id].update({"step": 1, "progress": 30,
                               "message": "Roughing the paper..."})

        def do_work():
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Cannot load image: {image_path}")
            out = raw_scan(img, strength=strength, style=style)
            cv2.imwrite(output_path, out)
            return out.shape

        shape = await asyncio.get_event_loop().run_in_executor(None, do_work)
        if wm and (wm.get("watermark") or wm.get("credit")):
            _stamp_all(output_path, wm["watermark"], wm["wm_place"],
                       wm["wm_opacity"], wm["wm_size"], wm["credit"],
                       wm.get("wm_style", "clean"))
        tasks[task_id].update({
            "status": "done", "step": 2, "progress": 100,
            "message": f"Raw look ready! ({shape[1]}×{shape[0]})",
            "result": {"output_path": output_path, "translations": {}},
            "output_url": f"/api/result/{task_id}",
            "original_url": f"/api/original/{task_id}",
        })
    except Exception as e:
        if task_id in tasks:
            tasks[task_id].update(
                {"status": "error", "message": str(e), "progress": 0})
    finally:
        _note_job_done()


async def _run_upscale(task_id: str, image_path: str, output_path: str,
                       wm: dict = None):
    try:
        from core.upscale import Upscaler
        tasks[task_id].update({"step": 1, "progress": 10,
                               "message": "Loading manga upscaler..."})

        def do_work():
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Cannot load image: {image_path}")
            up = Upscaler()
            if not up.ok:
                raise RuntimeError("No upscale model installed — run "
                                   "./setup_gpu.sh --mangajanai")
            tasks[task_id].update({"progress": 35,
                                   "message": "Upscaling to HD (faithful, keeps art)..."})
            out = up.upscale(img, target_long=3600)
            cv2.imwrite(output_path, out)
            return out.shape

        loop = asyncio.get_event_loop()
        shape = await loop.run_in_executor(None, do_work)
        if wm and (wm.get("watermark") or wm.get("credit")):
            _stamp_all(output_path, wm["watermark"], wm["wm_place"],
                       wm["wm_opacity"], wm["wm_size"], wm["credit"],
                       wm.get("wm_style", "clean"))
        tasks[task_id].update({
            "status": "done",
            "step": 2,
            "progress": 100,
            "message": f"HD upscale ready! ({shape[1]}×{shape[0]})",
            "result": {"output_path": output_path, "translations": {}},
            "output_url": f"/api/result/{task_id}",
            "original_url": f"/api/original/{task_id}",
        })
    except Exception as e:
        if task_id in tasks:
            tasks[task_id].update(
                {"status": "error", "message": str(e), "progress": 0}
            )
    finally:
        _note_job_done()


@app.post("/api/endcard")
async def end_card(
    scanlation: str = Form("Kaisuki"),
    discord: str = Form(""),
    show_discord: str = Form("true"),
    style: str = Form("royal"),
    theme: str = Form(""),
    accent: str = Form(""),
    heading: str = Form("THANK YOU FOR READING"),
    kicker: str = Form("END OF CHAPTER"),
    footer: str = Form("Please support the official release"),
    width: int = Form(1200),
    height: int = Form(1700),
):
    """Generate a one-click 'thank you for reading' end page for a chapter.
    Discord is optional (leave blank to omit it); heading/kicker/footer let the
    user put a custom message. No upload or API key needed."""
    from core.endcard import make_end_card
    # Explicit opt-out wins regardless of what's typed in the discord field, so
    # "hide Discord" always produces a clean page.
    if show_discord.strip().lower() not in ("true", "1", "yes", "on"):
        discord = ""
    task_id = str(uuid.uuid4())
    output_path = f"output/{task_id}_end.png"
    try:
        img = make_end_card(
            scanlation=scanlation, discord=discord, style=style, theme=theme,
            accent=accent, heading=heading, kicker=kicker, footer=footer,
            width=width, height=height,
        )
        cv2.imwrite(output_path, img)
    except Exception as e:
        raise HTTPException(500, f"Could not build end page: {e}")

    tasks[task_id] = {
        "status": "done",
        "step": 2,
        "progress": 100,
        "message": "End page ready!",
        "name": "end-page.png",
        "mode": "endcard",
        "result": {"output_path": output_path, "base_path": output_path,
                   "translations": {}, "items": []},
        "output_url": f"/api/result/{task_id}",
        "original_url": f"/api/original/{task_id}",
    }
    return {"task_id": task_id}


_IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _collect_page_refs(blobs):
    """Build a sorted list of page references from uploads, expanding any ZIPs,
    WITHOUT decoding. Each ref is (sortkey, reader) where reader yields raw bytes
    on demand — so a 30-chapter ZIP only decodes the pages we actually study.
    ZipFile handles are kept open for the lifetime of the returned refs."""
    refs = []
    for name, data in blobs:
        low = (name or "").lower()
        if low.endswith(".zip") or data[:2] == b"PK":
            try:
                zf = zipfile.ZipFile(io.BytesIO(data))
            except Exception as e:
                print(f"[profile] bad zip {name}: {e}")
                continue
            for zi in zf.namelist():
                if zi.endswith("/") or "__MACOSX" in zi:
                    continue
                if zi.lower().endswith(_IMG_EXT):
                    refs.append((zi, (zf, zi)))
            continue
        if low.endswith(_IMG_EXT) or data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n":
            refs.append((name, (None, data)))

    refs.sort(key=lambda r: r[0])
    return refs


def _decode_ref(ref):
    zf, payload = ref
    try:
        data = zf.read(payload) if zf is not None else payload
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def _sample_evenly(items, n):
    if n <= 0 or len(items) <= n:
        return items
    step = len(items) / float(n)
    return [items[int(i * step)] for i in range(n)]


def _chunk(items, n):
    return [items[i:i + n] for i in range(0, len(items), n)]


@app.get("/api/profiles")
async def profiles_list():
    from core import profiles
    return {"profiles": profiles.list_profiles()}


@app.get("/api/profile/{slug}")
async def profile_get(slug: str):
    from core import profiles
    p = profiles.load(slug)
    if not p:
        raise HTTPException(404, "Profile not found")
    return p


@app.post("/api/profile/{slug}")
async def profile_save(slug: str, request: Request):
    """Save an edited profile (the review step)."""
    from core import profiles
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")
    saved = profiles.save(profiles.normalize(body))
    return saved


@app.delete("/api/profile/{slug}")
async def profile_delete(slug: str):
    from core import profiles
    return {"deleted": profiles.delete(slug)}


@app.post("/api/profile/learn")
async def profile_learn(
    name: str = Form(...),
    provider: str = Form("claude"),
    api_key: str = Form(""),
    model: str = Form(""),
    target_lang: str = Form("English"),
    source_lang: str = Form("Japanese"),
    study_all: str = Form("false"),
    files: list[UploadFile] = File(...),
):
    """Learn (or enrich) a series profile from already-translated chapter pages.
    Accepts loose images and/or ZIPs (drop in 10-30 chapters at once). Samples a
    representative spread across ALL the pages, studies them in batches, and
    merges everything learned into the profile's glossary + house style."""
    if not api_key:
        raise HTTPException(400, "api_key is required to learn a profile")
    if not name.strip():
        raise HTTPException(400, "A series name is required")

    blobs = [(f.filename or "page.png", await f.read()) for f in files]
    refs = _collect_page_refs(blobs)
    total = len(refs)
    if not total:
        raise HTTPException(400, "No readable images found in the upload")

    # Study a spread across the whole upload. By default cap at 100 pages to keep
    # the cost sane; "study_all" removes the cap and reads every page (slower,
    # more thorough — best for understanding a big series).
    n_study = total if study_all == "true" else min(total, 100)
    sample = _sample_evenly(refs, n_study)
    images = [im for im in (_decode_ref(r[1]) for r in sample) if im is not None]
    if not images:
        raise HTTPException(400, "Could not decode any of the uploaded pages")

    from core import profiles
    from core.translator import make_translator

    def work():
        translator = make_translator(provider, api_key, model,
                                     source_lang=source_lang)
        prof = profiles.load(name)
        studied = 0
        for batch in _chunk(images, 8):
            learned = translator.analyze_pages(batch, target_lang)
            prof = profiles.merge_learned(prof, learned, name,
                                          added_sources=len(batch))
            studied += len(batch)
        if prof is None:
            raise RuntimeError("no pages were studied")
        return profiles.save(prof), studied

    try:
        loop = asyncio.get_event_loop()
        prof, studied = await loop.run_in_executor(None, work)
    except Exception as e:
        raise HTTPException(500, f"Learning failed: {e}")
    return {"profile": prof, "pages_seen": total, "pages_studied": studied}


@app.get("/api/status/{task_id}")
async def status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")
    return tasks[task_id]


_NO_CACHE = {"Cache-Control": "no-store, must-revalidate"}


@app.get("/api/result/{task_id}")
async def result(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404)
    t = tasks[task_id]
    if t["status"] != "done":
        raise HTTPException(400, "Not ready")
    p = t["result"]["output_path"]
    if not os.path.exists(p):
        raise HTTPException(404)
    mt = "image/jpeg" if p.lower().endswith((".jpg", ".jpeg")) else "image/png"
    return FileResponse(p, media_type=mt, headers=_NO_CACHE)


@app.get("/api/original/{task_id}")
async def original(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404)
    t = tasks[task_id]
    # Prefer the processed base (same dimensions as the result, so the
    # before/after comparison aligns perfectly); fall back to the upload.
    p = (t.get("result") or {}).get("base_path", "")
    if not p or not os.path.exists(p):
        p = t.get("upload_path", "")
    if not p or not os.path.exists(p):
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


@app.post("/api/rerender/{task_id}")
async def rerender(task_id: str, request: Request):
    _note_activity()   # editing keeps the models loaded
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")
    t = tasks[task_id]
    r = t.get("result") or {}
    # For a Clean page, re-render builds on the already-cleaned base (so erasing
    # a leftover doesn't bring the removed text back); otherwise the normal base.
    base = r.get("clean_base_path") or r.get("base_path", "")
    if not base or not os.path.exists(base):
        raise HTTPException(400, "This page can't be re-rendered")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")
    excluded = {str(i) for i in payload.get("excluded", [])}
    erased = {str(i) for i in payload.get("erased", [])}
    glows = {str(i) for i in payload.get("glows", [])}
    # "Fit box": grow this region's text until it really fills its box,
    # hyphenating a word too long for the column instead of letting that one
    # word cap the whole block. Opt-in per region.
    fits = {str(i) for i in payload.get("fits", [])}
    raw_effect = bool(payload.get("raw_effect", False))
    try:
        raw_strength = float(payload.get("raw_strength", 1.0))
    except (TypeError, ValueError):
        raw_strength = 1.0
    raw_style = payload.get("raw_style")
    raw_style = raw_style if raw_style in ("photo", "scan") else "photo"
    edits = {str(k): v for k, v in (payload.get("edits") or {}).items()}
    font_scale = float(payload.get("font_scale") or 1.0)
    offsets = {str(k): v for k, v in (payload.get("offsets") or {}).items()}
    covers = payload.get("covers") or []
    colors = {str(k): v for k, v in (payload.get("colors") or {}).items()}
    font_scales = {str(k): v for k, v in (payload.get("font_scales") or {}).items()}
    boxes = {str(k): v for k, v in (payload.get("boxes") or {}).items()}
    rotations = {}
    for k, v in (payload.get("rotations") or {}).items():
        try:
            # Full ±180: that covers every orientation (beyond 180 just wraps
            # around), so upside-down and any diagonal are all reachable. The
            # old ±90 cap silently straightened anything past it.
            rotations[str(k)] = max(-180.0, min(180.0, float(v)))
        except (TypeError, ValueError):
            pass

    def _scale(nid):
        try:
            return max(0.4, min(float(font_scales.get(nid, 1.0)), 3.0))
        except (TypeError, ValueError):
            return 1.0

    def _bbox(nid, default):
        b = boxes.get(nid)
        if b and len(b) == 4:
            try:
                x, y, bw, bh = (int(v) for v in b)
                if bw >= 6 and bh >= 6:
                    return [x, y, bw, bh]
            except (TypeError, ValueError):
                pass
        return default

    items = []
    for it in r.get("items", []):
        nid = str(it["id"])
        text = edits.get(nid, it.get("translation", ""))
        if nid in excluded:
            text = ""
        # Marked "erase" in the editor: wipe the region from the art and place
        # no text (for a watermark/garbage region the AI typeset by mistake).
        if nid in erased:
            items.append({
                "id": it["id"], "bbox": _bbox(nid, it["bbox"]), "original": it.get("original", ""),
                "translation": "", "type": "watermark", "erase": True,
                "in_bubble": it.get("in_bubble", True), "dark": it.get("dark", False),
                "rotation": it.get("rotation", 0),
            })
            continue
        items.append({
            "id": it["id"],
            "bbox": _bbox(nid, it["bbox"]),
            "original": it.get("original", ""),
            "translation": text,
            "type": it.get("type", ""),
            "in_bubble": it.get("in_bubble", True),
            "dark": it.get("dark", False),
            "src_rect": it.get("src_rect"),
            "title_caption": it.get("title_caption", False),
            "color": colors.get(nid, "auto"),
            "rotation": rotations.get(nid, it.get("rotation", 0)),
            "manual_rot": nid in rotations,
            "font_scale": _scale(nid),
            "glow": nid in glows,
            "fit_box": nid in fits,
            # A box the user resized by hand is authoritative — the compositor
            # must use it as-is (erase + fit text to it), not re-shrink it.
            "manual_box": nid in boxes,
        })

    # Manually added text regions (drawn over missed / leftover spots).
    added = []
    for a in (payload.get("added") or []):
        bbox = a.get("bbox")
        if not bbox:
            continue
        text = (a.get("translation") or "").strip()
        aid = str(a.get("id", f"m{len(added) + 1}"))
        poly = a.get("poly")
        if not (isinstance(poly, list) and len(poly) >= 3):
            poly = None
        # ⌫ on a DRAWN box: pure erase region. This never worked before —
        # added items skipped the `erased` check entirely, and a drawn box
        # without a translation was silently dropped, so "erase" on a manual
        # box did nothing at all.
        if aid in erased:
            added.append({
                "id": aid, "bbox": _bbox(aid, [int(v) for v in bbox]),
                "original": a.get("original", ""), "translation": "",
                "type": "watermark", "erase": True, "in_bubble": False,
                "poly": poly, "dark": False, "rotation": 0,
            })
            continue
        if not text:
            continue
        added.append({
            "id": aid,
            "bbox": _bbox(aid, [int(v) for v in bbox]),
            "original": a.get("original", ""),
            "translation": text,
            "type": "manual",
            "in_bubble": False,
            "manual": True,
            "poly": poly,      # point-selected outline: text stays inside it
            "rotation": rotations.get(aid, 0),
            "manual_rot": aid in rotations,
            "color": colors.get(aid, "auto"),
            "font_scale": _scale(aid),
            "fit_box": aid in fits,
        })

    all_items = items + added

    def work():
        from core.pipeline import scan_finish
        base_img = cv2.imread(base)
        if base_img is None:
            raise ValueError("Base image missing")
        # "Remove BG" outlines: keep inside the polygon(s), white out everything
        # outside. Kept separate from erase covers so compose ignores them.
        keep_polys = [c["keep_poly"] for c in covers
                      if isinstance(c, dict) and c.get("keep_poly")]
        restore_polys = [c["restore_poly"] for c in covers
                         if isinstance(c, dict) and c.get("restore_poly")]
        restore_clicks = [c["restore_click"] for c in covers
                          if isinstance(c, dict) and c.get("restore_click")]
        erase_covers = [c for c in covers
                        if not (isinstance(c, dict)
                                and (c.get("keep_poly") or c.get("restore_poly")))]
        comp = Compositor(t.get("font_path"), font_scale=font_scale,
                          uppercase=(t.get("text_case", "upper") != "keep"),
                          translate_sfx=bool(t.get("translate_sfx", False)),
                          replace_watermark=bool(t.get("replace_watermark", False)),
                          watermark_text=t.get("watermark", ""))
        out = comp.compose(base_img, all_items, MASKS.get(task_id), offsets, erase_covers)
        # Re-renders always keep the art surgical — same rule as the first
        # pass. "clean"/"api" get the local clean-scan finish; "off" keeps the
        # original pixels untouched. No generative repaint, ever.
        if t.get("finish", "clean") in ("clean", "api"):
            out = scan_finish(out)
        if keep_polys:
            h, w = out.shape[:2]
            mask = np.zeros((h, w), np.uint8)
            for poly in keep_polys:
                pts = np.array(poly, np.int32).reshape(-1, 2)
                if len(pts) >= 3:
                    cv2.fillPoly(mask, [pts], 255)
            if cv2.countNonZero(mask):
                out[mask == 0] = (255, 255, 255)   # background → white (no crop, keeps coords aligned)
        if restore_polys or restore_clicks:
            # Restore eraser: put the ORIGINAL pixels back. Two gestures:
            # - a drawn outline restores exactly that shape;
            # - a single CLICK finds the damaged blob under it automatically
            #   (diff vs the original page, take the connected changed region
            #   around the click) — one click un-deletes an eaten eye/detail.
            h, w = out.shape[:2]
            rmask = np.zeros((h, w), np.uint8)
            for poly in restore_polys:
                pts = np.array(poly, np.int32).reshape(-1, 2)
                if len(pts) >= 3:
                    cv2.fillPoly(rmask, [pts], 255)
            if restore_clicks:
                orig_p0 = r.get("base_path", "")
                src0 = cv2.imread(orig_p0) if orig_p0 and os.path.exists(orig_p0) else base_img
                if src0.shape[:2] != (h, w):
                    src0 = cv2.resize(src0, (w, h), interpolation=cv2.INTER_AREA)
                cmp_src = scan_finish(src0) if t.get("finish", "clean") in ("clean", "api") else src0
                diff = (np.abs(out.astype(np.int16) - cmp_src.astype(np.int16))
                        .max(axis=2) > 14).astype(np.uint8)
                diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE,
                                        np.ones((7, 7), np.uint8))
                nl, labels = cv2.connectedComponents(diff, 8)
                for cx, cy in restore_clicks:
                    cx = int(np.clip(int(cx), 0, w - 1))
                    cy = int(np.clip(int(cy), 0, h - 1))
                    lab = int(labels[cy, cx])
                    if lab == 0:
                        # click landed just off the blob — search nearby
                        y0, y1 = max(0, cy - 40), min(h, cy + 40)
                        x0, x1 = max(0, cx - 40), min(w, cx + 40)
                        win = labels[y0:y1, x0:x1]
                        vals = win[win > 0]
                        lab = int(np.bincount(vals).argmax()) if vals.size else 0
                    if lab > 0:
                        rmask |= cv2.dilate((labels == lab).astype(np.uint8) * 255,
                                            np.ones((9, 9), np.uint8))
                    else:
                        cv2.circle(rmask, (cx, cy), 30, 255, -1)
            if cv2.countNonZero(rmask):
                orig_p = r.get("base_path", "")
                src = cv2.imread(orig_p) if orig_p and os.path.exists(orig_p) else None
                if src is None:
                    src = base_img
                if src.shape[:2] != (h, w):
                    src = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)
                if t.get("finish", "clean") in ("clean", "api"):
                    src = scan_finish(src)
                rm = rmask > 0
                out[rm] = src[rm]
        if raw_effect:
            out = raw_scan(out, strength=raw_strength, style=raw_style)
        _write_atomic(r["output_path"], out)
        wm = t.get("watermark", "")
        if wm:
            _stamp_watermark(r["output_path"], wm,
                             t.get("wm_place", "br"), t.get("wm_opacity", 50),
                             t.get("wm_size", "m"), t.get("wm_style", "clean"))

    # Serialise per page: two Applies landing together used to write the same
    # file at once and corrupt each other's output.
    async with _task_lock(task_id):
        await asyncio.get_event_loop().run_in_executor(None, work)

    # Reflect new placement / edits back into the stored result.
    r["items"] = [
        {
            "id": it["id"], "bbox": it["bbox"], "original": it["original"],
            "translation": it["translation"], "type": it["type"],
            "in_bubble": it["in_bubble"], "dark": it.get("dark", False),
            "placed": it.get("placed", False),
            "rotation": it.get("rotation", 0),
            "src_rect": it.get("src_rect"),
            "title_caption": it.get("title_caption", False),
        }
        for it in items
    ]
    r["added"] = [
        {
            "id": it["id"], "bbox": it["bbox"], "translation": it["translation"],
            "original": it.get("original", ""), "placed": it.get("placed", False),
        }
        for it in added
    ]
    r["covers"] = covers
    r["translations"] = {
        str(it["id"]): {
            "original": it["original"], "translation": it["translation"], "type": it["type"]
        }
        for it in items
    }
    r["num_translated"] = sum(1 for it in all_items if it.get("placed"))

    return {"items": r["items"], "added": r["added"], "ts": time.time()}


@app.post("/api/rescan/{task_id}")
async def rescan(task_id: str, request: Request):
    _note_activity()   # a re-scan runs the detectors — keep them loaded
    """One-click 'find missed text': re-run AI detection on the page and merge in
    any region that isn't already covered, keeping all existing translations and
    edits. Returns the newly found regions."""
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")
    t = tasks[task_id]
    r = t.get("result") or {}
    base = r.get("base_path", "")
    if not base or not os.path.exists(base):
        raise HTTPException(400, "This page can't be re-scanned")
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")
    api_key = (payload.get("api_key") or "").strip()
    if not api_key:
        raise HTTPException(400, "api_key is required")
    target_lang = payload.get("target_lang", "English")
    provider = payload.get("provider", "claude")
    model = payload.get("model", "")
    style_prompt = payload.get("style_prompt", "")

    def work():
        from core.pipeline import TranslationPipeline
        img = cv2.imread(base)
        if img is None:
            raise ValueError("Base image missing")
        pipe = TranslationPipeline(
            api_key=api_key, target_lang=target_lang, provider=provider, model=model,
            use_smart_detection=True,  # smart pass is the most thorough finder
            font_path=t.get("font_path"), style_prompt=style_prompt,
            text_case=t.get("text_case", "upper"), finish=t.get("finish", "clean"),
            source_lang=t.get("source_lang", "Japanese"),
            translate_sfx=bool(t.get("translate_sfx", False)),
            remove_watermark=bool(t.get("remove_watermark", True)),
            replace_watermark=bool(t.get("replace_watermark", False)),
            watermark_text=t.get("watermark", ""),
        )
        out_tmp = r.get("output_path") or base
        items, _ann, masks = pipe._smart_detect(img, out_tmp, lambda *a, **k: None)
        return items, masks

    try:
        items, masks = await asyncio.get_event_loop().run_in_executor(None, work)
    except Exception as e:
        raise HTTPException(500, f"Re-scan failed: {e}")

    from core.pipeline import _boxes_overlap
    existing = r.get("items", [])
    existing_boxes = [it["bbox"] for it in existing if it.get("bbox")]
    next_id = max([it["id"] for it in existing if isinstance(it.get("id"), int)] + [0]) + 1
    task_masks = MASKS.setdefault(task_id, {})

    fresh = []
    for it in items:
        b = it.get("bbox")
        tr = (it.get("translation") or "").strip()
        if not b or (not tr and not it.get("erase")):
            continue
        if any(_boxes_overlap(list(b), list(eb)) for eb in existing_boxes):
            continue  # already have a region here
        nid, orig_id = next_id, it.get("id")
        next_id += 1
        new_it = {
            "id": nid, "bbox": [int(v) for v in b],
            "original": it.get("original", ""), "translation": tr,
            "type": it.get("type", "dialogue"), "in_bubble": it.get("in_bubble", True),
            "dark": it.get("dark", False), "rotation": it.get("rotation", 0),
            "placed": False, "erase": bool(it.get("erase", False)),
        }
        existing.append(new_it)
        existing_boxes.append(b)
        if masks.get(orig_id) is not None:
            task_masks[nid] = masks[orig_id]
        fresh.append(new_it)

    r["items"] = existing
    r["translations"] = {
        str(it["id"]): {"original": it.get("original", ""),
                        "translation": it.get("translation", ""), "type": it.get("type", "")}
        for it in existing
    }
    r["num_regions"] = len(existing)
    return {"added_count": len(fresh), "added": fresh, "items": r["items"]}


_OCR_INSTANCE = None

def _get_ocr():
    global _OCR_INSTANCE
    if _OCR_INSTANCE is None:
        try:
            from core.ocr import MangaOCR
            _OCR_INSTANCE = MangaOCR()
        except Exception:
            pass
    return _OCR_INSTANCE


@app.post("/api/ocr-translate/{task_id}")
async def ocr_translate(task_id: str, request: Request):
    _note_activity()   # editing keeps the models loaded
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")
    t = tasks[task_id]
    r = t.get("result") or {}
    base = r.get("base_path", "")
    if not base or not os.path.exists(base):
        raise HTTPException(400, "Base image not available")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    bbox = payload.get("bbox")
    api_key = payload.get("api_key", "")
    provider = payload.get("provider", "claude")
    model = payload.get("model", "")
    target_lang = payload.get("target_lang", "English")
    style_prompt = payload.get("style_prompt", "")
    poly = payload.get("poly")   # optional point-selected outline (image coords)

    if not bbox or len(bbox) != 4:
        raise HTTPException(400, "bbox must be [x, y, w, h]")
    if not api_key and provider not in ("local", "offline"):
        raise HTTPException(400, "api_key is required")

    x, y, w, h = [int(v) for v in bbox]

    def work():
        from core.translator import make_translator
        img = cv2.imread(base)
        if img is None:
            raise ValueError("Cannot read base image")
        H, W = img.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return {"original": "", "translation": ""}

        crop = img[y0:y1, x0:x1]
        # Point-selected outline: white out everything OUTSIDE the polygon so
        # OCR / vision reads ONLY the text the user selected — neighbouring
        # lines can't bleed into the reading.
        if poly and isinstance(poly, list) and len(poly) >= 3:
            try:
                pm = np.zeros(crop.shape[:2], np.uint8)
                pts = np.array([[int(px) - x0, int(py) - y0] for px, py in poly],
                               np.int32)
                cv2.fillPoly(pm, [pts], 255)
                if cv2.countNonZero(pm):
                    crop = crop.copy()
                    crop[pm == 0] = (255, 255, 255)
            except Exception as e:
                print(f"[ocr-translate] poly mask skipped: {e}")
        translator = make_translator(provider, api_key, model, style_prompt,
                                     source_lang=t.get("source_lang", "Japanese"),
                                     translate_sfx=bool(t.get("translate_sfx", False)))

        # Vision read+translate — works for ANY language (Japanese, Arabic, …),
        # so weird-shaped / non-Japanese regions translate too.
        try:
            res = translator.translate_crop(crop, target_lang)
            if (res.get("translation") or "").strip():
                return {"original": res.get("original", ""),
                        "translation": res.get("translation", "")}
        except Exception as e:
            print(f"[ocr-translate] vision crop failed: {e}")

        # Fallback: local Japanese OCR + text translate.
        original = ""
        ocr = _get_ocr()
        if ocr and ocr.ok:
            padded = cv2.copyMakeBorder(crop, 12, 12, 12, 12,
                                        cv2.BORDER_CONSTANT, value=(255, 255, 255))
            original = ocr.read(padded)
        if not original:
            return {"original": "", "translation": ""}
        out = translator.translate_texts({"0": original}, target_lang, image=crop)
        entry = out.get(0) or out.get("0") or {}
        return {"original": original, "translation": entry.get("translation", original)}

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, work)
    return result


@app.post("/api/retranslate-ordered/{task_id}")
async def retranslate_ordered(task_id: str, request: Request):
    _note_activity()   # editing keeps the models loaded
    """Re-translate a page's bubbles in the order the USER put them in.

    The detector's guess at panel order is what makes lines answer the wrong
    bubble. Sending the source text in the reader's order — as one numbered
    conversation — gives the model the context it needs to get the
    back-and-forth right, without re-running detection or touching the art."""
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")
    t = tasks[task_id]
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    items = payload.get("items") or []
    api_key = payload.get("api_key", "")
    if not items:
        raise HTTPException(400, "No items to re-translate")
    if not api_key and payload.get("provider") not in ("local", "offline"):
        raise HTTPException(400, "api_key is required")

    provider = payload.get("provider", "claude")
    model = payload.get("model", "")
    target_lang = payload.get("target_lang", "English")
    source_lang = payload.get("source_lang", t.get("source_lang", "Japanese"))
    style_prompt = payload.get("style_prompt", "")

    # Sequential keys preserve the user's order for the model, while `back`
    # maps each one home again — the region ids themselves may be any value.
    ordered, back = {}, {}
    for n, it in enumerate(items, start=1):
        text = (it.get("original") or "").strip()
        if not text:
            continue
        ordered[str(n)] = text
        back[n] = str(it.get("id"))
    if not ordered:
        raise HTTPException(400, "None of those items carry source text")

    def work():
        from core.translator import make_translator
        translator = make_translator(
            provider, api_key, model, style_prompt, source_lang=source_lang,
            translate_sfx=bool(t.get("translate_sfx", False)),
            webtoon=bool(t.get("webtoon", False)))
        img = None
        r = t.get("result") or {}
        base = r.get("base_path", "")
        if base and os.path.exists(base):
            img = cv2.imread(base)          # panel context helps the wording
        out = translator.translate_texts(ordered, target_lang, image=img)
        res = []
        for n, rid in back.items():
            entry = out.get(n) or out.get(str(n)) or {}
            tr = (entry.get("translation") or "").strip()
            if tr:
                res.append({"id": rid, "translation": tr})
        return res

    loop = asyncio.get_event_loop()
    try:
        translations = await loop.run_in_executor(None, work)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return {"translations": translations, "count": len(translations)}


@app.post("/api/translate-text")
async def translate_text(request: Request):
    _note_activity()   # editing keeps the models loaded
    """Translate SOURCE text the user typed or pasted, with no image involved.

    Backs the built-in Japanese/Korean keyboard: when OCR can't read a bubble
    the user can key the original in by hand (or paste it) and get the same
    quality of translation the automatic path produces."""
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    text = (payload.get("text") or "").strip()
    api_key = payload.get("api_key", "")
    if not text:
        raise HTTPException(400, "Type some text to translate")
    if not api_key and payload.get("provider") not in ("local", "offline"):
        raise HTTPException(400, "api_key is required")

    provider = payload.get("provider", "claude")
    model = payload.get("model", "")
    target_lang = payload.get("target_lang", "English")
    source_lang = payload.get("source_lang", "Japanese")
    style_prompt = payload.get("style_prompt", "")

    def work():
        from core.translator import make_translator
        translator = make_translator(provider, api_key, model, style_prompt,
                                     source_lang=source_lang)
        out = translator.translate_texts({"0": text}, target_lang)
        entry = out.get(0) or out.get("0") or {}
        return {"original": text,
                "translation": (entry.get("translation") or "").strip()}

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, work)
    except RuntimeError as e:
        # e.g. the offline model isn't downloaded yet — that message is written
        # for the user, so show it instead of a bare 500.
        raise HTTPException(400, str(e))


@app.post("/api/check-orientation")
async def check_orientation(file: UploadFile = File(...),
                            source_lang: str = Form("Japanese")):
    """Is this page upside down?

    Reads a few of its balloons both ways up and reports which way produced
    real language. Runs entirely on this machine — no API call, nothing
    charged — so a whole chapter can be checked before a single page is
    translated.
    """
    import tempfile
    from core import orient
    suffix = os.path.splitext(file.filename or "")[1] or ".png"
    fd, tmp = tempfile.mkstemp(suffix=suffix, dir="uploads")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(await file.read())
        loop = asyncio.get_event_loop()
        verdict = await loop.run_in_executor(
            None, lambda: orient.check_file(tmp, source_lang))
    except Exception as e:
        print(f"[orient] check failed: {e}")
        verdict = {"upside_down": False, "sure": False,
                   "why": f"could not be checked ({e})", "up": 0, "down": 0}
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    verdict["name"] = file.filename or ""
    return JSONResponse(verdict)


@app.get("/api/wm-preview")
async def wm_preview(text: str = "@YourName", style: str = "clean",
                     place: str = "br", opacity: int = 70, size: str = "m",
                     credit: str = ""):
    """Live preview: renders the CURRENT watermark settings with the exact
    same code that stamps real pages, onto a sample manga-ish page."""
    w, h = 620, 420
    img = np.full((h, w, 3), 246, np.uint8)
    rng = np.random.default_rng(7)
    g = cv2.GaussianBlur((rng.random((h, w)).astype(np.float32) - .5) * 26, (0, 0), 3)
    img = np.clip(img.astype(np.float32) + g[..., None], 0, 255).astype(np.uint8)
    cv2.rectangle(img, (12, 12), (w - 12, h // 2 - 6), (0, 0, 0), 2)
    cv2.rectangle(img, (12, h // 2 + 6), (w // 2 - 6, h - 12), (0, 0, 0), 2)
    cv2.rectangle(img, (w // 2 + 6, h // 2 + 6), (w - 12, h - 12), (0, 0, 0), 2)
    for i in range(30, w - 30, 14):
        cv2.line(img, (i, 30), (i - 60, h // 2 - 20), (150, 150, 150), 1)
    cv2.ellipse(img, (150, h - 110), (85, 60), 0, 0, 360, (0, 0, 0), 2)
    cv2.putText(img, "SAMPLE", (95, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
    cv2.ellipse(img, (470, 300), (60, 45), 0, 0, 360, (30, 30, 30), -1)
    import tempfile
    fd, tmp = tempfile.mkstemp(suffix=".png", dir="output")
    os.close(fd)
    try:
        cv2.imwrite(tmp, img)
        _stamp_all(tmp, (text or "").strip()[:60],
                   (place or "br").strip() or "br",
                   int(np.clip(opacity, 5, 100)),
                   (size or "m").strip() or "m",
                   (credit or "").strip()[:60],
                   (style or "clean").strip() or "clean")
        out = cv2.imread(tmp)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    ok, png = cv2.imencode(".png", out if out is not None else img)
    return Response(content=png.tobytes(), media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@app.post("/api/stamp/{task_id}")
async def stamp_page(task_id: str, request: Request):
    """Stamp the watermark / credit onto ONE finished page, on demand — for
    outputs produced before the toggle was on, or when you only want a single
    page marked. Writes onto the current output in place."""
    t = tasks.get(task_id)
    if not t or t.get("status") != "done":
        raise HTTPException(404, "Page not ready")
    p = (t.get("result") or {}).get("output_path", "")
    if not p or not os.path.exists(p):
        raise HTTPException(404, "Output missing")
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    wmk = str(payload.get("watermark") or "").strip()
    cr = str(payload.get("credit") or "").strip()
    if not wmk and not cr:
        raise HTTPException(400, "Set a watermark or credit text first")
    place = str(payload.get("wm_place") or "br").strip() or "br"
    size = str(payload.get("wm_size") or "m").strip() or "m"
    try:
        op = int(payload.get("wm_opacity") or 50)
    except (TypeError, ValueError):
        op = 50
    _stamp_all(p, wmk, place, op, size, cr,
               str(payload.get("wm_style") or "clean").strip() or "clean")
    return {"ok": True}


@app.post("/api/zip")
async def make_zip(request: Request):
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")
    ids = payload.get("task_ids", [])
    # Optional chapter name: becomes the zip filename AND the page filenames
    # inside ("Name - 001.png"), so the download drops straight into a reader
    # or site upload without renaming.
    name = re.sub(r'[\\/:*?"<>|]+', "", str(payload.get("name") or "")).strip()
    name = name[:80]

    buf = io.BytesIO()
    count = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, tid in enumerate(ids, 1):
            t = tasks.get(tid)
            if not t or t.get("status") != "done":
                continue
            path = (t.get("result") or {}).get("output_path", "")
            if not path or not os.path.exists(path):
                continue
            stem = os.path.splitext(os.path.basename(t.get("name", f"page_{i}")))[0]
            arcname = (f"{name} - {i:03d}.png" if name
                       else f"{i:03d}_{stem}.png")
            # Re-encode to a REAL PNG so the .png name always matches the bytes —
            # output files may actually be JPEG/WebP (they keep the upload's
            # extension, and Compress re-encodes to .jpg), which broke opening the
            # "*.png" in Photoshop. Decode whatever it is, write true PNG.
            img = cv2.imread(path)
            if img is None:
                continue
            ok, png = cv2.imencode(".png", img)
            if not ok:
                continue
            zf.writestr(arcname, png.tobytes())
            count += 1

    if count == 0:
        raise HTTPException(400, "No finished pages to download yet")

    buf.seek(0)
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition":
                 f'attachment; filename="{(name or "translated_pages")}.zip"'},
    )


@app.post("/api/unzip")
async def unzip(file: UploadFile = File(...)):
    """Expand an uploaded ZIP of manga pages into individual images so the drop
    zone can accept a whole chapter as a .zip. Returns base64 images in name
    order; the frontend turns them back into files and queues them."""
    import base64 as _b64
    data = await file.read()
    images = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for zi in sorted(zf.namelist()):
                if zi.endswith("/") or "__MACOSX" in zi:
                    continue
                low = zi.lower()
                if low.endswith(_IMG_EXT):
                    ext = low.rsplit(".", 1)[-1].replace("jpg", "jpeg")
                    images.append({
                        "name": os.path.basename(zi),
                        "type": f"image/{ext}",
                        "b64": _b64.b64encode(zf.read(zi)).decode(),
                    })
    except Exception as e:
        raise HTTPException(400, f"Could not read ZIP: {e}")
    if not images:
        raise HTTPException(400, "No images found in the ZIP")
    return {"images": images}


@app.get("/api/enhanced/{task_id}")
async def enhanced(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404)
    p = tasks[task_id].get("enhanced_path", "")
    if not p or not os.path.exists(p):
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


@app.get("/api/enhance-prompt")
async def enhance_prompt():
    return {"prompt": ImageEnhancer.DEFAULT_PROMPT, "models": ImageEnhancer.DEFAULT_MODELS}


@app.get("/api/annotated/{task_id}")
async def annotated(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404)
    t = tasks[task_id]
    if t["status"] != "done":
        raise HTTPException(400)
    p = t.get("result", {}).get("annotated_path", "")
    if not p or not os.path.exists(p):
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


@app.post("/api/upload-font")
async def upload_font(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith((".ttf", ".otf")):
        raise HTTPException(400, "Upload a .ttf or .otf font file")
    dest = f"fonts/{file.filename}"
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)
    return {"message": f"Font '{file.filename}' uploaded", "path": dest}


@app.get("/api/fonts")
async def list_fonts():
    fonts = []
    for f in sorted(os.listdir("fonts")):
        if f.lower().endswith((".ttf", ".otf")):
            fonts.append(f)
    return {"fonts": fonts}


if __name__ == "__main__":
    import socket
    import time
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))

    def _port_busy() -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", port)) == 0

    # A just-killed server (pkill -f app.py) takes a moment to release the
    # port while uvicorn shuts down — wait for it instead of failing the
    # restart the user was told to do.
    if _port_busy():
        print(f"Port {port} is busy — waiting for the old server to exit", end="", flush=True)
        for _ in range(20):
            time.sleep(0.5)
            print(".", end="", flush=True)
            if not _port_busy():
                print(" freed.")
                break
        else:
            print(f"\n\nPort {port} is still in use after 10s — something else owns it.")
            print(f"  See what it is:         ss -tlnp | grep {port}")
            print(f"  Force-free the port:    fuser -k {port}/tcp")
            print(f"  Or use another port:    PORT={port + 1} python3 app.py\n")
            raise SystemExit(1)

    # The frontend polls /api/status every second — hundreds of identical
    # access-log lines drown the real progress messages and make a working
    # server look stuck. Keep every other request in the log.
    import logging

    class _QuietPolls(logging.Filter):
        def filter(self, record):
            m = record.getMessage()
            return "/api/status/" not in m and "/api/wm-preview" not in m

    logging.getLogger("uvicorn.access").addFilter(_QuietPolls())

    uvicorn.run(app, host="0.0.0.0", port=port)
