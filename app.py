import asyncio
import io
import os
import time
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


def _stamp_watermark(image_path: str, text: str, place: str = "br",
                     opacity: int = 50, size: str = "m"):
    """Watermark the finished page. `place` puts the text ONCE in a corner
    ('br','bl','tr','tl' — the default, discreet look) or tiles it diagonally
    across the whole page ('tile'). `opacity` is 0–100%; `size` is
    's' / 'm' / 'l'."""
    img = cv2.imread(image_path)
    if img is None:
        return
    h, w = img.shape[:2]
    alpha = int(np.clip(int(opacity or 50), 5, 100) * 255 / 100)
    # Corner divisors: l = the old size, m/s progressively smaller.
    corner_div = {"l": 42, "m": 58, "s": 78}.get(size, 58)
    tile_div = {"l": 28, "m": 38, "s": 50}.get(size, 38)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    def _font(size):
        try:
            return ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        except Exception:
            return ImageFont.load_default()

    if place == "tile":
        font = _font(max(16, min(w, h) // tile_div))
        bb = draw.textbbox((0, 0), text, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        step_x = tw + max(80, tw)
        step_y = th + max(120, th * 3)
        for y in range(-h, h * 2, step_y):
            for x in range(-w, w * 2, step_x):
                draw.text((x, y), text, fill=(128, 128, 128, alpha), font=font)
        overlay = overlay.rotate(30, expand=False, center=(w // 2, h // 2))
    else:
        font = _font(max(10, min(w, h) // corner_div))
        bb = draw.textbbox((0, 0), text, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        m = max(10, min(w, h) // 60)
        if place == "random":
            # Fresh roll on every render — hit Apply again to move it, or
            # switch to a fixed corner. Samples a dozen spots and takes the
            # QUIETEST one, so the stamp lands on flat art or paper instead
            # of across a face or lettering.
            gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            x_lo, x_hi = int(0.06 * w), max(int(0.94 * w) - tw, int(0.06 * w) + 1)
            y_lo, y_hi = int(0.06 * h), max(int(0.94 * h) - th, int(0.06 * h) + 1)
            best_xy, best_score = (m, m), 1e18
            for _ in range(12):
                cx = random.randint(x_lo, x_hi)
                cy = random.randint(y_lo, y_hi)
                reg = gray_full[cy:cy + th, cx:cx + tw]
                if reg.size == 0:
                    continue
                score = float(reg.std())
                if score < best_score:
                    best_score, best_xy = score, (cx, cy)
            x, y = best_xy
        else:
            x = (w - tw - m) if place in ("br", "tr") else m
            y = (h - th - m) if place in ("br", "bl") else m
        # Readable on whatever the corner holds: white text on dark art,
        # dark text on paper — with a thin contrasting outline either way.
        region = img[max(0, y - 4):min(h, y + th + 4), max(0, x - 4):min(w, x + tw + 4)]
        lum = float(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).mean()) if region.size else 255.0
        fill = (255, 255, 255, alpha) if lum < 128 else (25, 25, 25, alpha)
        stroke = (25, 25, 25, alpha) if lum < 128 else (255, 255, 255, alpha)
        draw.text((x - bb[0], y - bb[1]), text, font=font, fill=fill,
                  stroke_width=max(1, font.size // 14), stroke_fill=stroke)

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

LAST_JOB_TS = time.time()
_IDLE_TRIMMED = False


def _note_job_done():
    global LAST_JOB_TS, _IDLE_TRIMMED
    LAST_JOB_TS = time.time()
    _IDLE_TRIMMED = False
    memsweep.light_sweep()


async def _housekeeper():
    global LAST_JOB_TS, _IDLE_TRIMMED
    loop = asyncio.get_event_loop()
    while True:
        await asyncio.sleep(60)
        try:
            if any(t.get("status") == "processing" for t in tasks.values()):
                LAST_JOB_TS = time.time()
                continue
            idle = time.time() - LAST_JOB_TS
            if idle >= 1800:
                released = await loop.run_in_executor(None, memsweep.unload_models)
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
        request=request, name="index.html", context={"version": _asset_version()})


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
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip() or "unknown"
    except Exception:
        return "unknown"


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
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload an image file")
    from core.gpu_throttle import set_cap as _set_gpu_cap
    _set_gpu_cap(gpu_cap)
    is_clean = clean_only == "true"
    if not is_clean and not api_key:
        raise HTTPException(400, "api_key is required to translate")

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
    content = compress_upload(await file.read(), full=(max_quality == "true"))
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
        "translate_sfx": translate_sfx == "true",
        "max_quality": max_quality == "true",
        "remove_watermark": remove_watermark == "true",
        "replace_watermark": replace_watermark == "true",
    }

    asyncio.create_task(
        _run(
            task_id, upload_path, output_path, api_key, target_lang, provider, model,
            smart_mode == "true", font_path,
            enhance == "true", enhance_provider, enhance_key, enhance_prompt, enhance_model,
            watermark=watermark.strip(),
            wm_place=wm_place.strip() or "br",
            wm_opacity=int(wm_opacity) if str(wm_opacity).strip().isdigit() else 50,
            wm_size=wm_size.strip() or "m",
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
):
    try:
        loop = asyncio.get_event_loop()

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
        )

        def on_progress(update):
            tasks[task_id].update(update)

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
            _stamp_watermark(output_path, watermark, wm_place, wm_opacity, wm_size)

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
                     protect_dark=(protect_dark == "true"))
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
                       gpu_cap: str = Form("100")):
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

    asyncio.create_task(_run_upscale(task_id, upload_path, output_path))
    return {"task_id": task_id}


@app.post("/api/rawify")
async def rawify_only(file: UploadFile = File(...),
                      strength: str = Form("1.0"),
                      style: str = Form("photo")):
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
    asyncio.create_task(_run_rawify(task_id, upload_path, output_path,
                                    stren, wstyle))
    return {"task_id": task_id}


async def _run_rawify(task_id: str, image_path: str, output_path: str,
                      strength: float = 1.0, style: str = "photo"):
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


async def _run_upscale(task_id: str, image_path: str, output_path: str):
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
            rotations[str(k)] = max(-80.0, min(80.0, float(v)))
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
            "color": colors.get(nid, "auto"),
            "rotation": rotations.get(nid, it.get("rotation", 0)),
            "manual_rot": nid in rotations,
            "font_scale": _scale(nid),
            "glow": nid in glows,
            # A box the user resized by hand is authoritative — the compositor
            # must use it as-is (erase + fit text to it), not re-shrink it.
            "manual_box": nid in boxes,
        })

    # Manually added text regions (drawn over missed / leftover spots).
    added = []
    for a in (payload.get("added") or []):
        bbox = a.get("bbox")
        text = (a.get("translation") or "").strip()
        if not bbox or not text:
            continue
        aid = str(a.get("id", f"m{len(added) + 1}"))
        poly = a.get("poly")
        if not (isinstance(poly, list) and len(poly) >= 3):
            poly = None
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
        if restore_polys:
            # Restore eraser: put the ORIGINAL pixels back wherever the user
            # outlined — undoes content-aware fills that chewed the art. The
            # source is the pre-clean base (original art, original text); it
            # gets the same page finish as the rest so tones match seamlessly.
            h, w = out.shape[:2]
            rmask = np.zeros((h, w), np.uint8)
            for poly in restore_polys:
                pts = np.array(poly, np.int32).reshape(-1, 2)
                if len(pts) >= 3:
                    cv2.fillPoly(rmask, [pts], 255)
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
        cv2.imwrite(r["output_path"], out)
        wm = t.get("watermark", "")
        if wm:
            _stamp_watermark(r["output_path"], wm,
                             t.get("wm_place", "br"), t.get("wm_opacity", 50),
                             t.get("wm_size", "m"))

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
    if not api_key:
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


@app.post("/api/zip")
async def make_zip(request: Request):
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")
    ids = payload.get("task_ids", [])

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
            zf.writestr(f"{i:03d}_{stem}.png", png.tobytes())
            count += 1

    if count == 0:
        raise HTTPException(400, "No finished pages to download yet")

    buf.seek(0)
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="translated_pages.zip"'},
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

    uvicorn.run(app, host="0.0.0.0", port=port)
