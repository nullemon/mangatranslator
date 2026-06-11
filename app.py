import asyncio
import io
import os
import time
import uuid
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
                           preserve_dark_regions)
from core.compositor import Compositor
from core.enhancer import ImageEnhancer


def _stamp_watermark(image_path: str, text: str):
    """Render a repeating diagonal watermark at low opacity."""
    img = cv2.imread(image_path)
    if img is None:
        return
    h, w = img.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font_size = max(16, min(w, h) // 28)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    bb = draw.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    step_x = tw + max(80, tw)
    step_y = th + max(120, th * 3)
    alpha = 28
    for y in range(-h, h * 2, step_y):
        for x in range(-w, w * 2, step_x):
            draw.text((x, y), text, fill=(128, 128, 128, alpha), font=font)
    rotated = overlay.rotate(30, expand=False, center=(w // 2, h // 2))
    pil = pil.convert("RGBA")
    pil = Image.alpha_composite(pil, rotated)
    result = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, result)

app = FastAPI(title="MangaTranslator")

for d in ("uploads", "output", "fonts"):
    os.makedirs(d, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

tasks: dict = {}
# Balloon interior masks per task (numpy arrays — kept out of `tasks` so the
# JSON status endpoint stays serializable). Reused for clean re-renders.
MASKS: dict = {}


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/api/translate")
async def translate(
    file: UploadFile = File(...),
    api_key: str = Form(...),
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
    style_prompt: str = Form(""),
    text_case: str = Form("upper"),
    finish: str = Form("clean"),
    upscale: str = Form("false"),
    source_lang: str = Form("Japanese"),
    translate_sfx: str = Form("false"),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload an image file")

    task_id = str(uuid.uuid4())
    ext = Path(file.filename or "img.png").suffix or ".png"
    upload_path = f"uploads/{task_id}{ext}"
    output_path = f"output/{task_id}{ext}"

    content = compress_upload(await file.read())
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
        "text_case": text_case,
        "finish": finish,
        "enhance_provider": enhance_provider,
        "enhance_key": enhance_key,
        "enhance_model": enhance_model,
        "enhance_prompt": enhance_prompt,
        "name": file.filename or "page.png",
        "mode": "translate",
        "source_lang": source_lang,
        "translate_sfx": translate_sfx == "true",
    }

    asyncio.create_task(
        _run(
            task_id, upload_path, output_path, api_key, target_lang, provider, model,
            smart_mode == "true", font_path,
            enhance == "true", enhance_provider, enhance_key, enhance_prompt, enhance_model,
            watermark=watermark.strip(),
            style_prompt=style_prompt.strip(),
            text_case=text_case,
            finish=finish,
            upscale=(upscale == "true"),
            source_lang=source_lang,
            translate_sfx=(translate_sfx == "true"),
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
    style_prompt: str = "",
    text_case: str = "upper",
    finish: str = "clean",
    upscale: bool = False,
    source_lang: str = "Japanese",
    translate_sfx: bool = False,
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
                # is stretched and detection boxes stay aligned.
                if out.shape[:2] != img.shape[:2]:
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
        )

        def on_progress(update):
            tasks[task_id].update(update)

        result = await loop.run_in_executor(
            None,
            lambda: pipeline.process(translate_source, output_path, on_progress),
        )
        MASKS[task_id] = getattr(pipeline, "last_masks", {}) or {}

        # NOTE: the delivered translation is NEVER run through the generative
        # enhancer. A whole-page generative pass repaints the art — it redraws
        # hair, drops labels it doesn't understand, and bleaches screentone to
        # hard B&W — which violates the one rule: only the text is touched, the
        # art stays byte-for-byte as drawn. The "api" finish now just delivers
        # the clean-scanned surgical page (scan_finish, applied in the
        # pipeline). The generative model stays available as the explicit
        # "Enhance & Translate" workflow, where it produces a SEPARATE image.

        if watermark:
            _stamp_watermark(output_path, watermark)

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


@app.post("/api/enhance")
async def enhance_only(
    file: UploadFile = File(...),
    provider: str = Form("gemini"),
    api_key: str = Form(...),
    prompt: str = Form(""),
    model: str = Form(""),
    upscale: str = Form("false"),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload an image file")

    task_id = str(uuid.uuid4())
    ext = Path(file.filename or "img.png").suffix or ".png"
    upload_path = f"uploads/{task_id}{ext}"
    output_path = f"output/{task_id}_scan.png"

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
        "mode": "enhance",
    }

    asyncio.create_task(
        _run_enhance(task_id, upload_path, output_path, provider, api_key, prompt,
                     model, upscale=(upscale == "true"))
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
            tasks[task_id].update({"progress": 15, "message": "Cropping and cleaning page..."})
            cleaned = scan_cleanup(img)
            tasks[task_id].update(
                {"progress": 30,
                 "message": f"Sending to {provider.title()} (this can take 30-60s)..."}
            )
            ai_ok = False
            try:
                out = enhancer.enhance(cleaned, prompt, provider, api_key, model)
                ai_ok = True
                tasks[task_id].update({"progress": 70, "message": "AI enhancement complete!"})
            except Exception as e:
                print(f"[enhance] AI step failed, using local scan cleanup: {e}")
                tasks[task_id].update(
                    {"progress": 70,
                     "message": f"AI failed ({type(e).__name__}); used local clean scan"}
                )
                out = cleaned
            # Keep solid blacks the generative scan would otherwise bleach.
            if ai_ok:
                try:
                    out = preserve_dark_regions(out, cleaned)
                except Exception as e:
                    print(f"[enhance] dark-region preserve skipped: {e}")
            # Optional second stage: faithful HD upscale on top of the AI scan.
            if upscale:
                from core.upscale import Upscaler
                tasks[task_id].update({"progress": 85,
                                       "message": "Upscaling to HD (MangaJaNai)..."})
                up = Upscaler()
                if up.ok:
                    try:
                        out = up.upscale(out, target_long=3600)
                    except Exception as e:
                        print(f"[enhance] HD upscale step failed: {e}")
                else:
                    print("[enhance] HD upscale requested but no model installed")
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


@app.post("/api/upscale")
async def upscale_only(file: UploadFile = File(...)):
    """Faithful HD upscale only — no translation, no generative redraw. Runs
    the MangaJaNai (or Real-ESRGAN fallback) model and returns the bigger,
    sharper page with the art preserved exactly."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload an image file")

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
    return FileResponse(p, media_type="image/png", headers=_NO_CACHE)


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
    base = r.get("base_path", "")
    if not base or not os.path.exists(base):
        raise HTTPException(400, "This page can't be re-rendered")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")
    excluded = {str(i) for i in payload.get("excluded", [])}
    edits = {str(k): v for k, v in (payload.get("edits") or {}).items()}
    font_scale = float(payload.get("font_scale") or 1.0)
    offsets = {str(k): v for k, v in (payload.get("offsets") or {}).items()}
    covers = payload.get("covers") or []
    colors = {str(k): v for k, v in (payload.get("colors") or {}).items()}

    items = []
    for it in r.get("items", []):
        nid = str(it["id"])
        text = edits.get(nid, it.get("translation", ""))
        if nid in excluded:
            text = ""
        items.append({
            "id": it["id"],
            "bbox": it["bbox"],
            "original": it.get("original", ""),
            "translation": text,
            "type": it.get("type", ""),
            "in_bubble": it.get("in_bubble", True),
            "dark": it.get("dark", False),
            "color": colors.get(nid, "auto"),
            "rotation": it.get("rotation", 0),
        })

    # Manually added text regions (drawn over missed / leftover spots).
    added = []
    for a in (payload.get("added") or []):
        bbox = a.get("bbox")
        text = (a.get("translation") or "").strip()
        if not bbox or not text:
            continue
        aid = str(a.get("id", f"m{len(added) + 1}"))
        added.append({
            "id": aid,
            "bbox": [int(v) for v in bbox],
            "original": a.get("original", ""),
            "translation": text,
            "type": "manual",
            "in_bubble": False,
            "manual": True,
            "color": colors.get(aid, "auto"),
        })

    all_items = items + added

    def work():
        from core.pipeline import scan_finish
        base_img = cv2.imread(base)
        if base_img is None:
            raise ValueError("Base image missing")
        comp = Compositor(t.get("font_path"), font_scale=font_scale,
                          uppercase=(t.get("text_case", "upper") != "keep"),
                          translate_sfx=bool(t.get("translate_sfx", False)))
        out = comp.compose(base_img, all_items, MASKS.get(task_id), offsets, covers)
        # Re-renders always keep the art surgical — same rule as the first
        # pass. "clean"/"api" get the local clean-scan finish; "off" keeps the
        # original pixels untouched. No generative repaint, ever.
        if t.get("finish", "clean") in ("clean", "api"):
            out = scan_finish(out)
        cv2.imwrite(r["output_path"], out)
        wm = t.get("watermark", "")
        if wm:
            _stamp_watermark(r["output_path"], wm)

    await asyncio.get_event_loop().run_in_executor(None, work)

    # Reflect new placement / edits back into the stored result.
    r["items"] = [
        {
            "id": it["id"], "bbox": it["bbox"], "original": it["original"],
            "translation": it["translation"], "type": it["type"],
            "in_bubble": it["in_bubble"], "dark": it.get("dark", False),
            "placed": it.get("placed", False),
            "rotation": it.get("rotation", 0),
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

        original = ""
        ocr = _get_ocr()
        if ocr and ocr.ok:
            padded = cv2.copyMakeBorder(
                crop, 12, 12, 12, 12,
                cv2.BORDER_CONSTANT, value=(255, 255, 255))
            original = ocr.read(padded)

        if not original:
            return {"original": "", "translation": ""}

        translator = make_translator(provider, api_key, model, style_prompt,
                                     source_lang=t.get("source_lang", "Japanese"),
                                     translate_sfx=bool(t.get("translate_sfx", False)))
        out = translator.translate_texts({"0": original}, target_lang)
        entry = out.get(0) or out.get("0") or {}
        translation = entry.get("translation", original)
        return {"original": original, "translation": translation}

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
            zf.write(path, f"{i:03d}_{stem}.png")
            count += 1

    if count == 0:
        raise HTTPException(400, "No finished pages to download yet")

    buf.seek(0)
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="translated_pages.zip"'},
    )


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
