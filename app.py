import asyncio
import io
import os
import time
import uuid
import zipfile
from pathlib import Path

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.requests import Request

from core.pipeline import TranslationPipeline, scan_cleanup
from core.compositor import Compositor
from core.enhancer import ImageEnhancer

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
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload an image file")

    task_id = str(uuid.uuid4())
    ext = Path(file.filename or "img.png").suffix or ".png"
    upload_path = f"uploads/{task_id}{ext}"
    output_path = f"output/{task_id}{ext}"

    content = await file.read()
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
        "name": file.filename or "page.png",
        "mode": "translate",
    }

    asyncio.create_task(
        _run(
            task_id, upload_path, output_path, api_key, target_lang, provider, model,
            smart_mode == "true", font_path,
            enhance == "true", enhance_provider, enhance_key, enhance_prompt, enhance_model,
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
):
    try:
        loop = asyncio.get_event_loop()

        if enhance:
            tasks[task_id].update(
                {"step": 0, "progress": 4,
                 "message": f"Enhancing rough page with {enhance_provider.title()}..."}
            )
            enhanced_path = f"uploads/{task_id}_enhanced.png"
            enhancer = ImageEnhancer()

            def do_enhance():
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError(f"Cannot load image: {image_path}")
                cleaned = scan_cleanup(img)
                try:
                    out = enhancer.enhance(cleaned, enhance_prompt, enhance_provider, enhance_key, enhance_model)
                except Exception as e:
                    print(f"[enhance] AI step failed, using local scan cleanup: {e}")
                    out = cleaned
                cv2.imwrite(enhanced_path, out)

            await loop.run_in_executor(None, do_enhance)
            tasks[task_id]["enhanced_path"] = enhanced_path
            tasks[task_id]["enhanced_url"] = f"/api/enhanced/{task_id}"
            image_path = enhanced_path

        pipeline = TranslationPipeline(
            api_key=api_key,
            target_lang=target_lang,
            provider=provider,
            model=model,
            use_smart_detection=smart_mode,
            font_path=font_path,
        )

        def on_progress(update):
            tasks[task_id].update(update)

        result = await loop.run_in_executor(
            None,
            lambda: pipeline.process(image_path, output_path, on_progress),
        )
        MASKS[task_id] = getattr(pipeline, "last_masks", {}) or {}

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
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload an image file")

    task_id = str(uuid.uuid4())
    ext = Path(file.filename or "img.png").suffix or ".png"
    upload_path = f"uploads/{task_id}{ext}"
    output_path = f"output/{task_id}_scan.png"

    content = await file.read()
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
        _run_enhance(task_id, upload_path, output_path, provider, api_key, prompt, model)
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
):
    try:
        tasks[task_id].update(
            {"step": 1, "progress": 15,
             "message": f"Sending to {provider.title()} image model..."}
        )
        enhancer = ImageEnhancer()

        def do_work():
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Cannot load image: {image_path}")
            # Local cleanup first: deskew, crop the background away, and
            # normalize the paper to pure white. This already looks like a
            # scan and removes the side background reliably.
            cleaned = scan_cleanup(img)
            try:
                out = enhancer.enhance(cleaned, prompt, provider, api_key, model)
            except Exception as e:
                # If the AI step fails, fall back to the local clean scan so
                # Raw → Scan always produces a usable result.
                print(f"[enhance] AI step failed, using local scan cleanup: {e}")
                tasks[task_id]["message"] = f"AI step failed ({e}); used local clean scan"
                out = cleaned
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
    if not base or not os.path.exists(base) or not r.get("items"):
        raise HTTPException(400, "This page can't be re-rendered")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")
    excluded = {str(i) for i in payload.get("excluded", [])}
    edits = {str(k): v for k, v in (payload.get("edits") or {}).items()}
    font_scale = float(payload.get("font_scale", 1.0))
    offsets = {str(k): v for k, v in (payload.get("offsets") or {}).items()}

    items = []
    for it in r["items"]:
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
        })

    def work():
        base_img = cv2.imread(base)
        if base_img is None:
            raise ValueError("Base image missing")
        comp = Compositor(t.get("font_path"), font_scale=font_scale)
        out = comp.compose(base_img, items, MASKS.get(task_id), offsets)
        cv2.imwrite(r["output_path"], out)

    await asyncio.get_event_loop().run_in_executor(None, work)

    # Reflect new placement / edits back into the stored result.
    r["items"] = [
        {
            "id": it["id"], "bbox": it["bbox"], "original": it["original"],
            "translation": it["translation"], "type": it["type"],
            "in_bubble": it["in_bubble"], "dark": it.get("dark", False),
            "placed": it.get("placed", False),
        }
        for it in items
    ]
    r["translations"] = {
        str(it["id"]): {
            "original": it["original"], "translation": it["translation"], "type": it["type"]
        }
        for it in items
    }
    r["num_translated"] = sum(1 for it in items if it.get("placed"))

    return {"items": r["items"], "ts": time.time()}


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
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
