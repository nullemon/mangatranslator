import asyncio
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, JSONResponse
from starlette.requests import Request

from core.pipeline import TranslationPipeline

app = FastAPI(title="MangaTranslator")

for d in ("uploads", "output", "fonts"):
    os.makedirs(d, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

tasks: dict = {}


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/api/translate")
async def translate(
    file: UploadFile = File(...),
    api_key: str = Form(...),
    target_lang: str = Form("English"),
    model: str = Form("claude-sonnet-4-6"),
    smart_mode: str = Form("false"),
    font: str = Form(""),
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

    tasks[task_id] = {
        "status": "processing",
        "step": 0,
        "message": "Queued",
        "progress": 0,
        "upload_path": upload_path,
        "output_path": output_path,
    }

    font_path = f"fonts/{font}" if font and os.path.exists(f"fonts/{font}") else None

    asyncio.create_task(
        _run(task_id, upload_path, output_path, api_key, target_lang, model, smart_mode == "true", font_path)
    )

    return {"task_id": task_id}


async def _run(
    task_id: str,
    image_path: str,
    output_path: str,
    api_key: str,
    target_lang: str,
    model: str,
    smart_mode: bool,
    font_path: str = None,
):
    try:
        pipeline = TranslationPipeline(
            api_key=api_key,
            target_lang=target_lang,
            model=model,
            use_smart_detection=smart_mode,
            font_path=font_path,
        )

        def on_progress(update):
            tasks[task_id].update(update)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: pipeline.process(image_path, output_path, on_progress),
        )

        tasks[task_id].update(
            {
                "status": "done",
                "progress": 100,
                "message": "Complete!",
                "result": result,
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
    return FileResponse(p, media_type="image/png")


@app.get("/api/original/{task_id}")
async def original(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404)
    p = tasks[task_id].get("upload_path", "")
    if not p or not os.path.exists(p):
        raise HTTPException(404)
    return FileResponse(p)


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
