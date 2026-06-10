"""GPU text-pixel segmentation (comic-text-detector).

Produces a per-pixel mask of every text stroke on the page — trained on manga,
so it cleanly separates lettering from screentones and line art. Used to make
text removal surgical: we erase exactly the strokes the model marks, nothing
else.

Runs the ONNX export with onnxruntime (CUDA when available, CPU otherwise).
Weights auto-download to models/comictextdetector.pt.onnx on first use, or run
./setup_gpu.sh to pre-fetch. Optional: if onnxruntime or the weights are
missing, `ok` is False and callers fall back to the deviation heuristic.

Override the weights path with env var TEXT_SEG_MODEL.
"""

import os
import cv2
import numpy as np

_SESSION = None
_TRIED = False

INPUT_SIZE = 1024
DEFAULT_PATH = "models/comictextdetector.pt.onnx"
WEIGHT_URLS = [
    "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx",
    "https://github.com/dmMaze/comic-text-detector/releases/download/data/comictextdetector.pt.onnx",
]


def _weights_path() -> str:
    return os.environ.get("TEXT_SEG_MODEL", "").strip() or DEFAULT_PATH


def _download_weights(dest: str) -> bool:
    import httpx
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    for url in WEIGHT_URLS:
        try:
            print(f"[text_seg] downloading weights (~90MB): {url}")
            with httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as r:
                if r.status_code != 200:
                    continue
                tmp = dest + ".part"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
            os.replace(tmp, dest)
            print(f"[text_seg] weights saved to {dest}")
            return True
        except Exception as e:
            print(f"[text_seg] download failed from {url}: {e}")
    return False


def _load():
    """Load (and cache process-wide) the ONNX session."""
    global _SESSION, _TRIED
    if _SESSION is not None or _TRIED:
        return _SESSION
    _TRIED = True
    try:
        import onnxruntime as ort
        path = _weights_path()
        if not os.path.exists(path) and not _download_weights(path):
            print("[text_seg] weights unavailable; using ink-deviation fallback")
            return None
        providers = [
            p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if p in ort.get_available_providers()
        ] or ["CPUExecutionProvider"]
        _SESSION = ort.InferenceSession(path, providers=providers)
        print(f"[text_seg] comic-text-detector ready ({providers[0]})")
    except Exception as e:
        print(f"[text_seg] unavailable ({e}); using ink-deviation fallback")
        _SESSION = None
    return _SESSION


class TextSegmenter:
    """Page-level text stroke mask. Lazy: nothing heavy happens until `ok`."""

    @property
    def ok(self) -> bool:
        return _load() is not None

    def mask(self, image: np.ndarray) -> np.ndarray:
        """Binary mask (uint8 0/255, same HxW as `image`) of text strokes."""
        sess = _load()
        h, w = image.shape[:2]
        if sess is None:
            return np.zeros((h, w), np.uint8)

        # Letterbox to the model's fixed square input, keeping aspect ratio.
        scale = INPUT_SIZE / max(h, w)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), np.uint8)
        canvas[:nh, :nw] = resized

        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None]

        name = sess.get_inputs()[0].name
        outs = sess.run(None, {name: blob})

        # The stroke mask is the 4-D single-channel output with the largest
        # spatial size (the other heads are detection boxes / line maps).
        seg = None
        for o in outs:
            a = np.asarray(o)
            if a.ndim == 4 and a.shape[1] == 1:
                if seg is None or a.shape[2] * a.shape[3] > seg.shape[2] * seg.shape[3]:
                    seg = a
        if seg is None:
            return np.zeros((h, w), np.uint8)

        m = seg[0, 0]
        if m.max() <= 1.5:  # sigmoid output 0-1
            m = m * 255.0
        m = np.clip(m, 0, 255).astype(np.uint8)
        if m.shape != (INPUT_SIZE, INPUT_SIZE):
            m = cv2.resize(m, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)

        # Undo the letterbox: crop the valid area, then scale to page size.
        m = m[:nh, :nw]
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        _, binary = cv2.threshold(m, 60, 255, cv2.THRESH_BINARY)
        return binary
