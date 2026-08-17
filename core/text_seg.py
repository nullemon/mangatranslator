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
import hashlib
import cv2
import numpy as np

_SESSION = None
_TRIED = False

# Page stroke-mask cache: re-rendering the same page (an editor Apply) shouldn't
# re-run the GPU segmentation — the base image is identical every time. Keyed by
# a cheap content signature, bounded to a handful of recent pages.
_MASK_CACHE = {}

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


def _preload_cuda12_libs():
    """onnxruntime-gpu dlopens CUDA 12 libraries by soname (libcublasLt.so.12,
    libcudnn.so.9, ...). A torch built for CUDA 13 ships only .so.13, so the
    CUDA provider fails and ORT silently falls back to CPU. Load the libs from
    the nvidia-*-cu12 pip wheels (installed by setup_gpu.sh) into the process
    so the provider can initialize. Harmless no-op when the wheels are absent."""
    import ctypes
    import glob
    import site
    paths = []
    try:
        paths += site.getsitepackages()
    except Exception:
        pass
    try:
        paths.append(site.getusersitepackages())
    except Exception:
        pass
    sonames = (
        "cuda_runtime/lib/libcudart.so.12*",
        "cublas/lib/libcublasLt.so.12*",
        "cublas/lib/libcublas.so.12*",
        "cufft/lib/libcufft.so.11*",
        "curand/lib/libcurand.so.10*",
        "cudnn/lib/libcudnn.so.9*",
    )
    for sp in paths:
        nv = os.path.join(sp, "nvidia")
        if not os.path.isdir(nv):
            continue
        for pat in sonames:
            for lib in sorted(glob.glob(os.path.join(nv, pat))):
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


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
        from .device import onnx_providers
        providers = onnx_providers(ort.get_available_providers())
        if providers[0] == "CUDAExecutionProvider":
            _preload_cuda12_libs()
        _SESSION = ort.InferenceSession(path, providers=providers)
        # Report the provider the session ACTUALLY uses — ORT can list CUDA as
        # available, fail to load its libs, and quietly run on CPU.
        used = (_SESSION.get_providers() or ["CPUExecutionProvider"])[0]
        print(f"[text_seg] comic-text-detector ready ({used})")
    except Exception as e:
        print(f"[text_seg] unavailable ({e}); using ink-deviation fallback")
        _SESSION = None
    return _SESSION


class TextSegmenter:
    """Page-level text stroke mask + text-block boxes.
    Lazy: nothing heavy happens until `ok`."""

    @property
    def ok(self) -> bool:
        return _load() is not None

    def _run(self, image: np.ndarray):
        """Letterbox to the model's fixed square input, run, and return
        (outputs, scale, valid_w, valid_h)."""
        sess = _load()
        if sess is None:
            return None
        h, w = image.shape[:2]
        scale = INPUT_SIZE / max(h, w)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), np.uint8)
        canvas[:nh, :nw] = resized

        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None]
        name = sess.get_inputs()[0].name
        from .gpu_throttle import limit as _gpu_limit
        with _gpu_limit():
            outputs = sess.run(None, {name: blob})
        return outputs, scale, nw, nh

    @staticmethod
    def _sig(image: np.ndarray):
        """Cheap content signature for the mask cache (shape + hash of a 32x32
        thumbnail) — fast, and collisions are effectively impossible here."""
        small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
        return (image.shape, hashlib.blake2b(small.tobytes(), digest_size=16).digest())

    @staticmethod
    def _strip_nontext_blobs(m):
        """Drop genuinely SOLID blobs (eyes, ornaments, tone patches) from the
        stroke mask, so they don't get treated as text and erased.

        Judged by STROKE THICKNESS, not fill ratio. A glyph is built from
        strokes: however dense it looks, no ink pixel sits far from an edge. A
        filled shape has a centre far from every edge. Measured as
        2 x (max distance-to-edge) / smaller side:

            bold kanji  0.21 - 0.42      eyes / solid shapes  0.80 - 1.04

        Fill ratio alone was the old test and it does NOT separate them — 囲
        fills 0.66 of its box and was being stripped as "solid", which is
        exactly why big bold bubble text survived erasure and the translation
        landed on top of it. Fill is still required as a second opinion, so a
        blob must look solid BOTH ways before it goes."""
        if m is None:
            return m
        binary = (m > 0).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        out = m.copy()
        removed = 0
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < 400:
                continue
            if area / max(w * h, 1) <= 0.62:
                continue                      # clearly stroke-like
            comp = (labels[y:y + h, x:x + w] == i).astype(np.uint8)
            # pad so the distance transform sees the real edges
            comp = cv2.copyMakeBorder(comp, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
            dt = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
            thickness = float(2.0 * dt.max()) / max(min(w, h), 1)
            if thickness >= 0.60:             # solid lump, not lettering
                out[labels == i] = 0
                removed += 1
        if removed:
            print(f"[text_seg] stripped {removed} solid non-text blob(s) "
                  f"(eyes/ornaments) from the stroke mask")
        return out

    def mask(self, image: np.ndarray) -> np.ndarray:
        """Binary mask (uint8 0/255, same HxW as `image`) of text strokes.
        Cached per page so an editor re-render doesn't re-run the model."""
        h, w = image.shape[:2]
        key = self._sig(image)
        hit = _MASK_CACHE.get(key)
        if hit is not None and hit.shape == (h, w):
            return hit
        ran = self._run(image)
        if ran is None:
            return np.zeros((h, w), np.uint8)
        outs, scale, nw, nh = ran

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
        binary = self._strip_nontext_blobs(binary)
        _MASK_CACHE[key] = binary
        if len(_MASK_CACHE) > 8:
            _MASK_CACHE.pop(next(iter(_MASK_CACHE)))
        return binary

    def detect_blocks(self, image: np.ndarray, conf_thresh: float = 0.45,
                      nms_thresh: float = 0.35):
        """Text-block boxes [(x, y, w, h), ...] from the model's detection
        head. Finds EVERY block of lettering on the page (it's manga-trained),
        including the vertical columns and tilted banners the LLM pass can
        miss. Callers verify each box with OCR, so a stray detection is
        harmless — it just reads as no Japanese and gets dropped."""
        h, w = image.shape[:2]
        ran = self._run(image)
        if ran is None:
            return []
        outs, scale, nw, nh = ran

        # Detection head: the 3-D output of (1, N, 5+nc) decoded predictions.
        det = None
        for o in outs:
            a = np.asarray(o)
            if a.ndim == 3 and a.shape[1] > 64 and 6 <= a.shape[2] <= 16:
                det = a[0]
                break
        if det is None:
            return []

        obj = det[:, 4]
        cls = det[:, 5:].max(axis=1) if det.shape[1] > 5 else np.ones_like(obj)
        conf = obj * cls
        keep = conf > conf_thresh
        if not np.any(keep):
            return []
        det, conf = det[keep], conf[keep]

        boxes = []
        for cx, cy, bw, bh in det[:, :4]:
            boxes.append([float(cx - bw / 2), float(cy - bh / 2),
                          float(bw), float(bh)])
        idx = cv2.dnn.NMSBoxes(boxes, conf.astype(float).tolist(),
                               conf_thresh, nms_thresh)
        if idx is None or len(idx) == 0:
            return []

        out = []
        for i in np.array(idx).flatten():
            bx, by, bw, bh = boxes[int(i)]
            # Map back through the letterbox to page coordinates.
            x = int(bx / scale)
            y = int(by / scale)
            ww = int(bw / scale)
            hh = int(bh / scale)
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            ww = min(ww, w - x)
            hh = min(hh, h - y)
            if ww < 12 or hh < 12 or ww * hh > 0.25 * w * h:
                continue
            out.append((x, y, ww, hh))
        return out
