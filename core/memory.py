"""Give memory back when we're done with it.

The heavy models (manga-ocr, LaMa, bubble seg, CRAFT, text seg, upscaler)
are process-wide singletons — great while translating, but between jobs
they sit on GPU VRAM, and torch's caching allocator plus Python's heap
never hand freed memory back on their own. On WSL2 that reads as "vmmem
keeps growing and the GPU stays hogged" long after the last page.

Three tiers, cheapest first:

  light_sweep()   — after every job: collect garbage, return torch's cached
                    VRAM to the driver.
  deep_sweep()    — after a few idle minutes: light sweep + malloc_trim, so
                    freed heap pages go back to the Linux kernel (WSL2's
                    page reporting then lets Windows reclaim them and vmmem
                    shrinks).
  unload_models() — after a long idle stretch: drop every cached model so
                    VRAM goes to ~0. Models lazy-reload on the next job
                    (a few seconds of warm-up, nothing else changes).
"""
import gc


def _cuda_release():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _malloc_trim():
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def light_sweep():
    gc.collect()
    _cuda_release()


def deep_sweep():
    light_sweep()
    _malloc_trim()


def unload_models() -> bool:
    """Release every cached model. Returns True if anything was actually
    loaded (so the caller can log once instead of every sweep)."""
    released = False
    try:
        from .ocr import MangaOCR
        if MangaOCR._shared is not None:
            MangaOCR._shared = None
            released = True
    except Exception:
        pass
    try:
        from . import local_mt
        if local_mt.unload():
            released = True
    except Exception:
        pass
    try:
        from .lama import LamaInpaint
        if LamaInpaint._shared is not None:
            LamaInpaint._shared = None
            released = True
    except Exception:
        pass
    try:
        from . import bubble_seg
        if bubble_seg._MODEL is not None:
            bubble_seg._MODEL = None
            released = True
        bubble_seg._TRIED = False
    except Exception:
        pass
    try:
        from . import text_detect
        if text_detect._CRAFT is not None:
            try:
                text_detect._CRAFT.unload_craftnet_model()
                text_detect._CRAFT.unload_refinenet_model()
            except Exception:
                pass
            text_detect._CRAFT = None
            released = True
        text_detect._TRIED = False
    except Exception:
        pass
    try:
        from . import text_seg
        if text_seg._SESSION is not None:
            text_seg._SESSION = None
            released = True
        text_seg._TRIED = False
        text_seg._MASK_CACHE.clear()
    except Exception:
        pass
    try:
        from . import upscale
        if upscale._CACHE:
            upscale._CACHE.clear()
            released = True
    except Exception:
        pass
    deep_sweep()
    return released
