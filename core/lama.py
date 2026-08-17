"""Local GPU inpainting (LaMa) for clean text removal over artwork and
screentones — far better than OpenCV inpaint. Optional: if not installed,
`ok` is False and callers fall back to cv2.inpaint.

The weights download ONCE into torch's hub cache and are reused forever
after; loading them again later is a disk read, not a download."""
import os
import cv2
import numpy as np
from PIL import Image


class LamaInpaint:
    _shared = None

    def __init__(self):
        self._lama = None
        self._failed = False

    @property
    def ok(self) -> bool:
        if self._failed:
            return False
        if self._lama is not None:
            return True
        return self._load()

    @staticmethod
    def weights_path():
        """Where the LaMa weights live once downloaded.

        simple-lama-inpainting keeps them in torch's hub cache, so the ~200MB
        download happens ONCE and every later start just reads the file. Set
        TORCH_HOME to move that cache somewhere else."""
        try:
            import torch
            return os.path.join(torch.hub.get_dir(), "checkpoints", "big-lama.pt")
        except Exception:
            return os.path.join(os.path.expanduser("~"), ".cache", "torch",
                                "hub", "checkpoints", "big-lama.pt")

    @classmethod
    def cached(cls) -> bool:
        """True when the weights are already on disk (no download needed)."""
        try:
            p = cls.weights_path()
            return os.path.exists(p) and os.path.getsize(p) > 1_000_000
        except Exception:
            return False

    def _load(self) -> bool:
        if LamaInpaint._shared is not None:
            self._lama = LamaInpaint._shared
            return True
        try:
            from simple_lama_inpainting import SimpleLama
            # Say which is actually happening: the old message claimed "first
            # run downloads ~200MB" on EVERY load, which made a normal
            # few-second read off disk look like a repeat download.
            if self.cached():
                print("[lama] loading inpainting model from disk cache...")
            else:
                print("[lama] downloading the inpainting model (~200MB, "
                      "one time — kept at "
                      f"{self.weights_path()})...", flush=True)
            self._lama = SimpleLama()
            LamaInpaint._shared = self._lama
            print("[lama] LaMa ready")
            return True
        except Exception as e:
            print(f"[lama] unavailable ({e}); using cv2.inpaint")
            self._failed = True
            return False

    def inpaint(self, bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Erase the white area of `mask` from `bgr` and return the result.
        Returns None on failure so the caller can fall back."""
        if not self.ok:
            return None
        try:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            m = (mask > 0).astype(np.uint8) * 255
            from .gpu_throttle import limit as _gpu_limit
            with _gpu_limit():
                out = self._lama(Image.fromarray(rgb), Image.fromarray(m))
            out = np.array(out.convert("RGB"))
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            if out.shape[:2] != bgr.shape[:2]:
                out = cv2.resize(out, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_AREA)
            return out
        except Exception as e:
            print(f"[lama] inpaint failed: {e}")
            return None
