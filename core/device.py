"""One place that decides which compute device the local models run on.

Three backends, picked automatically:

  cuda  — an NVIDIA card (Windows / Linux desktops)
  mps   — Apple Silicon's GPU (M1/M2/M3/M4 Macs), via Metal
  cpu   — everything else, and the fallback whenever the above misbehave

Set MANGA_DEVICE=cpu (or cuda / mps) to override the choice by hand — useful
on a Mac when a model hits an unimplemented Metal kernel, or for comparing
speed. Apple's Metal backend still lacks a few operators, so we also switch on
PyTorch's automatic CPU fallback for those instead of letting them crash.
"""
import os

_FORCED = (os.environ.get("MANGA_DEVICE") or "").strip().lower()
_REPORTED = False


def _mps_ready() -> bool:
    try:
        import torch
        return bool(getattr(torch.backends, "mps", None)
                    and torch.backends.mps.is_available()
                    and torch.backends.mps.is_built())
    except Exception:
        return False


def torch_device() -> str:
    """'cuda', 'mps' or 'cpu' — the best available PyTorch device."""
    global _REPORTED
    if _FORCED in ("cpu", "cuda", "mps"):
        return _FORCED
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    if _mps_ready():
        # A handful of ops (some interpolation / padding modes the inpainting
        # and upscaling nets use) have no Metal kernel yet. Without this they
        # raise; with it PyTorch quietly runs just those on the CPU.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        if not _REPORTED:
            print("[device] Apple Silicon GPU (Metal/MPS) in use — "
                  "set MANGA_DEVICE=cpu to disable", flush=True)
            _REPORTED = True
        return "mps"
    return "cpu"


def is_apple_silicon() -> bool:
    import platform
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def onnx_providers(available) -> list:
    """Preferred onnxruntime execution providers, best first, filtered to the
    ones this build actually has. CoreML is Apple's accelerator — it's what
    makes the text-stroke model usable on a Mac instead of CPU-slow."""
    order = ("CUDAExecutionProvider", "CoreMLExecutionProvider",
             "CPUExecutionProvider")
    if _FORCED == "cpu":
        order = ("CPUExecutionProvider",)
    return [p for p in order if p in available] or ["CPUExecutionProvider"]
