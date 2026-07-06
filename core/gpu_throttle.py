"""Duty-cycle GPU cap.

User space can't tell CUDA "run at 60%" — while a kernel executes the GPU
is ~100% busy, full stop. What we CAN control is the duty cycle: time every
model call and sleep afterwards so the GPU idles a matching share of wall
clock. At a 60% cap, a 1.0s inference is followed by a ~0.67s pause, so
averaged over any window the GPU sits near 60% busy and clocks, fans and
the rest of the desktop get room to breathe. Translation gets slower by
the same factor — that's the trade.

Every GPU-touching module (bubble seg, OCR, CRAFT, text seg, LaMa,
upscaler) wraps its inference in `limit()`. At the default 100% cap the
wrapper is a no-op.
"""
import time

_cap = 100  # percent of wall clock the GPU may be busy; 100 = uncapped


def set_cap(percent) -> int:
    """Set the cap (20-100). Anything out of range or unparsable = 100."""
    global _cap
    try:
        p = int(float(percent))
    except (TypeError, ValueError):
        p = 100
    new = 100 if p >= 100 or p <= 0 else max(20, p)
    if new != _cap:
        print(f"[gpu] usage cap: {new}%" + ("" if new == 100 else
              " (duty-cycled — jobs pause between model calls)"))
    _cap = new
    return _cap


def cap() -> int:
    return _cap


class limit:
    """`with limit(): model(x)` — sleeps after the call to hold the cap."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if _cap >= 100 or exc_type is not None:
            return False
        try:
            # Make sure the kernel actually finished so `busy` is real GPU
            # time, not just the async launch.
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        busy = time.perf_counter() - self._t0
        pause = busy * (100.0 / _cap - 1.0)
        if pause > 0.002:
            time.sleep(min(pause, 4.0))
        return False
