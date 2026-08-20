"""What a page actually costs.

Until now the app spent money silently. You picked a model, ran a chapter,
and the only place the bill showed up was the provider's billing page a day
later — by which point there was no way to tell which pages, or which part of
the pipeline, ate it.

This keeps a running tally of the tokens every API call reports, so the log
says what each page cost and where it went. The token counts are exact: they
come straight back from the API. The dollar figure is an ESTIMATE from the
table below, and it is the part to distrust — providers change prices, and
your account may be on different terms.

The number that usually surprises people is `think`. Reasoning models bill
their thinking at the OUTPUT rate, which is the expensive one, and a model
left to think as long as it likes can spend more on deliberating about a
speech bubble than on translating it.
"""
import os
import threading
from typing import Dict, Optional, Tuple

# USD per 1,000,000 tokens, as (input, output). List prices at the time of
# writing — treat as a rough guide, not a quote. Longest matching prefix wins,
# so a dated or "-latest" variant picks up its family's price.
#
# Override any of it without touching this file:
#   MANGA_PRICE_gemini_2_5_pro="1.25,10"
PRICES: Dict[str, Tuple[float, float]] = {
    "gemini-3-pro":          (2.00, 12.00),
    "gemini-3-flash":        (0.30,  2.50),
    "gemini-2.5-pro":        (1.25, 10.00),
    "gemini-2.5-flash-lite": (0.10,  0.40),
    "gemini-2.5-flash":      (0.30,  2.50),
    "gemini-2.0-flash":      (0.10,  0.40),
}

# What the rolling aliases point at. Google repoints these over time; if an
# alias moves to a new family the estimate drifts until this is updated, which
# is exactly why the log prints the token counts next to the dollars.
ALIASES = {
    "gemini-pro-latest":        "gemini-3-pro",
    "gemini-flash-latest":      "gemini-2.5-flash",
    "gemini-flash-lite-latest": "gemini-2.5-flash-lite",
}

# Cached input is billed at a fraction of the normal input rate. Gemini's
# implicit cache discount is 75% off, i.e. you pay a quarter.
CACHE_RATE = 0.25


def _price(model: str) -> Optional[Tuple[float, float]]:
    m = (model or "").strip().lower()
    m = ALIASES.get(m, m)
    env = os.environ.get("MANGA_PRICE_" + m.replace("-", "_").replace(".", "_"))
    if env:
        try:
            a, b = env.split(",")
            return float(a), float(b)
        except Exception:
            pass
    best = None
    for key, val in PRICES.items():
        if m.startswith(key) and (best is None or len(key) > len(best[0])):
            best = (key, val)
    return best[1] if best else None


class Usage:
    """One call's token counts, in the shape both APIs can be mapped onto."""

    __slots__ = ("model", "calls", "input", "cached", "thinking", "output")

    def __init__(self, model="", calls=0, input=0, cached=0, thinking=0,
                 output=0):
        self.model = model
        self.calls = calls
        self.input = int(input)
        self.cached = int(cached)
        self.thinking = int(thinking)
        self.output = int(output)

    def add(self, other: "Usage") -> "Usage":
        self.model = self.model or other.model
        self.calls += other.calls
        self.input += other.input
        self.cached += other.cached
        self.thinking += other.thinking
        self.output += other.output
        return self

    @property
    def billed_output(self) -> int:
        """Thinking is billed at the output rate, so it belongs here."""
        return self.output + self.thinking

    def cost(self) -> Optional[float]:
        p = _price(self.model)
        if not p:
            return None
        pin, pout = p
        fresh = max(0, self.input - self.cached)
        return ((fresh * pin + self.cached * pin * CACHE_RATE
                 + self.billed_output * pout) / 1_000_000.0)

    def line(self, label: str = "") -> str:
        bits = [f"in {self.input:,}"]
        if self.cached:
            bits.append(f"cached {self.cached:,}")
        if self.thinking:
            bits.append(f"think {self.thinking:,}")
        bits.append(f"out {self.output:,}")
        text = " · ".join(bits)
        c = self.cost()
        money = f"  ≈ ${c:.4f}" if c is not None else "  (no price for this model)"
        head = f"{label} " if label else ""
        return f"{head}{text}{money}"


def from_gemini(model: str, data: dict) -> Usage:
    """Map Gemini's usageMetadata onto Usage.

    `candidatesTokenCount` counts only the visible answer; the reasoning is
    reported separately as `thoughtsTokenCount` and is easy to miss — which is
    how a bill quietly grows without the answers getting any longer.
    """
    u = (data or {}).get("usageMetadata") or {}

    def n(*names):
        for k in names:
            v = u.get(k)
            if isinstance(v, (int, float)):
                return int(v)
        return 0

    return Usage(
        model=model, calls=1,
        input=n("promptTokenCount"),
        cached=n("cachedContentTokenCount"),
        thinking=n("thoughtsTokenCount", "thoughtTokenCount"),
        output=n("candidatesTokenCount"),
    )


def from_claude(model: str, resp) -> Usage:
    u = getattr(resp, "usage", None)
    if u is None:
        return Usage(model=model, calls=1)

    def n(*names):
        for k in names:
            v = getattr(u, k, None)
            if isinstance(v, (int, float)):
                return int(v)
        return 0

    return Usage(
        model=model, calls=1,
        input=n("input_tokens"),
        cached=n("cache_read_input_tokens"),
        output=n("output_tokens"),
    )


# ── running totals ──────────────────────────────────────────────────────
# Two scopes: the page currently being worked on, and everything since the
# server started. Both are process-wide, and pages are processed one at a
# time behind the task lock, so a plain lock is enough.
_lock = threading.Lock()
_page = Usage()
_session = Usage()


def record(u: Usage) -> Usage:
    with _lock:
        _page.add(u)
        _session.add(u)
    return u


def start_page() -> None:
    global _page
    with _lock:
        _page = Usage()


def page() -> Usage:
    with _lock:
        return Usage(_page.model, _page.calls, _page.input, _page.cached,
                     _page.thinking, _page.output)


def session() -> Usage:
    with _lock:
        return Usage(_session.model, _session.calls, _session.input,
                     _session.cached, _session.thinking, _session.output)


def report_page(what: str = "page") -> Optional[dict]:
    """Print what this page cost, and what the session has cost so far."""
    p, s = page(), session()
    if not p.calls:
        return None
    print(f"[cost] {what}: {p.calls} API call(s) · {p.line()}", flush=True)
    if s.calls > p.calls:
        c = s.cost()
        extra = f"  ≈ ${c:.2f}" if c is not None else ""
        print(f"[cost] since start: {s.calls} call(s), "
              f"{s.input + s.billed_output:,} tokens{extra}", flush=True)
    return as_dict(p, s)


def short_note() -> str:
    """A few words for the progress line, so the price shows up in the app and
    not only in the server log. Empty when nothing was spent (offline engine,
    a re-render, a clean-only pass)."""
    p = page()
    if not p.calls:
        return ""
    c = p.cost()
    money = f"≈ ${c:.3f}" if c is not None else f"{p.input + p.billed_output:,} tokens"
    think = f", {p.thinking:,} thinking" if p.thinking else ""
    return f"{money} · {p.calls} AI call{'s' if p.calls != 1 else ''}{think}"


def as_dict(p: Optional[Usage] = None, s: Optional[Usage] = None) -> dict:
    p = p or page()
    s = s or session()
    return {
        "page": {"model": p.model, "calls": p.calls, "input": p.input,
                 "cached": p.cached, "thinking": p.thinking,
                 "output": p.output, "cost": p.cost()},
        "session": {"model": s.model, "calls": s.calls, "input": s.input,
                    "cached": s.cached, "thinking": s.thinking,
                    "output": s.output, "cost": s.cost()},
    }
