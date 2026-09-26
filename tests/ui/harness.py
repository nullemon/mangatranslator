"""Shared Playwright harness for the editor's on-image tools.

Drives the real page in Chromium against a running server: uploads pages
through the "Clean (remove text)" workflow (no API key needed), switches to
the Translated tab, picks tools, performs real pointer gestures on the
overlay, presses Apply & Re-render and fetches the re-rendered result so the
tests can check pixels.

Environment:
  MT_URL     server base URL        (default http://127.0.0.1:8022)
  MT_CHROME  chromium executable    (default the playwright chromium build)
  MT_PAGES   directory of test pages (default /tmp/claude-0/op)
"""
import io
import json
import os
import sys
import time
import urllib.request

import cv2
import numpy as np

BASE = os.environ.get("MT_URL", "http://127.0.0.1:8022")
CHROME = os.environ.get("MT_CHROME", "/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
PAGES_DIR = os.environ.get("MT_PAGES", "/tmp/claude-0/op")
# LaMa on a CPU takes 30-45 s per region on an idle machine and several
# times that on a busy one; MT_SLOW scales every wait (default 1.0).
SLOW = float(os.environ.get("MT_SLOW", "1"))
CLEAN_TIMEOUT = 900 * SLOW      # two pages through Clean
APPLY_TIMEOUT = 600 * SLOW      # one Apply & Re-render


def server_up():
    try:
        with urllib.request.urlopen(BASE + "/", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def fetch_image(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        data = r.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None, f"could not decode {path}"
    return img


# ── pixel helpers ────────────────────────────────────────────────────────

def region(img, x, y, w, h):
    H, W = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    return img[y0:y1, x0:x1]


def dark_fraction(img, x, y, w, h, thr=110):
    g = cv2.cvtColor(region(img, x, y, w, h), cv2.COLOR_BGR2GRAY)
    return float((g < thr).mean()) if g.size else 0.0


def changed_mask(a, b, thr=8):
    """Boolean mask of pixels that differ between two same-size images."""
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2) > thr


def changed_outside(a, b, rects, margin=48, thr=8):
    """How many pixels differ between a and b OUTSIDE the given rects
    (each grown by `margin`). Returns (count, bbox-of-changes or None)."""
    m = changed_mask(a, b, thr)
    H, W = m.shape
    for (x, y, w, h) in rects:
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(W, x + w + margin), min(H, y + h + margin)
        m[y0:y1, x0:x1] = False
    n = int(m.sum())
    if n:
        ys, xs = np.where(m)
        return n, (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return n, None


def poly_rect(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


# ── the browser session ──────────────────────────────────────────────────

class UI:
    def __init__(self, pw, viewport=(1400, 1000)):
        self.browser = pw.chromium.launch(executable_path=CHROME, headless=True)
        self.ctx = self.browser.new_context(
            viewport={"width": viewport[0], "height": viewport[1]})
        self.page = self.ctx.new_page()
        self.page.set_default_timeout(int(60_000 * SLOW))
        self.js_errors = []
        self.console_errors = []
        self.rerender_bodies = []      # every /api/rerender request body, in order
        self.page.on("pageerror", lambda e: self.js_errors.append(str(e)))
        self.page.on("console", lambda m: (m.type == "error"
                                           and self.console_errors.append(m.text)))

        def on_req(req):
            if "/api/rerender/" in req.url and req.method == "POST":
                try:
                    self.rerender_bodies.append(json.loads(req.post_data or "{}"))
                except Exception:
                    self.rerender_bodies.append({"_unparsed": req.post_data})
        self.page.on("request", on_req)
        self.page.goto(BASE + "/")
        self.page.wait_for_selector("#fileInput", state="attached")
        # Fresh state: the app remembers the workflow etc. in localStorage.
        self.page.evaluate("localStorage.clear()")
        self.page.reload()
        self.page.wait_for_selector("#fileInput", state="attached")

    def close(self):
        try:
            self.ctx.close()
        finally:
            self.browser.close()

    # ── workflow ─────────────────────────────────────────────────────────
    def run_clean(self, files, credit="", timeout=None):
        """Upload pages through 'Clean (remove text)' and wait until every
        page is done. Returns the list of task ids in page order."""
        p = self.page
        p.click('.wf-card[data-wf="clean"]')
        if credit:
            p.fill("#credit", credit)
        p.set_input_files("#fileInput", files)
        p.wait_for_timeout(500)
        p.click("#goBtn")
        deadline = time.time() + (timeout or CLEAN_TIMEOUT)
        n = len(files)
        while time.time() < deadline:
            if n == 1:
                done = p.evaluate(
                    "() => document.getElementById('pageResult').style.display !== 'none'"
                    " && !!document.getElementById('transFull').src")
                err = p.evaluate(
                    "() => (document.getElementById('progressMsg') || {}).textContent || ''")
            else:
                done = p.evaluate(
                    "() => document.querySelectorAll('.pg-chip .pg-dot.done').length") == n
                err = p.evaluate(
                    "() => document.querySelectorAll('.pg-chip .pg-dot.error').length")
                err = "error" if err else ""
            if done:
                break
            if "⚠" in str(err) or err == "error":
                raise RuntimeError(f"a page failed: {err}")
            time.sleep(2)
        else:
            raise TimeoutError("pages did not finish in time")
        ids = []
        if n == 1:
            ids.append(self.task_id())
        else:
            for i in range(n):
                self.select_page(i)
                ids.append(self.task_id())
            self.select_page(0)
        return ids

    def select_page(self, i):
        chips = self.page.query_selector_all(".pg-chip .pg-idx")
        chips[i].click()
        self.page.wait_for_function(
            "() => { const im = document.getElementById('transFull');"
            " return im.complete && im.naturalWidth > 0; }")
        self.page.wait_for_timeout(300)

    def task_id(self):
        src = self.page.evaluate("() => document.getElementById('transFull').getAttribute('src')")
        return src.split("/api/result/")[1].split("?")[0]

    def rev(self):
        src = self.page.evaluate("() => document.getElementById('transFull').getAttribute('src')")
        return int(src.split("?t=")[1]) if "?t=" in src else 0

    def result_image(self):
        return fetch_image(f"/api/result/{self.task_id()}?t={self.rev()}&x={time.time()}")

    def original_image(self):
        return fetch_image(f"/api/original/{self.task_id()}?x={time.time()}")

    # ── editor ───────────────────────────────────────────────────────────
    def open_editor(self):
        self.page.click('.tab[data-tab="translated"]')
        self.page.wait_for_function(
            "() => { const im = document.getElementById('transFull');"
            " return im.complete && im.naturalWidth > 0; }")

    def set_tool(self, name):
        """Select a tool (idempotent: does not toggle it off if already on)."""
        btn = self.page.locator(f'.tool-btn[data-tool="{name}"]')
        btn.scroll_into_view_if_needed()
        if "btn-primary" not in (btn.get_attribute("class") or ""):
            btn.click()
        self.page.wait_for_timeout(150)

    def current_tool(self):
        return self.page.evaluate("() => document.getElementById('moveLayer').dataset.tool || null")

    def dims(self):
        return self.page.evaluate(
            "() => [document.getElementById('transFull').naturalWidth,"
            " document.getElementById('transFull').naturalHeight]")

    def layer_rect(self):
        return self.page.evaluate(
            "() => { const r = document.getElementById('moveLayer').getBoundingClientRect();"
            " return [r.left, r.top, r.width, r.height]; }")

    def ensure_visible(self, x, y):
        """Scroll so that image point (x, y) is inside the viewport."""
        W, H = self.dims()
        self.page.evaluate(
            "([x, y, W, H]) => { const ml = document.getElementById('moveLayer');"
            " const r = ml.getBoundingClientRect();"
            " const cy = r.top + y / H * r.height; const cx = r.left + x / W * r.width;"
            " const vh = window.innerHeight, vw = window.innerWidth;"
            " if (cy < 40 || cy > vh - 40) window.scrollBy(0, cy - vh / 2);"
            " if (cx < 40 || cx > vw - 40) window.scrollBy(cx - vw / 2, 0); }",
            [x, y, W, H])
        self.page.wait_for_timeout(100)

    def to_client(self, x, y):
        W, H = self.dims()
        l, t, w, h = self.layer_rect()
        return l + x / W * w, t + y / H * h

    def drag(self, x0, y0, x1, y1, steps=12):
        """Press at image (x0,y0), move to (x1,y1), release."""
        self.ensure_visible(x0, y0)
        cx0, cy0 = self.to_client(x0, y0)
        cx1, cy1 = self.to_client(x1, y1)
        m = self.page.mouse
        m.move(cx0, cy0)
        m.down()
        for i in range(1, steps + 1):
            f = i / steps
            m.move(cx0 + (cx1 - cx0) * f, cy0 + (cy1 - cy0) * f)
        m.up()
        self.page.wait_for_timeout(150)

    def drag_path(self, pts):
        """Free-form drag through image points (lasso / restore / keep)."""
        self.ensure_visible(pts[0][0], pts[0][1])
        m = self.page.mouse
        cx, cy = self.to_client(*pts[0])
        m.move(cx, cy)
        m.down()
        for (x, y) in pts[1:]:
            cx, cy = self.to_client(x, y)
            m.move(cx, cy)
        m.up()
        self.page.wait_for_timeout(150)

    def click_at(self, x, y, **kw):
        self.ensure_visible(x, y)
        cx, cy = self.to_client(x, y)
        self.page.mouse.click(cx, cy, **kw)
        self.page.wait_for_timeout(120)

    def pen_polygon(self, pts):
        """Click each point, then click the first again to close."""
        for (x, y) in pts:
            self.click_at(x, y)
        self.click_at(*pts[0])

    def covers(self):
        """The active page's cover list as the app holds it (read through the
        overlay + the next request is what the tests assert on; this is only
        for quick diagnostics)."""
        return self.page.evaluate(
            "() => Array.from(document.querySelectorAll('#moveLayer > *'))"
            ".map(e => e.className.baseVal !== undefined ? 'svg' : e.className)")

    def overlay_counts(self):
        return self.page.evaluate(
            "() => { const L = document.getElementById('moveLayer');"
            " return { cover: L.querySelectorAll('.cover-box').length,"
            "  svg: L.querySelectorAll('svg').length,"
            "  polygon: L.querySelectorAll('svg polygon').length,"
            "  line: L.querySelectorAll('svg line').length,"
            "  dab: L.querySelectorAll('.clone-dab').length,"
            "  restoreClick: L.querySelectorAll('.restore-click').length,"
            "  children: L.children.length }; }")

    def apply(self, timeout=None):
        """Press Apply & Re-render, wait for the re-render to land. Returns
        the request body that was sent."""
        n0 = len(self.rerender_bodies)
        rev0 = self.rev()
        btn = self.page.locator("#editApply")
        btn.scroll_into_view_if_needed()
        btn.click()
        deadline = time.time() + (timeout or APPLY_TIMEOUT)
        while time.time() < deadline:
            if self.rev() > rev0 and not self.page.evaluate(
                    "() => document.getElementById('editApply').disabled"):
                break
            time.sleep(1)
        else:
            raise TimeoutError("Apply & Re-render did not finish")
        self.page.wait_for_function(
            "() => { const im = document.getElementById('transFull');"
            " return im.complete && im.naturalWidth > 0; }")
        self.page.wait_for_timeout(300)
        assert len(self.rerender_bodies) > n0, "no rerender request was sent"
        return self.rerender_bodies[-1]

    def hint(self):
        return self.page.evaluate("() => document.getElementById('editHint').textContent")

    def drain_errors(self):
        errs = list(self.js_errors)
        cons = list(self.console_errors)
        self.js_errors.clear()
        self.console_errors.clear()
        return errs, cons


class Report:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, detail=""):
        self.rows.append((name, ok, detail))
        print(("PASS " if ok else "FAIL ") + name + (f" — {detail}" if detail else ""),
              flush=True)

    def check(self, name, cond, detail=""):
        self.add(name, bool(cond), detail if not cond else "")
        return bool(cond)

    def summary(self):
        bad = [r for r in self.rows if not r[1]]
        print(f"\n{len(self.rows) - len(bad)}/{len(self.rows)} checks passed")
        for name, _, detail in bad:
            print(f"  FAIL {name}: {detail}")
        return not bad
