"""Shared Playwright harness for the editor's on-image tool tests.

Drives the REAL page in a real Chromium against a REAL running server. Nothing
here is mocked: pages are uploaded through the file input, processed by the
server, and the tools are exercised with mouse gestures on the overlay.

Environment:
  MT_BASE   server base URL (default http://127.0.0.1:8023)
  MT_PAGE   raw page image to upload (default /tmp/claude-0/op/3.jpg)
  MT_CHROME chromium executable (default the playwright-1194 build)

Every test script skips (exit 0, message on stderr) when the server isn't up,
so a plain `python tests/ui/test_*.py` never fails just because nothing is
listening.
"""
import io
import json
import os
import sys
import time

import numpy as np

try:
    import requests
except ImportError:                       # pragma: no cover
    requests = None

BASE = os.environ.get("MT_BASE", "http://127.0.0.1:8023")
PAGE_IMG = os.environ.get("MT_PAGE", "/tmp/claude-0/op/3.jpg")
CHROME = os.environ.get("MT_CHROME", "/opt/pw-browsers/chromium-1194/chrome-linux/chrome")

# Clean-only on a CPU box is ~75 s (LaMa); re-renders 30-60 s — several
# times that when the box is busy with another page.
PROCESS_TIMEOUT = 600_000
RERENDER_TIMEOUT = 600_000


def server_up():
    if requests is None:
        return False
    try:
        return requests.get(BASE + "/", timeout=3).status_code == 200
    except Exception:
        return False


def skip_unless_server():
    if not server_up():
        sys.stderr.write(f"SKIP: no MangaTranslator server at {BASE}\n")
        sys.exit(0)
    if not os.path.exists(PAGE_IMG):
        sys.stderr.write(f"SKIP: test page {PAGE_IMG} missing\n")
        sys.exit(0)


def fetch_result(task_id, rev=0):
    """The page's current output image as a BGR numpy array."""
    import cv2
    r = requests.get(f"{BASE}/api/result/{task_id}?t={rev}", timeout=60)
    r.raise_for_status()
    arr = np.frombuffer(r.content, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def ink_fraction(img, x, y, w, h, thresh=100):
    """Share of pixels darker than `thresh` inside an image-coords rect."""
    roi = img[max(0, y):y + h, max(0, x):x + w]
    if roi.size == 0:
        return 0.0
    g = roi.mean(axis=2)
    return float((g < thresh).mean())


def changed_fraction(a, b, x, y, w, h, thresh=40):
    """Share of pixels that differ noticeably between two renders in a rect."""
    ra = a[max(0, y):y + h, max(0, x):x + w].astype(np.int16)
    rb = b[max(0, y):y + h, max(0, x):x + w].astype(np.int16)
    if ra.size == 0 or ra.shape != rb.shape:
        return 1.0
    return float((np.abs(ra - rb).max(axis=2) > thresh).mean())


class Results:
    """Collects PASS/FAIL lines so one script can report every scenario."""

    def __init__(self):
        self.rows = []

    def check(self, name, ok, detail=""):
        self.rows.append((name, bool(ok), detail))
        print(("PASS  " if ok else "FAIL  ") + name + (f"  -- {detail}" if detail else ""),
              flush=True)
        return bool(ok)

    def summary(self):
        bad = [r for r in self.rows if not r[1]]
        print(f"\n{len(self.rows) - len(bad)}/{len(self.rows)} passed")
        for name, _, detail in bad:
            print(f"  FAIL {name}: {detail}")
        return 0 if not bad else 1


class Editor:
    """One browser tab on the app, with helpers for the result-tab tools."""

    def __init__(self, pw, headless=True):
        self.browser = pw.chromium.launch(executable_path=CHROME, headless=headless)
        self.ctx = self.browser.new_context(viewport={"width": 1280, "height": 900})
        try:
            self.ctx.grant_permissions(["clipboard-read", "clipboard-write"], origin=BASE)
        except Exception:
            pass
        self.page = self.ctx.new_page()
        self.js_errors = []
        self.console_errors = []
        self.requests = []          # (url, method, json-or-None)
        self.responses = []         # (url, status, text)
        self.page.on("pageerror", lambda e: self.js_errors.append(str(e)))
        self.page.on("console", self._on_console)
        self.page.on("request", self._on_request)
        self.page.on("response", self._on_response)
        self.task_id = None

    def _on_console(self, msg):
        if msg.type == "error":
            t = msg.text
            # A failed fetch logs "Failed to load resource: 400" — that is the
            # server answering, not a client bug; JS exceptions arrive via
            # pageerror.
            if "Failed to load resource" in t:
                return
            self.console_errors.append(t)

    def _on_request(self, req):
        if "/api/" not in req.url:
            return
        body = None
        if req.method == "POST":
            try:
                body = req.post_data_json
            except Exception:
                body = None
        self.requests.append((req.url, req.method, body))

    def _on_response(self, resp):
        if "/api/" not in resp.url or "/api/status/" in resp.url:
            return
        # No body reads inside the event handler (they block the dispatcher);
        # the response object is kept for a later, lazy look.
        self.responses.append((resp.url, resp.status, resp))

    def close(self):
        try:
            self.ctx.close()
        finally:
            self.browser.close()

    # ── setup ────────────────────────────────────────────────────────────
    def open(self):
        self.page.goto(BASE + "/", wait_until="domcontentloaded")
        self.page.wait_for_selector("#goBtn", state="attached")
        # Fresh state: no saved key, default Claude engine.
        self.page.evaluate("localStorage.clear()")
        self.page.reload(wait_until="domcontentloaded")
        self.page.wait_for_selector("#goBtn", state="attached")

    def set_engine(self, eng):
        self.page.select_option("#engine", eng)

    def run_clean(self, img_path=PAGE_IMG):
        """Upload a page via the UI and run the 'Clean — no key' workflow."""
        p = self.page
        p.click('.wf-card[data-wf="clean"]')
        p.set_input_files("#fileInput", img_path)
        p.wait_for_selector("#goBtn:not([disabled])")
        p.click("#goBtn")
        p.wait_for_selector("#pageResult", state="visible", timeout=PROCESS_TIMEOUT)
        p.wait_for_function(
            "() => { const i = document.getElementById('transFull'); return i.complete && i.naturalWidth > 0; }",
            timeout=PROCESS_TIMEOUT)
        src = p.get_attribute("#transFull", "src")
        self.task_id = src.split("/api/result/")[1].split("?")[0]
        return self.task_id

    def go_tab(self, name):
        self.page.click(f'.tab[data-tab="{name}"]')

    def select_tool(self, name):
        self.go_tab("translated")
        btn = self.page.locator(f'.tool-btn[data-tool="{name}"]')
        btn.scroll_into_view_if_needed()
        active = "btn-primary" in (btn.get_attribute("class") or "")
        if not active:
            btn.click()
        self.page.wait_for_function(
            f"() => document.getElementById('moveLayer').dataset.tool === '{name}'")

    def tool(self):
        return self.page.evaluate("document.getElementById('moveLayer').dataset.tool")

    def hint(self):
        return self.page.text_content("#editHint")

    # ── geometry ─────────────────────────────────────────────────────────
    def dims(self):
        return self.page.evaluate(
            "[document.getElementById('transFull').naturalWidth, document.getElementById('transFull').naturalHeight]")

    def layer_rect(self):
        self.page.locator("#moveLayer").scroll_into_view_if_needed()
        return self.page.evaluate(
            "(() => { const r = document.getElementById('moveLayer').getBoundingClientRect(); return [r.left, r.top, r.width, r.height]; })()")

    def to_client(self, x, y):
        W, H = self.dims()
        L, T, w, h = self.layer_rect()
        return L + x / W * w, T + y / H * h

    def ensure_visible(self, x, y):
        """Scroll so image point (x, y) is inside the viewport."""
        cx, cy = self.to_client(x, y)
        vh = self.page.evaluate("window.innerHeight")
        if cy < 60 or cy > vh - 60:
            self.page.evaluate(f"window.scrollBy(0, {cy - vh / 2})")
            self.page.wait_for_timeout(100)

    # ── gestures (image coordinates) ─────────────────────────────────────
    def drag_box(self, x0, y0, x1, y1, steps=8):
        self.ensure_visible((x0 + x1) / 2, (y0 + y1) / 2)
        ax, ay = self.to_client(x0, y0)
        bx, by = self.to_client(x1, y1)
        m = self.page.mouse
        m.move(ax, ay)
        m.down()
        m.move(bx, by, steps=steps)
        m.up()

    def lasso(self, pts):
        self.ensure_visible(pts[0][0], pts[0][1])
        m = self.page.mouse
        cx, cy = self.to_client(*pts[0])
        m.move(cx, cy)
        m.down()
        for x, y in pts[1:]:
            cx, cy = self.to_client(x, y)
            m.move(cx, cy, steps=3)
        m.up()

    def click_points(self, pts, close=True):
        """Point-translate: click each point, then click the first to close."""
        self.ensure_visible(pts[0][0], pts[0][1])
        m = self.page.mouse
        for x, y in pts:
            cx, cy = self.to_client(x, y)
            m.click(cx, cy)
            self.page.wait_for_timeout(50)
        if close:
            cx, cy = self.to_client(*pts[0])
            m.click(cx, cy)

    # ── IME dialog ───────────────────────────────────────────────────────
    def ime_visible(self):
        return self.page.evaluate(
            "(() => { const b = document.querySelector('.ime-back'); return !!b && b.style.display !== 'none'; })()")

    def ime_wait(self, timeout=30_000):
        self.page.wait_for_function(
            "() => { const b = document.querySelector('.ime-back'); return !!b && b.style.display !== 'none'; }",
            timeout=timeout)

    def ime_wait_gone(self, timeout=10_000):
        self.page.wait_for_function(
            "() => { const b = document.querySelector('.ime-back'); return !b || b.style.display === 'none'; }",
            timeout=timeout)

    def ime_title(self):
        return self.page.text_content(".ime-title")

    def ime_msg(self):
        return self.page.text_content(".ime-msg")

    def ime_use(self, original, translation):
        self.page.fill(".ime-src", original)
        self.page.fill(".ime-out", translation)
        self.page.click(".ime-use")
        self.ime_wait_gone()

    def ime_cancel(self):
        self.page.click(".ime-cancel")
        self.ime_wait_gone()

    # ── state ────────────────────────────────────────────────────────────
    def state(self):
        """The active page's editable state, read straight out of the app."""
        return self.page.evaluate("window.__mtState ? window.__mtState() : null")

    def overlay_counts(self):
        return self.page.evaluate("""(() => {
          const L = document.getElementById('moveLayer');
          return {
            add: L.querySelectorAll('.add-box').length,
            cover: L.querySelectorAll('.cover-box').length,
            move: L.querySelectorAll('.move-box').length,
            resize: L.querySelectorAll('.resize-box').length,
            svg: L.querySelectorAll('svg').length,
            polys: L.querySelectorAll('svg polygon').length,
            pen: L.querySelectorAll('svg polyline').length,
            pop: L.querySelectorAll('.edit-pop').length,
          };
        })()""")

    def details_tab_visible(self):
        return self.page.locator('.tab[data-tab="details"]').is_visible()

    def details_rows(self):
        if not self.details_tab_visible():
            return []            # a Clean page with nothing added has no list
        self.go_tab("details")
        return self.page.evaluate("""(() =>
          [...document.querySelectorAll('#translationsList .tl-item')].map(d => ({
            added: d.classList.contains('added-item'),
            id: (d.querySelector('.tl-edit') || {}).dataset ? d.querySelector('.tl-edit').dataset.id : null,
            text: d.querySelector('.tl-edit') ? d.querySelector('.tl-edit').value : null,
            font: d.querySelector('.tl-font') ? d.querySelector('.tl-font').value : null,
            disabled: d.querySelector('.tl-edit') ? d.querySelector('.tl-edit').disabled : null,
          })))()""")

    def apply(self, timeout=RERENDER_TIMEOUT):
        """Press Apply & Re-render on the Translated tab and wait for it."""
        self.go_tab("translated")
        btn = self.page.locator("#editApply")
        if not btn.is_visible():
            self.go_tab("details")
            btn = self.page.locator("#applyBtn")
        n = len([r for r in self.responses if "/api/rerender/" in r[0]])
        btn.click()
        t0 = time.time()
        while len([r for r in self.responses if "/api/rerender/" in r[0]]) <= n:
            if time.time() - t0 > timeout / 1000:
                raise TimeoutError("no re-render response")
            self.page.wait_for_timeout(200)   # lets the event dispatcher run
        self.page.wait_for_function(
            "() => !document.getElementById('applyBtn').disabled && !document.getElementById('editApply').disabled",
            timeout=timeout)
        # the result img reloads with a new ?t= — wait for it
        self.page.wait_for_function(
            "() => { const i = document.getElementById('transFull'); return i.complete && i.naturalWidth > 0; }",
            timeout=timeout)
        new = [r for r in self.responses if "/api/rerender/" in r[0]][n:]
        return new[-1] if new else None

    def last_request(self, path_part):
        for url, method, body in reversed(self.requests):
            if path_part in url:
                return body
        return None

    def rev(self):
        src = self.page.get_attribute("#transFull", "src")
        return int(src.split("?t=")[1]) if "?t=" in src else 0

    def errors(self):
        return {"js": list(self.js_errors), "console": list(self.console_errors)}
