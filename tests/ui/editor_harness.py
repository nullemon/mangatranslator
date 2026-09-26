"""Shared helpers for the browser tests of the on-image editor.

The tests drive the REAL page in Chromium against a RUNNING server. They
never reach into app.js's closures: state is checked the way a user would
see it — the overlay boxes on screen, the Details tab, the body of the
/api/rerender request the page sends, and the pixels of the result image.

Environment (all optional):
    MT_URL       the app, default http://127.0.0.1:8021
    MT_PAGES     comma-separated page images, default /tmp/claude-0/op/3.jpg,4.jpg
    MT_CHROME    chromium binary (default: the Playwright bundle under /opt)
    MT_HEADED    set to 1 to watch the browser
"""
import io
import json
import os
import re
import sys
import time
import urllib.request

import numpy as np

URL = os.environ.get("MT_URL", "http://127.0.0.1:8021")
PAGES = [p for p in os.environ.get(
    "MT_PAGES", "/tmp/claude-0/op/3.jpg,/tmp/claude-0/op/4.jpg").split(",") if p]
CHROME = os.environ.get(
    "MT_CHROME", "/opt/pw-browsers/chromium-1194/chrome-linux/chrome")


def server_up() -> bool:
    try:
        with urllib.request.urlopen(URL + "/", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def skip_unless_ready():
    """Exit 0 with a SKIP line when the server (or a page image) is missing."""
    if not server_up():
        print(f"SKIP: no server at {URL} (start it: PORT=8021 python app.py)")
        sys.exit(0)
    missing = [p for p in PAGES if not os.path.exists(p)]
    if missing:
        print(f"SKIP: test page(s) missing: {missing}")
        sys.exit(0)
    if not os.path.exists(CHROME):
        print(f"SKIP: chromium not found at {CHROME}")
        sys.exit(0)


def fetch_result(task_id: str, rev) -> np.ndarray:
    """The rendered page as an RGB array (no browser cache in the way)."""
    import cv2
    with urllib.request.urlopen(f"{URL}/api/result/{task_id}?t={rev}&x={time.time()}") as r:
        buf = np.frombuffer(r.read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def dark_count(img: np.ndarray, bbox, pad=0) -> int:
    """How many text-dark pixels sit inside bbox — lettering is dark on a
    cleaned balloon, so this rises where text is and falls where it left."""
    x, y, w, h = [int(v) for v in bbox]
    H, W = img.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    if x1 <= x0 or y1 <= y0:
        return 0
    g = img[y0:y1, x0:x1].mean(axis=2)
    return int((g < 96).sum())


def region_diff(a: np.ndarray, b: np.ndarray, bbox, pad=0) -> float:
    x, y, w, h = [int(v) for v in bbox]
    H, W = a.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(np.abs(a[y0:y1, x0:x1].astype(int) - b[y0:y1, x0:x1].astype(int)).mean())


class Browser:
    """One Chromium tab on the app, with every JS error and console error
    recorded, and every /api/rerender request body captured."""

    def __init__(self, headed=False):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(
            executable_path=CHROME, headless=not headed)
        self.ctx = self.browser.new_context(viewport={"width": 1400, "height": 1000})
        self.page = self.ctx.new_page()
        self.js_errors = []
        self.console_errors = []
        self.dialogs = []
        self.rerenders = []          # [(task_id, body dict)] in send order
        self.page.on("pageerror", lambda e: self.js_errors.append(str(e)))

        def on_console(m):
            if m.type != "error":
                return
            # A blocked third-party fetch (fonts CDN behind a proxy) is not an
            # app error; anything from the app's own origin is.
            loc = (m.location or {}).get("url", "")
            if "Failed to load resource" in m.text and loc and not loc.startswith(URL):
                return
            self.console_errors.append(f"{m.text} @ {loc}")
        self.page.on("console", on_console)

        def on_dialog(d):
            self.dialogs.append((d.type, d.message))
            d.dismiss()
        self.page.on("dialog", on_dialog)

        def on_request(req):
            m = re.match(r".*/api/rerender/([^/?]+)$", req.url)
            if m and req.method == "POST":
                try:
                    self.rerenders.append((m.group(1), json.loads(req.post_data or "{}")))
                except Exception as e:
                    self.rerenders.append((m.group(1), {"_unparsed": str(e)}))
        self.page.on("request", on_request)
        self.page.goto(URL + "/")
        self.page.wait_for_selector("#goBtn", state="attached")   # hidden until a file is picked

    def close(self):
        try:
            self.ctx.close(); self.browser.close()
        finally:
            self._pw.stop()

    # ── setup ────────────────────────────────────────────────────────────
    def set_toggle(self, sel, on):
        self.page.evaluate("""([s, v]) => { const el = document.querySelector(s);
            if (el && el.checked !== v) { el.checked = v;
              el.dispatchEvent(new Event('change', {bubbles: true})); } }""", [sel, on])

    def translate_pages(self, paths, timeout_s=900):
        """Upload the pages through the picker and run them; returns the task
        ids in page order once every page is done."""
        p = self.page
        p.click('.wf-card[data-wf="scan-translate"]')
        p.select_option("#engine", "claude")
        p.fill("#apiKey", "sk-test-key")
        self.set_toggle("#smartMode", False)
        self.set_toggle("#hdUpscale", False)
        self.set_toggle("#compressOut", False)
        self.set_toggle("#styleFonts", False)
        p.set_input_files("#fileInput", paths)
        p.wait_for_selector("#goBtn:not([disabled])")
        task_ids = []

        def on_resp(resp):
            if resp.url.endswith("/api/translate") and resp.status == 200:
                try:
                    task_ids.append(resp.json()["task_id"])
                except Exception:
                    pass
        p.on("response", on_resp)
        p.click("#goBtn")
        p.wait_for_selector("#resultSection", state="visible")
        t0 = time.time()
        n = len(paths)
        while time.time() - t0 < timeout_s:
            if n == 1:
                done = p.evaluate("document.getElementById('pageResult').style.display !== 'none'")
            else:
                done = p.evaluate("document.querySelectorAll('.pg-dot.done').length") == n
            err = p.evaluate("document.querySelectorAll('.pg-dot.error').length") if n > 1 else 0
            if err:
                raise RuntimeError("a page failed to translate: " + p.inner_text("#progressMsg"))
            if done:
                break
            time.sleep(2)
        else:
            raise RuntimeError("translation did not finish in time")
        p.remove_listener("response", on_resp)
        self.select_page(1)
        return task_ids

    # ── navigation ──────────────────────────────────────────────────────
    def select_page(self, n):
        """Click page n (1-based) in the strip and wait for its result image."""
        p = self.page
        if p.evaluate("document.querySelectorAll('.pg-chip').length") > 1:
            p.click(f".pg-chip:nth-child({n}) .pg-idx")
        p.wait_for_function("document.getElementById('pageResult').style.display !== 'none'")
        p.wait_for_function("""() => { const i = document.getElementById('transFull');
            return i.complete && i.naturalWidth > 0; }""")

    def tab(self, name):
        self.page.click(f'.tab[data-tab="{name}"]')
        self.page.wait_for_selector(f"#panel-{name}.active")

    def active_task(self):
        """(task_id, rev) of the page on screen, read off the result image."""
        src = self.page.get_attribute("#transFull", "src") or ""
        m = re.search(r"/api/result/([^?]+)\?t=(\d+)", src)
        return (m.group(1), int(m.group(2))) if m else (None, None)

    def tool_on(self):
        return self.page.evaluate("document.getElementById('moveLayer').dataset.tool || ''")

    def set_tool(self, name):
        """Click the toolbar button until exactly `name` is active (None = off)."""
        cur = self.tool_on()
        if name is None:
            if cur:
                self.page.click(f'.tool-btn[data-tool="{cur}"]')
        elif cur != name:
            self.page.click(f'.tool-btn[data-tool="{name}"]')
        self.page.wait_for_function(
            "t => (document.getElementById('moveLayer').dataset.tool || '') === t", arg=name or "")
        time.sleep(0.15)

    # ── overlay ─────────────────────────────────────────────────────────
    def image_scale(self):
        """(sx, sy): multiply screen px by these to get image px."""
        return self.page.evaluate("""() => { const i = document.getElementById('transFull');
            const r = i.getBoundingClientRect();
            return [i.naturalWidth / r.width, i.naturalHeight / r.height]; }""")

    def overlay_boxes(self):
        """The item boxes on the overlay: [{id, cls, rect(screen), pct}]."""
        return self.page.evaluate("""() => {
            const img = document.getElementById('transFull').getBoundingClientRect();
            const W = document.getElementById('transFull').naturalWidth;
            const H = document.getElementById('transFull').naturalHeight;
            return [...document.querySelectorAll('#moveLayer .move-box, #moveLayer .resize-box')].map(b => {
              const r = b.getBoundingClientRect();
              const tag = (b.querySelector('.move-tag') || {}).textContent || '';
              return { id: tag.replace('#', ''), cls: b.className,
                       rect: [r.left, r.top, r.width, r.height],
                       img: [(r.left - img.left) * W / img.width, (r.top - img.top) * H / img.height,
                             r.width * W / img.width, r.height * H / img.height] };
            }); }""")

    def box_by_id(self, id_, scroll=True):
        """The overlay box for an item, scrolled into the viewport first so a
        mouse gesture on its rect actually lands on it."""
        if scroll:
            self.scroll_to_box(id_)
        for b in self.overlay_boxes():
            if b["id"] == str(id_):
                return b
        return None

    def added_box(self):
        self.page.evaluate("""() => { const b = document.querySelector('#moveLayer .added-box');
            if (b) b.scrollIntoView({block: 'center'}); }""")
        time.sleep(0.1)
        for b in self.overlay_boxes():
            if "added-box" in b["cls"]:
                return b
        return None

    def scroll_to_box(self, id_):
        self.page.evaluate("""id => { const b = [...document.querySelectorAll('#moveLayer .move-box, #moveLayer .resize-box')]
              .find(b => ((b.querySelector('.move-tag')||{}).textContent||'').replace('#','') === String(id));
            if (b) b.scrollIntoView({block: 'center'}); }""", str(id_))
        time.sleep(0.1)

    def image_rect(self):
        """Viewport rect of the result image after scrolling its top into view."""
        self.page.evaluate("document.getElementById('transStage').scrollIntoView({block: 'start'})")
        time.sleep(0.1)
        return self.page.evaluate("""() => { const r = document.getElementById('transFull').getBoundingClientRect();
            return [r.left, r.top, r.width, r.height]; }""")

    def hint(self):
        return self.page.inner_text("#editHint").strip()

    def drag(self, x0, y0, x1, y1, steps=12):
        m = self.page.mouse
        m.move(x0, y0); m.down()
        for i in range(1, steps + 1):
            m.move(x0 + (x1 - x0) * i / steps, y0 + (y1 - y0) * i / steps)
        m.up()
        time.sleep(0.1)

    # ── apply ───────────────────────────────────────────────────────────
    APPLY_TIMEOUT_S = int(os.environ.get("MT_APPLY_TIMEOUT", "600"))

    def apply(self, click="#editApply", btn="#editApply", timeout_s=None):
        """Click `click` (Apply & Re-render, or a popover button that triggers
        one) and wait for the re-render to land; returns the request body.
        A re-render is ~20s on an idle CPU box and much more on a loaded one,
        hence the generous default (MT_APPLY_TIMEOUT)."""
        timeout_s = timeout_s or self.APPLY_TIMEOUT_S
        n = len(self.rerenders)
        task, rev = self.active_task()
        t0 = time.time()
        try:
            # A previous re-render may still be running (the button is
            # disabled meanwhile): wait for it rather than failing the click.
            self.page.wait_for_function(
                "s => { const b = document.querySelector(s); return b && !b.disabled; }",
                arg=btn, timeout=timeout_s * 1000)
            with self.page.expect_response(
                    lambda r: "/api/rerender/" in r.url and r.request.method == "POST",
                    timeout=timeout_s * 1000) as ri:
                self.page.click(click, timeout=15000)
            resp = ri.value
            if time.time() - t0 > 60:
                print(f"   (slow re-render: {time.time() - t0:.0f}s)", flush=True)
        except Exception as e:
            shot = f"/tmp/mt_apply_fail_{int(time.time())}.png"
            try:
                self.page.screenshot(path=shot, full_page=True)
            except Exception:
                shot = "(no screenshot)"
            state = self.page.evaluate("""s => { const b = document.querySelector(s);
                return b ? {display: b.style.display, disabled: b.disabled, text: b.textContent,
                            visible: !!(b.offsetWidth || b.offsetHeight)} : null; }""", btn)
            raise RuntimeError(
                f"Apply via {click!r} produced no re-render: {type(e).__name__}: {str(e)[:120]}; "
                f"button={state}; requests so far={len(self.rerenders)} (was {n}); "
                f"js_errors={self.js_errors[-3:]}; screenshot={shot}") from e
        if resp.status != 200:
            raise RuntimeError(f"rerender failed: {resp.status} {resp.text()[:200]}")
        self.page.wait_for_function(
            "([b]) => { const el = document.querySelector(b); return el && !el.disabled; }",
            arg=[btn], timeout=timeout_s * 1000)
        # The image the page shows must have moved to the new revision.
        self.page.wait_for_function(
            "([t, r]) => { const s = document.getElementById('transFull').src;"
            " return s.includes('/api/result/' + t + '?t=') && parseInt(s.split('?t=')[1]) > r; }",
            arg=[task, rev], timeout=20000)
        self.page.wait_for_function("""() => { const i = document.getElementById('transFull');
            return i.complete && i.naturalWidth > 0; }""")
        time.sleep(0.2)
        bodies = self.rerenders[n:]
        return bodies[-1][1] if bodies else {}

    # ── details tab ─────────────────────────────────────────────────────
    def details(self):
        """{id: text} of every row on the Details tab (does not switch tabs)."""
        return self.page.evaluate("""() => { const o = {};
            document.querySelectorAll('#translationsList .tl-edit').forEach(t => o[t.dataset.id] = t.value);
            return o; }""")

    def listener_counts(self, selector):
        """{event: count} of the JS listeners bound directly on the first
        element matching `selector` (via the DevTools protocol)."""
        client = self.ctx.new_cdp_session(self.page)
        try:
            doc = client.send("DOM.getDocument", {"depth": 0})
            node = client.send("DOM.querySelector",
                               {"nodeId": doc["root"]["nodeId"], "selector": selector})
            obj = client.send("DOM.resolveNode", {"nodeId": node["nodeId"]})
            ls = client.send("DOMDebugger.getEventListeners",
                             {"objectId": obj["object"]["objectId"]})
            out = {}
            for l in ls["listeners"]:
                out[l["type"]] = out.get(l["type"], 0) + 1
            return out
        finally:
            client.detach()


class Report:
    def __init__(self):
        self.rows = []

    def check(self, name, ok, detail=""):
        self.rows.append((name, bool(ok), detail))
        print(("PASS " if ok else "FAIL ") + name + (f"  — {detail}" if detail and not ok else ""), flush=True)
        return ok

    def section(self, name):
        """`with rep.section("Move"):` — an exception inside the block is a
        FAIL for that scenario, and the next scenario still runs."""
        rep = self

        class _Section:
            def __enter__(self):
                print(f"\n== {name}", flush=True)

            def __exit__(self, et, ev, tb):
                if et is not None:
                    import traceback
                    rep.check(f"{name}: ran to the end", False, f"{et.__name__}: {str(ev)[:400]}")
                    traceback.print_exception(et, ev, tb)
                return True
        return _Section()

    def finish(self, br=None):
        if br is not None:
            self.check("no JS exceptions", not br.js_errors, "; ".join(br.js_errors)[:500])
            self.check("no console errors", not br.console_errors, "; ".join(br.console_errors)[:500])
        failed = [r for r in self.rows if not r[1]]
        print(f"\n{len(self.rows) - len(failed)} passed, {len(failed)} failed")
        for name, _, d in failed:
            print(f"  FAIL {name}: {d}")
        return 1 if failed else 0
