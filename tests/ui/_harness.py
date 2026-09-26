"""Shared Playwright harness for the browser tests in tests/ui/.

Run any test file directly with the project's Python:

    MT_URL=http://127.0.0.1:8024 python tests/ui/test_upload_paths.py

Every test opens the real app in headless Chromium, records every page error
and console error (a JS exception is a failure), and skips cleanly when the
server is not running.
"""
import io
import os
import sys
import time
import urllib.request
import zipfile

BASE = os.environ.get("MT_URL", "http://127.0.0.1:8024")
CHROME = os.environ.get(
    "MT_CHROME", "/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
INPUTS = os.environ.get("MT_INPUTS", os.path.join(HERE, "_inputs"))
SHOTS = os.environ.get("MT_SHOTS", os.path.join(HERE, "_shots"))
OP = os.environ.get("MT_PAGES", "/tmp/claude-0/op")


def server_up() -> bool:
    try:
        with urllib.request.urlopen(BASE + "/", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def skip_unless_server():
    if not server_up():
        print(f"SKIP: no server at {BASE} (start app.py and set MT_URL)")
        sys.exit(0)


def page_paths(*names):
    """Absolute paths of the sample One Piece raws (3.jpg, 4.jpg, ...)."""
    out = []
    for n in names:
        p = os.path.join(OP, n)
        if not os.path.exists(p):
            print(f"SKIP: sample page {p} is missing")
            sys.exit(0)
        out.append(p)
    return out


def fetch(path: str) -> bytes:
    with urllib.request.urlopen(BASE + path, timeout=120) as r:
        return r.read()


def decode(data: bytes):
    import cv2
    import numpy as np
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def task_of(src: str) -> str:
    """The task id inside a /api/result/<id>?t=.. image URL ("" if none)."""
    if not src or "/api/result/" not in src:
        return ""
    return src.split("/api/result/")[1].split("?")[0]


class Check:
    """Tiny assertion collector so one run reports every failure."""

    def __init__(self, name):
        self.name = name
        self.fails = []
        self.count = 0

    def ok(self, cond, msg):
        self.count += 1
        if cond:
            print(f"  ok   {msg}")
        else:
            print(f"  FAIL {msg}")
            self.fails.append(msg)
        return bool(cond)

    def eq(self, a, b, msg):
        return self.ok(a == b, f"{msg}: {a!r} == {b!r}")

    def finish(self, sess=None):
        if sess is not None:
            self.ok(not sess.errors, f"no JS page errors {sess.errors}")
            self.ok(not sess.console_errors,
                    f"no console errors {sess.console_errors[:5]}")
        if self.fails:
            print(f"\n{self.name}: {len(self.fails)} of {self.count} checks failed")
            for f in self.fails:
                print("   -", f)
            sys.exit(1)
        print(f"\n{self.name}: all {self.count} checks passed")


# Runs before the app: every POST's FormData / JSON fields land in
# window.__mtReqs, read back through Session.requests.
FETCH_SPY = """
window.__mtReqs = [];
(function () {
  const orig = window.fetch;
  window.fetch = function (url, opts) {
    try {
      if (opts && opts.method && String(opts.method).toUpperCase() === 'POST') {
        let fields = {};
        const b = opts.body;
        if (b instanceof FormData) {
          for (const [k, v] of b.entries()) fields[k] = (v instanceof Blob) ? '<file>' : String(v);
        } else if (typeof b === 'string') {
          try { fields = JSON.parse(b); } catch (_) { fields = { raw: b }; }
        }
        window.__mtReqs.push([String(url), fields]);
      }
    } catch (_) {}
    return orig.apply(this, arguments);
  };
})();
"""


class Session:
    """One headless browser tab on the app, with error capture."""

    def __init__(self, width=1400, height=1000, clear_storage=True):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(executable_path=CHROME)
        self.context = self.browser.new_context(
            viewport={"width": width, "height": height}, accept_downloads=True)
        self.context.add_init_script(FETCH_SPY)
        self.page = self.context.new_page()
        self.errors = []
        self.console_errors = []
        self.page.on("pageerror", lambda e: self.errors.append(str(e)))
        self.page.on("console", self._on_console)
        self.clear_storage = clear_storage

    def _on_console(self, msg):
        if msg.type == "error":
            txt = msg.text
            # A deliberate 4xx from the API is not a client bug; the app shows
            # it in the UI. Only script-level errors count.
            if "Failed to load resource" in txt:
                return
            self.console_errors.append(txt)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *a):
        self.close()

    def open(self):
        p = self.page
        p.goto(BASE + "/", wait_until="domcontentloaded")
        if self.clear_storage:
            p.evaluate("localStorage.clear()")
            p.reload(wait_until="domcontentloaded")
        p.wait_for_selector("#goBtn", state="attached")
        # Fonts and profiles load asynchronously on start.
        p.wait_for_timeout(300)
        return p

    def reload(self):
        self.page.reload(wait_until="domcontentloaded")
        self.page.wait_for_selector("#goBtn", state="attached")
        self.page.wait_for_timeout(300)

    def close(self):
        try:
            self.context.close()
            self.browser.close()
        finally:
            self._pw.stop()

    # ── app interactions ──
    def set_workflow(self, wf):
        self.page.click(f'.wf-card[data-wf="{wf}"]')

    def set_toggle(self, el_id, on):
        self.page.evaluate(
            """([id, on]) => { const el = document.getElementById(id);
                 if (!el) throw new Error('no #' + id);
                 el.checked = !!on; el.dispatchEvent(new Event('change', {bubbles:true})); }""",
            [el_id, on])

    def set_value(self, el_id, value, event="input"):
        self.page.evaluate(
            """([id, v, ev]) => { const el = document.getElementById(id);
                 if (!el) throw new Error('no #' + id);
                 el.value = v; el.dispatchEvent(new Event(ev, {bubbles:true})); }""",
            [el_id, value, event])

    def select(self, el_id, value):
        self.set_value(el_id, value, "change")

    def value(self, el_id):
        return self.page.evaluate("id => document.getElementById(id).value", el_id)

    def checked(self, el_id):
        return self.page.evaluate("id => document.getElementById(id).checked", el_id)

    def text(self, selector):
        return self.page.text_content(selector) or ""

    def visible(self, selector):
        return self.page.is_visible(selector)

    def upload(self, paths, selector="#fileInput"):
        self.page.set_input_files(selector, paths)
        # addFiles() builds thumbnails asynchronously; wait until the preview
        # or strip reflects the new count.
        self.page.wait_for_timeout(300)

    def wait_preview(self, timeout=30000):
        self.page.wait_for_selector("#previewRow", state="visible", timeout=timeout)
        self.page.wait_for_function(
            "() => document.getElementById('uploadNote').style.display === 'none'"
            " || !document.getElementById('uploadNote').textContent",
            timeout=timeout)

    def go(self):
        self.page.click("#goBtn")

    def strip_status(self):
        """[(index, status, task_id)] for every chip in the strip."""
        return self.page.evaluate("""() =>
            [...document.querySelectorAll('#pageStrip .pg-chip')].map((c, i) => ({
              i, active: c.classList.contains('active'),
              status: [...c.querySelector('.pg-dot').classList].filter(x => x !== 'pg-dot')[0] || '',
              src: c.querySelector('img').getAttribute('src') || '',
              uid: c.dataset.uid,
            }))""")

    def wait_all_done(self, n, timeout=300000, allow_error=False):
        """Wait until every one of n pages is done (or errored, if allowed)."""
        t0 = time.time()
        while time.time() - t0 < timeout / 1000:
            if n == 1:
                if self.visible("#pageResult"):
                    return "done"
                if allow_error and self.visible("#retryPageBtn"):
                    return "error"
            else:
                st = self.strip_status()
                if len(st) >= n:
                    finished = [s["status"] in ("done",) or
                                (allow_error and s["status"] == "error") for s in st[:n]]
                    if all(finished):
                        return "done"
            self.page.wait_for_timeout(400)
        raise TimeoutError("pages did not finish in time")

    def active_task(self):
        return task_of(self.page.get_attribute("#transImg", "src") or "")

    def click_chip(self, index):
        self.page.click(f"#pageStrip .pg-chip:nth-child({index + 1}) .pg-thumb")
        self.page.wait_for_timeout(150)

    def chip_action(self, index, act):
        self.page.click(f'#pageStrip .pg-chip:nth-child({index + 1}) button[data-act="{act}"]')
        self.page.wait_for_timeout(150)

    def download(self, click_selector, timeout=120000):
        with self.page.expect_download(timeout=timeout) as dl:
            self.page.click(click_selector)
        d = dl.value
        path = d.path()
        with open(path, "rb") as fh:
            data = fh.read()
        return d.suggested_filename, data

    def record_requests(self):
        """Start (or restart) recording every POST the page makes as
        (url, fields). Recorded INSIDE the page by wrapping window.fetch:
        Chromium does not expose the body of a multipart request that carries
        a File, so the network side never sees these fields."""
        self.page.evaluate("window.__mtReqs = []")

    @property
    def requests(self):
        return [tuple(x) for x in self.page.evaluate("window.__mtReqs || []")]

    def screenshot(self, name, full_page=True):
        os.makedirs(SHOTS, exist_ok=True)
        p = os.path.join(SHOTS, name)
        self.page.screenshot(path=p, full_page=full_page)
        return p


def zip_entries(data: bytes):
    """[(name, crc_ok, size)] for a zip, verifying every CRC."""
    zf = zipfile.ZipFile(io.BytesIO(data))
    bad = zf.testzip()
    out = []
    for zi in zf.infolist():
        out.append((zi.filename, bad != zi.filename, zi.file_size))
    return out, zf


def build_inputs():
    """Make the derived inputs (tif, webp, upside-down, photo, zip) once."""
    import cv2
    import numpy as np
    os.makedirs(INPUTS, exist_ok=True)
    want = ["page3.tif", "page4.webp", "page3_upside.png", "photo4.jpg",
            "chapter.zip", "broken.png"]
    if all(os.path.exists(os.path.join(INPUTS, w)) for w in want):
        return INPUTS
    p3, p4, p5 = [cv2.imread(x) for x in page_paths("3.jpg", "4.jpg", "5.jpg")]
    cv2.imwrite(os.path.join(INPUTS, "page3.tif"), p3)
    cv2.imwrite(os.path.join(INPUTS, "page4.webp"), p4, [cv2.IMWRITE_WEBP_QUALITY, 92])
    cv2.imwrite(os.path.join(INPUTS, "page3_upside.png"), cv2.rotate(p3, cv2.ROTATE_180))
    with open(os.path.join(INPUTS, "broken.png"), "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\nthis is not a png at all")
    # Photo of a page: the page shrunk, turned 4 degrees, on a carpet texture.
    h, w = p4.shape[:2]
    small = cv2.resize(p4, (int(w * .6), int(h * .6)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    H, W = int(sh * 1.35), int(sw * 1.45)
    rng = np.random.default_rng(3)
    bg = cv2.GaussianBlur(rng.normal(0, 1, (H, W, 3)).astype(np.float32), (0, 0), 2.5)
    bg = (bg - bg.min()) / (bg.max() - bg.min() + 1e-6)
    carpet = (np.array([70, 90, 120], np.float32) + bg * 70).clip(0, 255).astype(np.uint8)
    carpet = cv2.cvtColor(carpet, cv2.COLOR_RGB2BGR)
    M = cv2.getRotationMatrix2D((sw / 2, sh / 2), 4, 1.0)
    M[0, 2] += (W - sw) / 2
    M[1, 2] += (H - sh) / 2
    mask = np.full((sh, sw), 255, np.uint8)
    page_w = cv2.warpAffine(small, M, (W, H), flags=cv2.INTER_LINEAR)
    mask_w = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
    shadow = cv2.GaussianBlur(np.roll(mask_w, (18, 14), (0, 1)), (0, 0), 12).astype(np.float32) / 255
    photo = carpet.astype(np.float32) * (1 - .45 * shadow[..., None])
    photo = np.where(mask_w[..., None] > 0, page_w.astype(np.float32), photo)
    cv2.imwrite(os.path.join(INPUTS, "photo4.jpg"), photo.clip(0, 255).astype(np.uint8),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    with zipfile.ZipFile(os.path.join(INPUTS, "chapter.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(page_paths("3.jpg")[0], "ch01/page 2.jpg")
        zf.write(page_paths("4.jpg")[0], "ch01/page 10.jpg")
        zf.write(page_paths("5.jpg")[0], "ch01/第1話.jpg")
        zf.writestr("ch01/notes.txt", "not an image")
        zf.writestr("__MACOSX/ch01/._page 2.jpg", "junk")
    return INPUTS
