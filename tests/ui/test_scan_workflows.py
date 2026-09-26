"""The AI scan workflows end to end, against the stub image API: Raw -> Scan
(Gemini and Grok), AI Scan -> HD, the scan half of Raw -> Scan -> Translate,
their refusals and failures, Compress Output, big in / big out, TIFF and
WebP inputs, and the watermark rules.

Needs the app started with its providers pointed at the stub:

    python tests/ui/stub_image_api.py 8141 &
    python tests/ui/stub_llm.py 8142 &
    GEMINI_BASE_URL=http://127.0.0.1:8141 XAI_BASE_URL=http://127.0.0.1:8141 \\
    OPENAI_BASE_URL=http://127.0.0.1:8141 ANTHROPIC_BASE_URL=http://127.0.0.1:8142 \\
        PORT=8041 python app.py &
    MT_URL=http://127.0.0.1:8041 MT_STUB=http://127.0.0.1:8141 \\
        python tests/ui/test_scan_workflows.py

Set MT_SCAN_TRANSLATE=0 to skip the (slow, LaMa-bound) Raw -> Scan ->
Translate part.
"""
import json
import os
import sys
import urllib.request

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import (Check, Session, build_inputs, decode, fetch, page_paths,
                      skip_unless_server, task_of, zip_entries)

skip_unless_server()
STUB = os.environ.get("MT_STUB", "http://127.0.0.1:8141")
try:
    urllib.request.urlopen(STUB + "/__reset", timeout=5).read()
except Exception:
    print(f"SKIP: no stub image API at {STUB} (run tests/ui/stub_image_api.py)")
    sys.exit(0)

inputs = build_inputs()
c = Check("scan workflows")
BIG = os.environ.get("MT_BIG_PAGE", "/tmp/claude-0/ch/raw/p-0072.jpg")
if not os.path.exists(BIG):
    BIG = page_paths("4.jpg")[0]
P3, P4 = page_paths("3.jpg", "4.jpg")


def stub_log():
    return json.loads(urllib.request.urlopen(STUB + "/__log", timeout=10).read())


def stub_reset():
    urllib.request.urlopen(STUB + "/__reset", timeout=10).read()


def similar(a, b):
    """Correlation of two pages' grey levels at a common small size —
    high for the same page the same way up, low for a turned / mirrored /
    blank one."""
    ga = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (200, 290)).astype(np.float32)
    gb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (200, 290)).astype(np.float32)
    ga -= ga.mean()
    gb -= gb.mean()
    return float((ga * gb).sum() / (np.sqrt((ga * ga).sum() * (gb * gb).sum()) + 1e-6))


def check_page(tag, out, src):
    c.ok(out is not None, f"[{tag}] output decodes")
    if out is None:
        return
    c.ok(float(out.std()) > 25, f"[{tag}] output is not blank (std {out.std():.1f})")
    same = similar(out, src)
    flips = max(similar(out, cv2.rotate(src, cv2.ROTATE_180)), similar(out, cv2.flip(src, 1)))
    c.ok(same > 0.6 and same > flips + 0.2,
         f"[{tag}] output is the same page the same way up (corr {same:.2f} vs turned {flips:.2f})")


def fresh(s):
    if s.visible("#newBtn"):
        s.page.click("#newBtn")
    if s.visible("#clearBtn"):
        s.page.click("#clearBtn")


def run(s, wf, paths, setup=None, n=None, allow_error=False):
    fresh(s)
    s.set_workflow(wf)
    if setup:
        setup()
    s.record_requests()
    s.upload(paths)
    s.wait_preview()
    s.go()
    return s.wait_all_done(n or len(paths), timeout=900000, allow_error=allow_error)


def result_of(tid, wm=True):
    return decode(fetch(f"/api/result/{tid}" + ("" if wm else "?watermark=0")))


def status_of(tid):
    return json.loads(fetch(f"/api/status/{tid}"))


def upscaler_ok():
    h = json.loads(fetch("/api/health"))
    return bool(h.get("upscale_spandrel"))


with Session() as s:
    p = s.page
    s.set_value("watermark", "")
    s.set_value("credit", "")

    # ── the panel and the key check ──
    s.set_workflow("raw-scan")
    c.ok(s.visible("#enhancePanel"), "Raw -> Scan shows the scan settings")
    s.select("enhanceProvider", "gemini")
    s.set_value("enhanceKey", "")
    fresh(s)
    s.upload([P3])
    s.wait_preview()
    s.record_requests()
    s.go()
    p.wait_for_timeout(500)
    c.ok(not s.requests, "no key: nothing is sent")
    c.ok(p.evaluate("document.getElementById('enhanceKey').style.borderColor") != "",
         "no key: the key box is flagged")

    # ── Raw -> Scan, Gemini, a big page: exact source size, no mark ──
    stub_reset()

    def gem():
        s.select("enhanceProvider", "gemini")
        s.set_value("enhanceKey", "stub-gemini-key")
        s.set_value("enhanceModel", "", "change")
        s.set_value("enhancePrompt", "QA PROMPT: make it a clean scan", "change")
        s.set_toggle("hdUpscale", False)
        s.set_toggle("compressOut", False)
        s.select("tileMode", "1")
    run(s, "raw-scan", [BIG], gem)
    tid = s.active_task()
    url, fields = [r for r in s.requests if "/api/enhance" in r[0]][-1]
    c.eq(fields.get("provider"), "gemini", "request: provider")
    c.eq(fields.get("api_key"), "stub-gemini-key", "request: key")
    c.eq(fields.get("prompt"), "QA PROMPT: make it a clean scan", "request: the edited prompt")
    c.ok("upscale" not in fields, "request: no upscale with HD off")
    c.eq(fields.get("compress"), "false", "request: compress off")
    c.ok("watermark" not in fields, "request: no watermark field when none is typed")
    log = stub_log()
    c.ok(log and log[-1]["api"] == "gemini" and log[-1]["key"] == "stub-gemini-key"
         and log[-1]["prompt"] == "QA PROMPT: make it a clean scan"
         and log[-1]["model"] == "gemini-2.5-flash-image",
         f"the provider got the key, the prompt and the default model {log[-1:] and {k: log[-1][k] for k in ('api', 'model', 'key')}}")
    src = cv2.imread(BIG)
    out = result_of(tid)
    c.eq(out.shape[:2], src.shape[:2], "big in, big out: the scan is the source's exact size")
    check_page("gemini", out, src)
    st = status_of(tid)
    if upscaler_ok() and not st.get("warning"):
        c.ok(True, "rebuilt with the upscaler")
    else:
        c.ok("No HD upscale" in (st.get("warning") or ""),
             f"no upscale model: the page says so ({st.get('warning')!r})")
        c.ok(s.visible("#resultNote") and "No HD upscale" in s.text("#resultNote"),
             "…and the note is on screen beside the result")
    c.ok(np.array_equal(out, result_of(tid, wm=False)), "no watermark typed: nothing stamped")
    c.eq(s.text('.tab[data-tab="translated"]'), "Scan", "tab label")
    c.eq(s.text("#compLabelLeft"), "Rough", "compare left label")
    c.eq(s.text("#compLabelRight"), "Manga Scan", "compare right label")
    c.ok(not s.visible('.tab[data-tab="details"]'), "no Details tab on a scan")
    c.ok(s.visible("#translateScanBtn"), "'Translate This Scan' is offered")

    # tabs, full view and the comparison slider
    p.click('.tab[data-tab="original"]')
    c.ok(s.visible("#origFull"), "Original tab shows the full original")
    p.wait_for_function("() => document.getElementById('origFull').naturalWidth > 0")
    c.eq(p.evaluate("[origFull.naturalWidth, origFull.naturalHeight]"),
         [src.shape[1], src.shape[0]], "Original tab shows the upload at full size")
    p.click('.tab[data-tab="translated"]')
    c.ok(s.visible("#transFull"), "Scan tab shows the full result")
    p.click('.tab[data-tab="compare"]')
    p.wait_for_timeout(200)
    # Grab the picture itself (not the handle), as a user does: a press on an
    # <img> used to start the browser's image drag and freeze the split.
    p.locator("#comparisonContainer").scroll_into_view_if_needed()
    box = p.locator("#comparisonContainer").bounding_box()
    for a, b in ((0.7, 0.2), (0.3, 0.8)):
        y = box["y"] + 150
        p.mouse.move(box["x"] + box["width"] * a, y)
        p.mouse.down()
        p.mouse.move(box["x"] + box["width"] * b, y, steps=8)
        p.mouse.up()
        w = p.evaluate("parseFloat(document.getElementById('compOverlay').style.width)")
        c.ok(abs(w - b * 100) < 3, f"dragging the picture from {a:.0%} to {b:.0%} moves the split ({w:.1f}%)")

    # ── Raw -> Scan, Grok, a TIFF: aspect restore, exact size, watermark ──
    stub_reset()
    tif = os.path.join(inputs, "page3.tif")

    def grok():
        s.select("enhanceProvider", "xai")
        s.set_value("enhanceKey", "stub-xai-key")
        s.set_value("enhanceModel", "", "change")
        s.set_value("watermark", "@scanqa")
        s.select("wmStyle", "clean")
        s.select("wmPlace", "br")
    run(s, "raw-scan", [tif], grok)
    tid = s.active_task()
    url, fields = [r for r in s.requests if "/api/enhance" in r[0]][-1]
    c.eq(fields.get("provider"), "xai", "request: xai provider")
    c.eq(fields.get("watermark"), "@scanqa", "request: typed watermark is sent")
    log = stub_log()
    c.ok(log and log[-1]["api"] == "xai" and log[-1]["key"] == "Bearer stub-xai-key"
         and log[-1]["resolution"] == "2k" and log[-1]["response_format"] == "b64_json",
         "Grok got the key, the 2k tier and b64 output")
    ret = log[-1].get("returned") or [0, 0]
    src = cv2.imread(tif)
    c.ok(abs(ret[0] / ret[1] - src.shape[1] / src.shape[0]) > 0.01,
         f"the stub really answered with the wrong aspect ({ret})")
    out = result_of(tid)
    clean = result_of(tid, wm=False)
    c.eq(out.shape[:2], src.shape[:2], "TIFF through Grok: exact source size")
    check_page("grok tif", clean, src)
    m = np.any(out != clean, axis=2)
    h, w_ = m.shape
    c.ok(m.sum() > 0 and m[h // 2:, w_ // 2:].sum() == m.sum(),
         "watermark stamped, bottom-right only, clean twin kept")

    # ── a refusal: the page still lands, and says what happened ──
    s.set_value("watermark", "")
    stub_reset()

    def refuse():
        s.select("enhanceProvider", "xai")
        s.set_value("enhanceModel", "stub-refuse", "change")
    run(s, "raw-scan", [P3, P4], refuse, n=2)
    chips = s.strip_status()
    c.ok(all(ch["status"] == "done" for ch in chips), "refused pages still finish (local fallback)")
    c.ok(p.evaluate("document.querySelectorAll('#pageStrip .pg-dot.warn').length") == 2,
         "both chips are marked with a warning")
    s.click_chip(1)
    c.ok(s.visible("#resultNote") and "refused" in s.text("#resultNote"),
         f"the refusal is shown beside the result: {s.text('#resultNote')!r}")
    c.ok("Xai" not in s.text("#resultNote"), "the provider is named properly")
    out = result_of(s.active_task())
    c.eq(out.shape[:2], cv2.imread(P4).shape[:2], "fallback page keeps the size")

    # a bad key reads as a failure, not a refusal
    def badkey():
        s.select("enhanceProvider", "gemini")
        s.set_value("enhanceModel", "stub-badkey", "change")
    run(s, "raw-scan", [P3], badkey)
    note = s.text("#resultNote")
    c.ok("API key not valid" in note and "failed" in note, f"bad key is explained: {note!r}")

    # Gemini's own refusal (finishReason IMAGE_SAFETY) is a refusal too
    def gsafety():
        s.select("enhanceProvider", "gemini")
        s.set_value("enhanceModel", "stub-refuse", "change")
    run(s, "raw-scan", [P3], gsafety)
    c.ok("refused" in s.text("#resultNote"), f"Gemini IMAGE_SAFETY reads as a refusal: {s.text('#resultNote')!r}")

    # an oversized answer comes back down to the page's size
    def oversize():
        s.select("enhanceProvider", "gemini")
        s.set_value("enhanceModel", "stub-oversize", "change")
    run(s, "raw-scan", [P3], oversize)
    out = result_of(s.active_task())
    c.eq(out.shape[:2], cv2.imread(P3).shape[:2], "oversized provider answer comes down to the page size")
    check_page("oversize", out, cv2.imread(P3))
    c.ok(not s.visible("#resultNote"), "a clean run shows no note")

    # ── Compress Output: provider size kept, real JPEG, named .jpg ──
    def compressed():
        s.select("enhanceProvider", "gemini")
        s.set_value("enhanceModel", "", "change")
        s.set_toggle("compressOut", True)
    run(s, "raw-scan", [BIG], compressed)
    tid = s.active_task()
    data = fetch(f"/api/result/{tid}")
    c.ok(data[:3] == b"\xff\xd8\xff", "Compress Output: the scan is a JPEG")
    out = decode(data)
    src = cv2.imread(BIG)
    c.ok(max(out.shape[:2]) < max(src.shape[:2]), f"Compress Output keeps the provider's smaller size {out.shape[:2]}")
    c.ok(abs(out.shape[1] / out.shape[0] - src.shape[1] / src.shape[0]) < 0.01,
         "…on the page's own aspect (Gemini's bucket stretch undone)")
    name, _ = s.download("#downloadBtn")
    c.ok(name.endswith(".jpg"), f"download named .jpg ({name})")
    s.set_toggle("compressOut", False)

    # ── WebP through Raw -> Scan ──
    webp = os.path.join(inputs, "page4.webp")
    run(s, "raw-scan", [webp])
    out = result_of(s.active_task())
    c.eq(out.shape[:2], cv2.imread(webp).shape[:2], "WebP: exact source size")
    check_page("webp", out, cv2.imread(webp))

    # ── AI Scan -> HD ──
    stub_reset()
    run(s, "scan-upscale", [P3])
    tid = s.active_task()
    url, fields = [r for r in s.requests if "/api/enhance" in r[0]][-1]
    c.eq(fields.get("upscale"), "true", "AI Scan -> HD asks for the upscale")
    out = result_of(tid)
    src = cv2.imread(P3)
    st = status_of(tid)
    if st.get("warning"):
        c.ok("No HD upscale" in st["warning"], f"no model: says so ({st['warning']!r})")
        c.eq(out.shape[:2], src.shape[:2], "no model: the page's own size, not a blown-up 1K scan")
    else:
        c.eq(max(out.shape[:2]), 3600, "HD: long edge 3600")
        c.ok(abs(out.shape[1] / out.shape[0] - src.shape[1] / src.shape[0]) < 0.005,
             f"HD keeps the page's aspect {out.shape[:2]}")
    check_page("scan+hd", out, src)
    c.eq(s.text('.tab[data-tab="translated"]'), "HD", "AI Scan -> HD tab label")
    c.eq(s.text("#compLabelRight"), "Scan + HD", "AI Scan -> HD compare label")

    # ── Raw -> Scan -> Translate: the scan half ──
    if os.environ.get("MT_SCAN_TRANSLATE", "1") != "0":
        stub_reset()

        def rst():
            s.select("engine", "claude")
            s.set_value("apiKey", "stub-claude-key")
            s.select("enhanceProvider", "xai")
            s.set_value("enhanceModel", "stub-refuse", "change")
        run(s, "raw-scan-translate", [P3], rst)
        tid = s.active_task()
        url, fields = [r for r in s.requests if "/api/translate" in r[0]][-1]
        c.eq(fields.get("enhance"), "true", "scan+translate: enhance on")
        c.eq(fields.get("enhance_provider"), "xai", "scan+translate: provider sent")
        c.eq(fields.get("enhance_key"), "stub-xai-key", "scan+translate: scan key sent")
        c.eq(fields.get("enhance_model"), "stub-refuse", "scan+translate: model sent")
        st = status_of(tid)
        c.ok("refused" in (st.get("warning") or ""), f"scan+translate keeps the refusal note ({st.get('warning')!r})")
        c.ok("refused" in s.text("#resultNote"), "…and shows it beside the translated page")
        enh = decode(fetch(f"/api/enhanced/{tid}"))
        c.eq(enh.shape[:2], src.shape[:2], "the scanned base is the source's size")
        # and a successful scan: the base is the stub's (lightened) scan
        s.set_value("enhanceModel", "", "change")
        run(s, "raw-scan-translate", [P3])
        tid = s.active_task()
        st = status_of(tid)
        c.ok(not st.get("warning"), f"a clean scan+translate has no warning ({st.get('warning')!r})")
        enh = decode(fetch(f"/api/enhanced/{tid}"))
        c.eq(enh.shape[:2], src.shape[:2], "AI-scanned base snapped to the source's size")
        check_page("scan+translate base", enh, src)
        c.ok(stub_log()[-1]["api"] == "xai", "the scan half went to Grok")

        # a font added while a translated page is open reaches its per-line pickers
        n_lines = p.evaluate("document.querySelectorAll('#translationsList select.tl-font').length")
        if n_lines:
            import shutil
            import tempfile
            from _harness import ROOT
            tmp = tempfile.mkdtemp()
            face = os.path.join(tmp, "zz-qa-perline-face.ttf")
            shutil.copyfile(os.path.join(ROOT, "fonts", "Bangers-Regular.ttf"), face)
            p.set_input_files("#fontUpload", [face])
            p.wait_for_function("() => [...document.querySelectorAll('#fontSelect option')].some(o => o.value === 'zz-qa-perline-face.ttf')",
                                timeout=30000)
            p.wait_for_timeout(300)
            has = p.evaluate("""() => [...document.querySelectorAll('#translationsList select.tl-font')]
                .every(sel => [...sel.options].some(o => o.value === 'zz-qa-perline-face.ttf'))""")
            c.ok(has, f"the new font is offered in all {n_lines} per-line pickers")
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            c.ok(False, "the stub translation produced lines to test the per-line font picker on")

c.finish(s)
