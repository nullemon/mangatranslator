"""The no-translation image workflows, alone and chained: Scan -> Raw, Upscale
HD, Clean (remove text), Cut out pages; then one after another on the same
page (Use Result as Input), with the cards switched between runs, and with
the settings of one run not leaking into the next.

Needs the app with the image stub for the Raw -> Scan steps:

    MT_URL=http://127.0.0.1:8041 MT_STUB=http://127.0.0.1:8141 \\
        python tests/ui/test_image_chains.py

Upscale HD runs the real model on CPU, so it is fed small pages; set
MT_SKIP_SLOW=1 to leave out the Upscale HD and Clean (LaMa) runs.
"""
import json
import os
import sys
import urllib.request

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import (INPUTS, Check, Session, build_inputs, decode, fetch,
                      page_paths, skip_unless_server, task_of)

skip_unless_server()
STUB = os.environ.get("MT_STUB", "http://127.0.0.1:8141")
try:
    urllib.request.urlopen(STUB + "/__reset", timeout=5).read()
except Exception:
    print(f"SKIP: no stub image API at {STUB} (run tests/ui/stub_image_api.py)")
    sys.exit(0)
SLOW = os.environ.get("MT_SKIP_SLOW", "0") != "1"

inputs = build_inputs()
c = Check("image chains")
P3, P4 = page_paths("3.jpg", "4.jpg")
SMALL = os.path.join(INPUTS, "small3.png")
if not os.path.exists(SMALL):
    cv2.imwrite(SMALL, cv2.resize(cv2.imread(P3), (411, 600), interpolation=cv2.INTER_AREA))


def fresh(s):
    if s.visible("#retryBtn"):
        s.page.click("#retryBtn")
    if s.visible("#newBtn"):
        s.page.click("#newBtn")
    if s.visible("#clearBtn"):
        s.page.click("#clearBtn")


def start(s, wf, paths, setup=None, n=None, allow_error=False):
    s.set_workflow(wf)
    if setup:
        setup()
    s.record_requests()
    s.upload(paths)
    s.wait_preview()
    s.go()
    return s.wait_all_done(n or len(paths), timeout=1800000, allow_error=allow_error)


def run(s, wf, paths, setup=None, n=None, allow_error=False):
    fresh(s)
    return start(s, wf, paths, setup, n, allow_error)


def status_of(tid):
    return json.loads(fetch(f"/api/status/{tid}"))


def last_req(s, part):
    got = [r for r in s.requests if part in r[0]]
    return got[-1][1] if got else {}


def reuse(s):
    """↺ Use Result as Input, then wait for the upload preview."""
    s.page.click("#reuseBtn")
    s.page.wait_for_selector("#previewRow", state="visible", timeout=30000)
    s.page.wait_for_timeout(300)


with Session() as s:
    p = s.page
    s.set_value("watermark", "@chain")
    s.select("wmPlace", "br")

    # ── Scan -> Raw: size kept, page roughed, watermark stamped ──
    run(s, "scan-raw", [P3], lambda: (s.select("rawStyle", "scan"), s.set_value("rawStrength", "1.4")))
    f = last_req(s, "/api/rawify")
    c.eq((f.get("style"), f.get("strength"), f.get("watermark")), ("scan", "1.4", "@chain"),
         "Scan -> Raw request: style, strength, watermark")
    tid = s.active_task()
    o = cv2.imread(P3)
    raw = decode(fetch(f"/api/result/{tid}"))
    raw_clean = decode(fetch(f"/api/result/{tid}?watermark=0"))
    c.eq(raw.shape[:2], o.shape[:2], "Scan -> Raw keeps the page size")
    paper = raw_clean[o.max(axis=2) > 245]
    c.ok(paper[:, 2].mean() > paper[:, 0].mean() + 5, "aged scan: paper goes tan (red > blue)")
    c.ok(not np.array_equal(raw, raw_clean), "Scan -> Raw is watermarked, clean twin kept")
    c.eq(s.text("#compLabelLeft") + "/" + s.text("#compLabelRight"), "Clean/Raw feel", "Scan -> Raw labels")
    c.ok(s.visible("#rawIntensityPanel"), "raw style + intensity shown for Scan -> Raw")

    # ── then Raw -> Scan on that very result (Use Result as Input) ──
    reuse(s)
    s.select("enhanceProvider", "gemini")
    s.set_value("enhanceKey", "stub-gemini-key")
    s.set_value("enhanceModel", "", "change")
    s.set_toggle("hdUpscale", False)
    s.set_workflow("raw-scan")
    c.ok(not s.visible("#rawIntensityPanel"), "switching card hides the raw controls")
    c.ok(s.visible("#enhancePanel"), "…and shows the scan settings")
    s.record_requests()
    s.go()
    s.wait_all_done(1, timeout=600000)
    f = last_req(s, "/api/enhance")
    c.ok("strength" not in f and "style" not in f, "the raw run's style/strength do not leak into the scan request")
    c.eq(f.get("watermark"), "@chain", "the typed mark still applies to the scan")
    fed = decode(fetch(f"/api/original/{s.active_task()}"))
    c.ok(np.array_equal(fed, raw_clean),
         "Use Result as Input feeds the UNSTAMPED result (no mark baked into the next step)")
    scan = decode(fetch(f"/api/result/{s.active_task()}"))
    c.eq(scan.shape[:2], o.shape[:2], "scan of the raw result: the page's size again")
    c.eq(s.text("#compLabelRight"), "Manga Scan", "the new page is labelled by ITS workflow")

    # ── Upscale HD alone, then Raw -> Scan on the HD page ──
    if SLOW:
        run(s, "upscale-only", [SMALL])
        f = last_req(s, "/api/upscale")
        c.eq(f.get("watermark"), "@chain", "Upscale HD request carries the mark")
        c.ok("provider" not in f and "api_key" not in f, "Upscale HD sends no scan provider or key")
        tid = s.active_task()
        st = status_of(tid)
        if st["status"] == "done":
            up = decode(fetch(f"/api/result/{tid}"))
            c.eq(up.shape[:2], (2400, 1644), "Upscale HD: 411x600 -> 1644x2400 (model x4, under 3600)")
            c.ok(not np.array_equal(up, decode(fetch(f"/api/result/{tid}?watermark=0"))),
                 "Upscale HD is watermarked")
            c.eq(s.text('.tab[data-tab="translated"]'), "HD", "Upscale HD tab label")
            reuse(s)
            s.set_workflow("raw-scan")
            s.record_requests()
            s.go()
            s.wait_all_done(1, timeout=600000)
            scan = decode(fetch(f"/api/result/{s.active_task()}"))
            c.eq(scan.shape[:2], (2400, 1644), "Raw -> Scan of the HD page keeps the HD size (big in, big out)")
            f = last_req(s, "/api/enhance")
            c.ok("upscale" not in f, "HD toggle off: the scan does not ask for another upscale")
        else:
            c.ok("Can't upscale" in st.get("message", ""), f"no model: says why ({st.get('message')!r})")

    # ── cards switched between runs: each page keeps its own labels ──
    run(s, "watermark-only", [P3, P4])
    s.set_workflow("raw-scan")          # set up the next job…
    s.click_chip(0)
    c.eq(s.text("#compLabelRight"), "Watermarked", "…a finished page keeps its own label")
    c.eq(s.text('.tab[data-tab="translated"]'), "Stamped", "…and its own tab name")

    # ── Clean (remove text): the translate pipeline's clean, no key ──
    if SLOW:
        s.set_value("watermark", "")
        run(s, "clean", [P3], lambda: (s.set_toggle("hdUpscale", False), s.select("pageFinish", "clean")))
        f = last_req(s, "/api/translate")
        c.eq(f.get("clean_only"), "true", "Clean sends clean_only")
        c.ok("api_key" not in f, "Clean needs no key")
        tid = s.active_task()
        cl = decode(fetch(f"/api/result/{tid}"))
        c.eq(cl.shape[:2], o.shape[:2], "Clean keeps the page size")
        c.ok(np.array_equal(cl, decode(fetch(f"/api/result/{tid}?watermark=0"))),
             "no mark typed: Clean stamps nothing")
        c.eq(s.text("#compLabelRight"), "Cleaned", "Clean labels")
        c.ok(not s.visible('.tab[data-tab="details"]'), "no Details tab on a cleaned page")

    # ── Cut out pages: quick re-check, and it never stamps ──
    s.set_value("watermark", "@chain")
    run(s, "cut-pages", [os.path.join(inputs, "photo4.jpg")])
    f = last_req(s, "/api/cutpage")
    c.ok("watermark" not in f, "Cut out pages sends no watermark (fix-up step)")
    cut = decode(fetch(f"/api/result/{s.active_task()}"))
    photo = cv2.imread(os.path.join(inputs, "photo4.jpg"))
    c.ok(cut.shape[0] < photo.shape[0] * 0.9, f"the page is cut out of the photo {cut.shape[:2]}")

c.finish(s)
