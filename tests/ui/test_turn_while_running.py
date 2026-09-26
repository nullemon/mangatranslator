"""Turning every page while a batch is running: queued pages run on the
turned file, the page that was mid-run drops its stale result instead of
showing an un-turned page under a turned thumbnail, and nothing polls
/api/status/null for ever. Clean (remove text) so a page takes long enough
to be caught mid-run (LaMa, ~40 s each on CPU)."""
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import (Check, Session, decode, fetch, page_paths,
                      skip_unless_server, task_of, zip_entries)

skip_unless_server()
c = Check("turn while running")
P3, P4, P5 = page_paths("3.jpg", "4.jpg", "5.jpg")

with Session() as s:
    p = s.page
    null_polls = []
    p.on("request", lambda r: null_polls.append(r.url) if "/api/status/null" in r.url else None)
    s.set_workflow("clean")
    s.set_toggle("hdUpscale", False)
    s.upload([P3, P4, P5])
    s.wait_preview()
    s.go()
    p.wait_for_function("() => (document.getElementById('progressMsg').textContent || '') !== ''")
    p.wait_for_timeout(3000)                       # page 1 is well into its run
    c.eq(s.strip_status()[0]["status"], "processing", "page 1 is mid-run")

    s.set_toggle("orientAll", True)
    p.click("#orientRight")
    p.wait_for_function("() => !document.getElementById('orientRight').disabled", timeout=120000)
    st = s.strip_status()
    c.eq(st[0]["status"], "processing", "the running page keeps running")
    c.ok(st[1]["status"] == "queued" and st[2]["status"] == "queued", f"queued pages stay queued {[x['status'] for x in st]}")

    # the running page finishes: its result is of the OLD file and is dropped
    t0 = time.time()
    while time.time() - t0 < 400 and s.strip_status()[0]["status"] == "processing":
        p.wait_for_timeout(500)
    st = s.strip_status()
    c.eq(st[0]["status"], "pending", "mid-run page drops the stale result and goes back to pending")
    s.click_chip(0)
    c.ok("changed while it ran" in s.text("#progressMsg"), f"it says why: {s.text('#progressMsg')!r}")
    c.ok(not s.visible("#pageResult"), "no un-turned result shown for the turned page")

    # the queued pages run on the turned files
    t0 = time.time()
    while time.time() - t0 < 600 and not all(x["status"] == "done" for x in s.strip_status()[1:]):
        p.wait_for_timeout(1000)
    st = s.strip_status()
    for i, src in ((1, P4), (2, P5)):
        r = decode(fetch(f"/api/result/{task_of(st[i]['src'])}"))
        o = cv2.imread(src)
        c.eq(r.shape[:2], (o.shape[1], o.shape[0]), f"page {i + 1} result has the turned page's shape")
    c.ok(not null_polls, f"no polling of /api/status/null ({len(null_polls)})")

    # and Save pages as ZIP has every page turned
    name, data = s.download("#savePagesBtn")
    entries, zf = zip_entries(data)
    dims = [decode(zf.read(n)).shape[:2] for n, _, _ in entries]
    c.ok(all(d == (1028, 1500) for d in dims), f"every saved page is turned {dims}")

    # the pending page can be run from here: the retry button doubles as Run
    s.click_chip(0)
    c.ok(s.visible("#retryPageBtn") and s.text("#retryPageBtn").strip() == "Run this page",
         f"pending page offers a Run button ({s.text('#retryPageBtn').strip()!r})")
    p.click("#retryPageBtn")
    p.wait_for_timeout(500)
    c.eq(s.strip_status()[0]["status"], "processing", "Run queues and starts the page")
    t0 = time.time()
    while time.time() - t0 < 400 and s.strip_status()[0]["status"] != "done":
        p.wait_for_timeout(500)
    st = s.strip_status()
    c.eq(st[0]["status"], "done", "the turned page ran")
    r = decode(fetch(f"/api/result/{task_of(st[0]['src'])}"))
    c.eq(r.shape[:2], (1028, 1500), "and its result is of the turned file")
    c.eq(s.active_task(), task_of(st[0]["src"]), "its result is on screen")

c.finish(s)
