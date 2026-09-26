"""Upload paths: many images, a ZIP, .tif / .webp, ordering, remove, retry, clear."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import (Check, Session, build_inputs, decode, fetch, page_paths,
                      skip_unless_server, task_of)

skip_unless_server()
inputs = build_inputs()
c = Check("upload paths")

with Session() as s:
    p = s.page

    # ── 1. several images, given out of order → natural sort by name ──
    s.upload(page_paths("5.jpg", "3.jpg", "4.jpg"))
    s.wait_preview()
    c.ok("3 pages selected" in s.text("#fileName"), "three pages selected")
    c.ok("3.jpg, 4.jpg, 5.jpg" in s.text("#fileSize"), "sorted by filename")
    c.ok(p.evaluate("document.getElementById('previewImg').naturalWidth") > 0,
         "preview thumbnail decodes")

    # ── 2. clear all ──
    p.click("#clearBtn")
    c.ok(s.visible("#dropZone") and not s.visible("#previewRow"), "Clear returns to the drop zone")

    # ── 3. a ZIP: expanded server-side, junk skipped, non-ASCII name kept ──
    s.upload([os.path.join(inputs, "chapter.zip")])
    s.wait_preview(timeout=60000)
    c.ok("3 pages selected" in s.text("#fileName"), "zip → 3 image pages (txt and __MACOSX skipped)")
    names = s.text("#fileSize")
    c.ok("page 2.jpg, page 10.jpg" in names, f"zip pages natural-sorted: {names}")
    c.ok("第1話.jpg" in names, "non-ASCII zip member name survives")
    p.click("#clearBtn")

    # ── 4. TIFF and WebP: TIFF is converted through /api/topng, WebP is native ──
    s.upload([os.path.join(inputs, "page3.tif"), os.path.join(inputs, "page4.webp")])
    s.wait_preview(timeout=60000)
    c.ok("2 pages selected" in s.text("#fileName"), "tif + webp accepted")
    c.ok("page3.png" in s.text("#fileSize"), "tif renamed to .png after conversion")
    # The stand-in is capped at 1100px on its long edge (makeThumb), so the
    # 1028x1500 page previews at 754x1100.
    tw, th = p.evaluate("[previewImg.naturalWidth, previewImg.naturalHeight]")
    c.ok(tw > 0 and abs(tw / th - 1028 / 1500) < 0.01,
         f"converted TIFF thumbnail decodes with the page's aspect ({tw}x{th})")

    # Run them through Watermark-only (fast, no models) to check the strip
    # and the per-page result belong to the right files.
    s.set_workflow("watermark-only")
    s.set_value("watermark", "@qa")
    s.go()
    s.wait_all_done(2, timeout=120000)
    st = s.strip_status()
    c.eq([x["status"] for x in st], ["done", "done"], "both pages done")
    a, b = [decode(fetch("/api/result/" + task_of(x["src"]))) for x in st]
    c.ok(a is not None and a.shape[:2] == (1500, 1028), f"tif page result decodes {a.shape}")
    c.ok(b is not None and b.shape[:2] == (1500, 1028), f"webp page result decodes {b.shape}")

    # ── 5. remove a page from the strip ──
    s.click_chip(1)
    c.ok(s.strip_status()[1]["active"], "second chip is active after click")
    s.chip_action(0, "remove")
    c.ok(not s.visible("#pageStrip"), "strip hides when one page remains")
    c.eq(s.active_task(), task_of(st[1]["src"]), "result shown is the surviving page")

    # ── 6. add pages to a running batch, then remove the ACTIVE page ──
    s.upload(page_paths("5.jpg", "6.jpg"))
    p.wait_for_timeout(500)
    s.wait_all_done(3, timeout=120000)
    st = s.strip_status()
    c.eq(len(st), 3, "two pages added to the batch")
    c.eq([x["status"] for x in st], ["done"] * 3, "added pages ran through")
    s.click_chip(1)
    mid_task = task_of(s.strip_status()[1]["src"])
    c.eq(s.active_task(), mid_task, "active result = clicked chip")
    s.chip_action(1, "remove")
    st = s.strip_status()
    c.eq(len(st), 2, "active page removed")
    c.ok(st[1]["active"], "the page that slid into its slot is now active")
    c.eq(s.active_task(), task_of(st[1]["src"]), "result image follows the new active page")

    # ── 7. reorder with ‹ › ──
    s.chip_action(1, "left")
    st2 = s.strip_status()
    c.eq([x["uid"] for x in st2], [st[1]["uid"], st[0]["uid"]], "move-left swaps the chips")
    c.eq(s.text("#pageStrip .pg-chip:nth-child(1) .pg-idx"), "1", "index labels renumber")

    # ── 8. retry on a failed page ──
    p.click("#newBtn")
    c.ok(s.visible("#dropZone"), "Start Over returns to upload")
    s.set_workflow("watermark-only")
    s.set_value("watermark", "")          # the server refuses: nothing to stamp
    s.upload(page_paths("3.jpg"))
    s.wait_preview()
    s.go()
    c.eq(s.wait_all_done(1, timeout=60000, allow_error=True), "error", "page fails without a mark")
    c.ok("nothing to stamp" in s.text("#progressMsg"), f"server reason shown: {s.text('#progressMsg')!r}")
    s.set_value("watermark", "@retry")
    p.click("#retryPageBtn")
    c.eq(s.wait_all_done(1, timeout=60000, allow_error=True), "done", "Retry re-runs the page and it succeeds")
    c.ok(s.active_task() != "", "result belongs to a real task")

    # ── 9. a corrupt file: error surfaces, no JS exception ──
    p.click("#newBtn")
    s.upload([os.path.join(inputs, "broken.png")])
    s.wait_preview()
    s.go()
    c.eq(s.wait_all_done(1, timeout=60000, allow_error=True), "error", "corrupt image reports an error")

c.finish(s)
