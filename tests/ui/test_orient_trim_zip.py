"""The orientation bar (turns, every page, find upside-down, cut-out), the
Trim tools with their interactive cut, and both ZIP exports (client-side
Save pages as ZIP with CRC check; server-side Download All)."""
import io
import os
import sys
import zipfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import (Check, Session, build_inputs, decode, fetch, page_paths,
                      skip_unless_server, task_of, zip_entries)

skip_unless_server()
inputs = build_inputs()
c = Check("orientation / trim / zip")
P3, P4, P5 = page_paths("3.jpg", "4.jpg", "5.jpg")
O3, O4, O5 = [cv2.imread(x) for x in (P3, P4, P5)]


def save_zip(s):
    name, data = s.download("#savePagesBtn")
    entries, zf = zip_entries(data)
    imgs = {n: decode(zf.read(n)) for n, ok, sz in entries}
    return name, entries, imgs


def note(s):
    return s.text("#orientNote")


with Session() as s:
    p = s.page
    s.set_workflow("watermark-only")
    s.set_value("watermark", "@qa")
    s.upload([P3, P4, P5])
    s.wait_preview()
    s.go()
    s.wait_all_done(3, timeout=120000)
    c.ok(s.visible("#orientBar"), "orientation bar shows once pages are loaded")

    # ── Save pages as ZIP straight away: originals, names, CRCs ──
    name, entries, imgs = save_zip(s)
    c.eq(name, "pages.zip", "default zip name")
    c.eq([e[0] for e in entries], ["3.jpg", "4.jpg", "5.jpg"], "entries keep upload names, in strip order")
    c.ok(all(e[1] for e in entries), "every CRC verifies (zipfile.testzip)")
    c.ok(np.array_equal(imgs["3.jpg"], O3) and np.array_equal(imgs["5.jpg"], O5), "untouched pages are byte-for-byte the uploads")

    # ── ⟳ 90° on the active page only ──
    s.click_chip(0)
    s.set_toggle("orientAll", False)
    p.click("#orientRight")
    p.wait_for_function("() => !document.getElementById('orientRight').disabled", timeout=60000)
    c.ok("turned 90° right" in note(s), f"note: {note(s)!r}")
    st = s.strip_status()
    c.eq(st[0]["status"], "pending", "turned page goes back to pending (not re-run)")
    c.eq(st[1]["status"], "done", "other pages keep their result")
    name, entries, imgs = save_zip(s)
    c.eq([e[0] for e in entries], ["3.png", "4.jpg", "5.jpg"], "turned page is renamed .png (it is one now)")
    c.ok(np.array_equal(imgs["3.png"], cv2.rotate(O3, cv2.ROTATE_90_CLOCKWISE)), "page 1 turned 90° right, lossless")
    c.ok(np.array_equal(imgs["4.jpg"], O4), "page 2 untouched")

    # ── every page: 180 then mirror then 90° left ──
    s.set_toggle("orientAll", True)
    p.click("#orient180")
    p.wait_for_function("() => !document.getElementById('orient180').disabled", timeout=60000)
    c.ok("3 pages flipped 180°" in note(s), f"note: {note(s)!r}")
    p.click("#orientMirror")
    p.wait_for_function("() => !document.getElementById('orientMirror').disabled", timeout=60000)
    p.click("#orientLeft")
    p.wait_for_function("() => !document.getElementById('orientLeft').disabled", timeout=60000)
    name, entries, imgs = save_zip(s)

    def chain(o, first_cw):
        x = cv2.rotate(o, cv2.ROTATE_90_CLOCKWISE) if first_cw else o
        x = cv2.rotate(x, cv2.ROTATE_180)
        x = cv2.flip(x, 1)
        return cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE)
    c.ok(np.array_equal(imgs["3.png"], chain(O3, True)), "page 1: cw, 180, mirror, ccw applied in order")
    c.ok(np.array_equal(imgs["4.png"], chain(O4, False)), "page 2: 180, mirror, ccw")
    c.ok(np.array_equal(imgs["5.png"], chain(O5, False)), "page 3: 180, mirror, ccw")
    c.ok(all(e[1] for e in entries), "CRCs verify after turns")
    c.ok(not s.visible("#pageResult") and s.visible("#pageProcessing"), "turned active page shows its pending state")

    # ── Find & fix upside-down: no OCR here → must degrade gracefully ──
    p.click("#orientAuto")
    p.wait_for_function("() => !document.getElementById('orientAuto').disabled", timeout=180000)
    n = note(s)
    c.ok(("Couldn't read" in n) or ("look the right way up" in n) or ("Turned" in n), f"auto-check reports: {n!r}")
    c.eq(s.text("#orientAuto").strip(), "🔎 Find & fix upside-down", "button label restored")
    c.ok(all(not p.is_disabled(f"#{b}") for b in ("orient180", "trimLeft", "savePagesBtn", "cutoutBtn")),
         "orientation bar unlocked afterwards")

    # ── strip ⟳ button turns one page regardless of "every page" ──
    before = save_zip(s)[2]
    s.chip_action(2, "flip")
    p.wait_for_timeout(1500)
    after = save_zip(s)[2]
    c.ok(np.array_equal(after["5.png"], cv2.rotate(before["5.png"], cv2.ROTATE_180)), "chip ⟳ flips that page 180°")
    c.ok(np.array_equal(after["3.png"], before["3.png"]), "chip ⟳ leaves the others alone")

    # ── Trim: interactive placement on a pending page (local cut) ──
    s.click_chip(0)
    p.click("#trimLeft")
    p.wait_for_selector("#trimModal", state="visible")
    p.wait_for_function("() => document.getElementById('trimImg').naturalWidth > 0")
    box = p.locator("#trimImg").bounding_box()
    p.mouse.click(box["x"] + box["width"] * 0.10, box["y"] + box["height"] / 2)
    ro = s.text("#trimReadout")
    c.ok("off the left (10.0%)" in ro or "off the left (9.9%)" in ro or "off the left (10.1%)" in ro,
         f"clicking at 10% places the cut there: {ro!r}")
    cur = after["3.png"]
    p.click("#trimApplyOne")
    p.wait_for_selector("#trimModal", state="hidden", timeout=60000)
    c.ok("Cut" in note(s) and "off the left" in note(s), f"trim note: {note(s)!r}")
    name, entries, imgs = save_zip(s)
    frac = float(ro.split("(")[1].split("%")[0]) / 100
    exp_cut = round(frac * cur.shape[1])
    c.ok(abs(imgs["3.png"].shape[1] - (cur.shape[1] - exp_cut)) <= 1, f"width {cur.shape[1]} → {imgs['3.png'].shape[1]} (cut {exp_cut})")
    c.ok(np.array_equal(imgs["3.png"], cur[:, cur.shape[1] - imgs["3.png"].shape[1]:]), "kept pixels are the right-hand remainder, untouched")
    c.ok(np.array_equal(imgs["4.png"], after["4.png"]), "trim is one page only")

    # ── Trim on a FINISHED page: server-side, result keeps its watermark ──
    p.click("#newBtn")
    s.set_workflow("watermark-only")
    s.upload([P3, P4])
    s.wait_preview()
    s.go()
    s.wait_all_done(2, timeout=120000)
    s.click_chip(1)
    tid = s.active_task()
    res_before = decode(fetch(f"/api/result/{tid}"))
    p.click("#trimBottom")
    p.wait_for_selector("#trimModal", state="visible")
    p.wait_for_function("() => document.getElementById('trimImg').naturalWidth > 0")
    c.ok("/api/result/" in (p.get_attribute("#trimImg", "src") or ""), "trim previews the finished page")
    box = p.locator("#trimImg").bounding_box()
    p.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] * 0.85)
    ro = s.text("#trimReadout")
    frac = float(ro.split("(")[1].split("%")[0]) / 100
    p.click("#trimApplyOne")
    p.wait_for_selector("#trimModal", state="hidden", timeout=60000)
    res_after = decode(fetch(f"/api/result/{tid}"))
    c.ok(abs(res_after.shape[0] - (res_before.shape[0] - round(frac * res_before.shape[0]))) <= 1,
         f"finished page trimmed on the server {res_before.shape[:2]} → {res_after.shape[:2]}")
    c.ok(np.array_equal(res_after, res_before[:res_after.shape[0]]), "the kept part of the result is unchanged (no re-run)")
    c.eq(s.strip_status()[1]["status"], "done", "page stays done after a server trim")
    c.eq(s.active_task(), tid, "result on screen is still this page")
    name, entries, imgs = save_zip(s)
    c.eq(imgs["4.png"].shape[0], res_after.shape[0], "local copy of the page cut to match")
    twin = decode(fetch(f"/api/result/{tid}?watermark=0"))
    c.eq(twin.shape[:2], res_after.shape[:2], "clean twin trimmed too")

    # ── ✂ Cut out: every page — the photo is cut, the scan skipped ──
    p.click("#newBtn")
    s.set_workflow("watermark-only")
    s.upload([P3, os.path.join(inputs, "photo4.jpg")])
    s.wait_preview()
    s.go()
    s.wait_all_done(2, timeout=120000)
    s.set_toggle("orientAll", True)
    p.click("#cutoutBtn")
    p.wait_for_function("() => !document.getElementById('cutoutBtn').disabled", timeout=300000)
    n = note(s)
    c.ok("Cut the page out of 1 photo" in n and "1 had no page" in n, f"cut-out note: {n!r}")
    name, entries, imgs = save_zip(s)
    photo = cv2.imread(os.path.join(inputs, "photo4.jpg"))
    c.ok(np.array_equal(imgs["3.jpg"], O3), "full-frame scan left alone by the cut-out")
    c.ok(imgs["photo4.png"].shape[0] < photo.shape[0] * 0.9 and imgs["photo4.png"].shape[1] < photo.shape[1] * 0.9,
         f"photo cut to the page: {photo.shape[:2]} → {imgs['photo4.png'].shape[:2]}")
    c.eq(s.strip_status()[1]["status"], "done", "cut-out leaves the page's status alone")

    # ── server ZIP (Download All): order follows the strip, names, clean twin ──
    p.click("#newBtn")
    s.set_workflow("watermark-only")
    s.upload([os.path.join(inputs, "chapter.zip")])
    s.wait_preview(timeout=60000)
    s.go()
    s.wait_all_done(3, timeout=120000)
    s.chip_action(2, "left")           # 第1話 moves to slot 2
    order = [task_of(x["src"]) for x in s.strip_status()]
    name, data = s.download("#zipBtn")
    entries, zf = zip_entries(data)
    c.eq(name, "translated_pages.zip", "server zip default name")
    c.eq([e[0] for e in entries], ["001_page 2.png", "002_第1話.png", "003_page 10.png"], "server zip names follow the strip order")
    c.ok(all(e[1] for e in entries), "server zip CRCs verify")
    c.ok(np.array_equal(decode(zf.read("002_第1話.png")), decode(fetch(f"/api/result/{order[1]}"))), "zip page 2 is the page shown as 2")
    s.set_value("chapterName", "QA Ch")
    name, data = s.download("#zipCleanBtn")
    entries, zf = zip_entries(data)
    c.eq(name, "QA Ch (no watermark).zip", "clean zip named after the chapter")
    c.eq([e[0] for e in entries], ["QA Ch - 001.png", "QA Ch - 002.png", "QA Ch - 003.png"], "chapter-named entries")
    c.ok(np.array_equal(decode(zf.read("QA Ch - 001.png")), decode(fetch(f"/api/original/{order[0]}"))), "no-watermark zip holds the clean page")

    # ── client ZIP with a non-ASCII upload name ──
    name, entries, imgs = save_zip(s)
    c.eq(name, "QA Ch.zip", "Save pages zip takes the chapter name")
    c.eq([e[0] for e in entries], ["page 2.jpg", "第1話.jpg", "page 10.jpg"], "Save pages keeps non-ASCII names readable")
    c.ok(all(e[1] for e in entries), "Save pages CRCs verify")

c.finish(s)
