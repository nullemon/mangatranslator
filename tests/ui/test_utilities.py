"""The utility features around the workflows: System status, the Font
"+ Add" button, the Rotate-only direction, Reset defaults in the middle of a
session, the End page (styles, custom accent, before and after a run, in the
strip, reordered, in the ZIP) and the page strip's own buttons.

Needs only the app (no keys, no stubs):

    MT_URL=http://127.0.0.1:8041 python tests/ui/test_utilities.py
"""
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import (Check, ROOT, Session, build_inputs, decode, fetch,
                      page_paths, skip_unless_server, task_of, zip_entries)

skip_unless_server()
inputs = build_inputs()
c = Check("utilities")
P3, P4, P5 = page_paths("3.jpg", "4.jpg", "5.jpg")
TMP = tempfile.mkdtemp(prefix="mt_util_")

# A real font under a name nothing else uses, and two files that are not fonts.
SRC_FONT = os.path.join(ROOT, "fonts", "Bangers-Regular.ttf")
GOOD_FONT = os.path.join(TMP, "zz-qa-added-face.ttf")
shutil.copyfile(SRC_FONT, GOOD_FONT)
FAKE_FONT = os.path.join(TMP, "zz-qa-not-a-font.ttf")
shutil.copyfile(P3, FAKE_FONT)                  # a JPEG wearing a .ttf name
TXT = os.path.join(TMP, "zz-qa-notes.txt")
open(TXT, "w").write("not a font")


def fonts_listed(s):
    return s.page.evaluate("""() => ({
        select: [...document.querySelectorAll('#fontSelect option')].map(o => o.value),
        picker: [...document.querySelectorAll('#fontPicker .font-picker-item')].map(x => x.dataset.value),
        button: document.querySelector('.font-picker-btn') ? document.querySelector('.font-picker-btn').textContent : ''})""")


def fresh(s):
    if s.visible("#newBtn"):
        s.page.click("#newBtn")
    if s.visible("#clearBtn"):
        s.page.click("#clearBtn")
    if s.visible("#retryBtn"):
        s.page.click("#retryBtn")
        if s.visible("#newBtn"):
            s.page.click("#newBtn")


def chip_tasks(s):
    return [task_of(ch["src"]) for ch in s.strip_status()]


with Session() as s:
    p = s.page

    # ── System status ──
    p.click("#checkSystem")
    p.wait_for_function("() => !/checking/.test(document.getElementById('systemStatus').textContent)",
                        timeout=60000)
    st = s.text("#systemStatus")
    c.ok("build" in st and "full stack" in st and "failed" not in st,
         f"System status reports the build and components: {st[:120]!r}")

    # ── Font + Add: a real font ──
    p.wait_for_function("() => document.querySelectorAll('#fontSelect option').length > 1", timeout=15000)
    before = fonts_listed(s)
    p.set_input_files("#fontUpload", [GOOD_FONT])
    p.wait_for_function("() => [...document.querySelectorAll('#fontSelect option')].some(o => o.value === 'zz-qa-added-face.ttf')",
                        timeout=30000)
    after = fonts_listed(s)
    c.ok("zz-qa-added-face.ttf" in after["select"], "added font is in the Font dropdown")
    c.ok("zz-qa-added-face.ttf" in after["picker"], "…and in the visible font picker")
    c.eq(s.value("fontSelect"), "zz-qa-added-face.ttf", "a single added font is selected")
    c.eq(after["button"].strip(), "zz-qa-added-face", "the picker button shows the font just added")
    c.eq(p.evaluate("localStorage.getItem('manga_font')"), "zz-qa-added-face.ttf", "…and it is saved")

    # ── Font + Add: not a font / wrong type → refused, clearly ──
    p.set_input_files("#fontUpload", [FAKE_FONT])
    p.wait_for_selector("#errorSection", state="visible", timeout=30000)
    msg = s.text("#errorMsg")
    c.ok("not a usable font" in msg and "zz-qa-not-a-font.ttf" in msg,
         f"a JPEG named .ttf is refused with a reason: {msg[:140]!r}")
    c.ok("zz-qa-not-a-font.ttf" not in fonts_listed(s)["select"], "…and is not in any picker")
    c.ok(not os.path.exists(os.path.join(ROOT, "fonts", "zz-qa-not-a-font.ttf"))
         or os.environ.get("MT_REMOTE"), "…and was not written to fonts/")
    p.click("#retryBtn")
    p.set_input_files("#fontUpload", [TXT])
    p.wait_for_selector("#errorSection", state="visible", timeout=30000)
    c.ok(".ttf or .otf" in s.text("#errorMsg"), f"a .txt is refused: {s.text('#errorMsg')!r}")
    p.click("#retryBtn")
    c.eq(s.value("fontSelect"), "zz-qa-added-face.ttf", "a refused file does not change the font")

    # ── Rotate only: the direction survives a reload ──
    s.set_workflow("rotate-pages")
    s.select("rotateTurn", "ccw")
    s.reload()
    c.eq(s.value("rotateTurn"), "ccw", "Rotate only direction is remembered")
    c.ok(s.visible("#rotatePanel"), "the picker is shown for the remembered workflow")

    # ── End page made BEFORE the run: stays on the upload screen ──
    s.set_workflow("watermark-only")
    s.set_value("watermark", "@util")
    s.upload([P3, P4])
    s.wait_preview()
    s.set_value("endScan", "QA Scans")
    s.set_value("endDiscord", "discord.gg/qa")
    s.select("endTheme", "bluelock")
    s.set_toggle("endUseColor", False)
    s.record_requests()
    p.click("#endCardBtn")
    p.wait_for_function("() => document.getElementById('fileName').textContent.includes('3 pages')",
                        timeout=30000)
    c.ok(s.visible("#goBtn"), "end page made before the run: the run button is still there")
    req = [r for r in s.requests if "/api/endcard" in r[0]][-1][1]
    c.eq((req.get("scanlation"), req.get("discord"), req.get("style")),
         ("QA Scans", "discord.gg/qa", "bluelock"), "end page request: name, link, style")
    c.eq((req.get("width"), req.get("height")), ("1028", "1500"),
         "end page is sized from the page FILE, not its 1100px thumbnail")
    c.ok("accent" not in req, "no accent sent while Custom accent is off")
    s.go()
    s.wait_all_done(3, timeout=300000)
    chips = s.strip_status()
    c.eq(len(chips), 3, "two pages and the end page in the strip")
    end_tid = task_of(chips[2]["src"])
    end = decode(fetch(f"/api/result/{end_tid}"))
    c.eq(end.shape[:2], (1500, 1028), "end page has the chapter's page size")
    blue = ((end[..., 0] > 200) & (end[..., 1] > 170) & (end[..., 2] < 90)).mean()
    c.ok(blue > 0.001, f"Striker blue draws its cyan accent ({blue:.4f} of pixels)")
    s.click_chip(2)
    c.eq(s.text('.tab[data-tab="translated"]'), "End page", "end page tab label")
    c.ok(not s.visible("#translateScanBtn"), "no 'Translate This Scan' on an end page")

    # ── custom accent ──
    s.set_toggle("endUseColor", True)
    s.set_value("endColor", "#ff0000")
    s.record_requests()
    p.click("#endCardBtn")
    p.wait_for_function("() => document.querySelectorAll('#pageStrip .pg-chip').length === 4", timeout=30000)
    req = [r for r in s.requests if "/api/endcard" in r[0]][-1][1]
    c.eq(req.get("accent"), "#ff0000", "custom accent is sent")
    red_tid = task_of(s.strip_status()[3]["src"])
    red = decode(fetch(f"/api/result/{red_tid}"))
    redfrac = ((red[..., 2] > 200) & (red[..., 1] < 60) & (red[..., 0] < 60)).mean()
    c.ok(redfrac > 0.001, f"the custom accent is drawn ({redfrac:.4f} red pixels)")
    c.ok(s.page.evaluate("document.querySelector('#pageStrip .pg-chip.active')") is not None
         and s.strip_status()[3]["active"], "an end page made after the run becomes the active page")

    # ── strip: remove the second end page, move the first one to the front ──
    s.chip_action(3, "remove")
    c.eq(len(s.strip_status()), 3, "✕ removes a page from the strip")
    s.chip_action(2, "left")
    s.chip_action(1, "left")
    order = chip_tasks(s)
    c.eq(order[0], end_tid, "‹ moves the end page to the front")
    c.ok(p.evaluate("document.querySelector('#pageStrip .pg-chip:nth-child(1) button[data-act=\"flip\"]').disabled"),
         "an end page cannot be flipped (it has no upload)")
    c.ok(p.evaluate("document.querySelector('#pageStrip .pg-chip:nth-child(1) button[data-act=\"left\"]').disabled"),
         "the first page cannot move further left")

    # ── ZIP export: the end page is in it, in the strip's order ──
    s.set_value("chapterName", "QA Util")
    name, data = s.download("#zipBtn")
    entries, zf = zip_entries(data)
    names = [e[0] for e in entries]
    c.eq(name, "QA Util.zip", "zip named after the chapter")
    c.eq(names, ["QA Util - 001.png", "QA Util - 002.png", "QA Util - 003.png"], "zip entries")
    c.ok(all(ok for _, ok, _ in entries), "every zip entry passes its CRC")
    c.ok(np.array_equal(decode(zf.read(names[0])), end), "the end page is first in the zip, as in the strip")
    name, data = s.download("#zipCleanBtn")
    entries, zf = zip_entries(data)
    c.eq(len(entries), 3, "the no-watermark zip has the end page too")
    first_page = decode(zf.read(entries[1][0]))
    c.ok(np.array_equal(first_page, cv2.imread(P3)), "no-watermark zip holds the clean pages")

    # ── Reset defaults in the middle of the session ──
    s.set_value("watermark", "@will-reset")
    s.select("wmStyle", "pill")
    s.set_toggle("compressOut", True)
    s.set_toggle("hdUpscale", True)
    s.select("pageFinish", "restore")
    s.select("rotateTurn", "180")
    s.select("engine", "claude")
    s.set_value("apiKey", "sk-keep-me")
    s.select("enhanceProvider", "xai")
    s.set_value("enhanceKey", "xai-keep-me")
    s.set_value("enhancePrompt", "changed prompt", "change")
    s.set_value("chapterName", "QA Util")
    p.click("#resetSettings")
    p.wait_for_timeout(500)
    c.eq(len(s.strip_status()), 3, "Reset defaults keeps the session's pages")
    c.ok(s.visible("#resultSection"), "…and stays on the workspace")
    defaults = {"watermark": "", "wmStyle": "clean", "wmPlace": "br", "wmSize": "m",
                "wmOpacity": "50", "pageFinish": "clean", "rotateTurn": "cw",
                "engine": "gemini", "enhanceProvider": "gemini", "fontSelect": "",
                "textCase": "upper", "fontVariety": "pro", "endTheme": "royal",
                "endScan": "", "chapterName": "", "tileMode": "1", "gpuCap": "100",
                "rawStyle": "photo", "rawStrength": "1", "targetLang": "English"}
    toggles = {"compressOut": False, "hdUpscale": False, "removeWatermark": True,
               "wmTileToo": False, "endUseColor": False, "maxQuality": False}
    got = {k: s.value(k) for k in defaults}
    got.update({k: s.checked(k) for k in toggles})
    want = dict(defaults); want.update(toggles)
    c.eq({k: v for k, v in got.items() if v != want[k]}, {}, "every control is back to its default")
    c.eq(p.evaluate("document.querySelector('.wf-card.active').dataset.wf"), "scan-translate",
         "workflow back to the default card")
    c.ok(s.value("enhancePrompt").startswith("Restore this"), "scan prompt back to the default")
    c.ok(not s.visible("#wmPreview"), "the watermark preview follows the cleared mark")
    c.eq(s.value("apiKey"), "", "gemini (the default engine) has no key saved")
    s.select("engine", "claude")
    c.eq(s.value("apiKey"), "sk-keep-me", "the Claude key survived the reset")
    s.select("enhanceProvider", "xai")
    c.eq(s.value("enhanceKey"), "xai-keep-me", "the Grok key survived the reset")
    s.select("engine", "gemini")
    s.select("enhanceProvider", "gemini")
    # persisted: a reload shows the same defaults
    s.reload()
    got2 = {k: s.value(k) for k in defaults}
    got2.update({k: s.checked(k) for k in toggles})
    c.eq({k: v for k, v in got2.items() if v != want[k]}, {}, "the defaults are what a reload restores")
    c.ok(s.value("enhancePrompt").startswith("Restore this"), "default prompt after reload")

    # nothing broken afterwards: a run works with the defaults
    fresh(s)
    s.set_workflow("scan-raw")
    s.upload([P5])
    s.wait_preview()
    s.go()
    s.wait_all_done(1, timeout=300000)
    out = decode(fetch(f"/api/result/{s.active_task()}"))
    c.eq(out.shape[:2], cv2.imread(P5).shape[:2], "a run after Reset defaults works (Scan -> Raw)")

    s.screenshot("utilities_end.png")

c.finish(s)
shutil.rmtree(TMP, ignore_errors=True)
