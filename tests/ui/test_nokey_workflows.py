"""The no-key workflows end to end: Watermark only (every style), Rotate only,
Cut out pages, Clean — no key; plus the per-page downloads and the transcript."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import (Check, ROOT, Session, build_inputs, decode, fetch,
                      page_paths, skip_unless_server, task_of)

skip_unless_server()
inputs = build_inputs()
c = Check("no-key workflows")
P3, P4 = page_paths("3.jpg", "4.jpg")


def diff_quadrants(a, b):
    """Fraction of changed pixels in (tl, tr, bl, br) and overall."""
    m = np.any(a != b, axis=2)
    h, w = m.shape
    q = [m[:h // 2, :w // 2], m[:h // 2, w // 2:], m[h // 2:, :w // 2], m[h // 2:, w // 2:]]
    return [float(x.mean()) for x in q], float(m.mean())


def run_single(s, wf, path, setup=None):
    s.page.click("#newBtn") if s.visible("#newBtn") else None
    if not s.visible("#dropZone"):
        s.page.click("#clearBtn")
    s.set_workflow(wf)
    if setup:
        setup()
    s.upload([path])
    s.wait_preview()
    s.go()
    s.wait_all_done(1, timeout=300000)
    tid = s.active_task()
    return tid, decode(fetch(f"/api/original/{tid}")), decode(fetch(f"/api/result/{tid}"))


with Session() as s:
    p = s.page

    def wm(text="@qa", style="clean", place="br", size="m", opacity="50", tile_too=False, credit=""):
        def _():
            s.set_value("watermark", text)
            s.select("wmStyle", style)
            s.select("wmPlace", place)
            s.select("wmSize", size)
            s.select("wmOpacity", opacity)
            s.set_toggle("wmTileToo", tile_too)
            s.set_value("credit", credit)
        return _

    # ── Watermark only: clean, bottom-right ──
    tid, orig, res = run_single(s, "watermark-only", P3, wm())
    clean = decode(fetch(f"/api/result/{tid}?watermark=0"))
    c.ok(np.array_equal(clean, orig), "clean twin (?watermark=0) is the untouched page")
    q, tot = diff_quadrants(orig, res)
    c.ok(0 < tot < 0.05 and q[3] > 0 and q[0] == q[1] == q[2] == 0,
         f"clean/br mark stays in the bottom-right quarter of a busy page {q}")
    c.eq(s.text("#compLabelRight"), "Watermarked", "compare label for watermark-only")
    c.eq(s.text('.tab[data-tab="translated"]'), "Stamped", "result tab label")
    c.ok(not s.visible('.tab[data-tab="details"]'), "no Details tab on a no-text page")

    # Downloads: watermarked / clean / chapter name / compressed → .jpg
    name, data = s.download("#downloadBtn")
    c.eq(name, "translated_3.png", "download name")
    c.ok(data[:8] == b"\x89PNG\r\n\x1a\n" and np.array_equal(decode(data), res), "download is the stamped PNG")
    name, data = s.download("#downloadCleanBtn")
    c.eq(name, "translated_3 (no watermark).png", "clean download name")
    c.ok(np.array_equal(decode(data), orig), "clean download equals the untouched page")
    s.set_value("chapterName", "QA Ch 1")
    name, _ = s.download("#downloadBtn")
    c.eq(name, "QA Ch 1 - 3.png", "chapter-named download")
    s.set_value("chapterName", "")

    # ── tile: the whole page ──
    _, orig, res = run_single(s, "watermark-only", P3, wm(style="tile"))
    q, tot = diff_quadrants(orig, res)
    c.ok(all(x > 0.002 for x in q), f"tiled mark covers every quadrant {q}")

    # ── clean + tiled as well: corner mark AND tile ──
    _, orig, res = run_single(s, "watermark-only", P3, wm(style="clean", place="tl", tile_too=True))
    q2, tot2 = diff_quadrants(orig, res)
    c.ok(all(x > 0.002 for x in q2) and q2[0] > q2[3] * 1.3,
         f"+tile adds the tile under a top-left corner mark {q2}")

    # ── size / opacity / bold ──
    _, orig, res_s = run_single(s, "watermark-only", P3, wm(style="bold", size="s", opacity="100"))
    _, _, res_l = run_single(s, "watermark-only", P3, wm(style="bold", size="l", opacity="100"))
    n_s = int(np.any(orig != res_s, axis=2).sum())
    n_l = int(np.any(orig != res_l, axis=2).sum())
    c.ok(n_l > n_s * 1.8, f"large mark changes more pixels than small ({n_l} vs {n_s})")
    _, _, res_lo = run_single(s, "watermark-only", P3, wm(style="bold", size="l", opacity="20"))
    d_hi = np.abs(orig.astype(int) - res_l.astype(int)).sum()
    d_lo = np.abs(orig.astype(int) - res_lo.astype(int)).sum()
    c.ok(d_hi > d_lo * 2, f"opacity 100 marks harder than 20 ({d_hi} vs {d_lo})")

    # ── pill / ribbon / ghost render without error ──
    for st in ("pill", "ribbon", "ghost"):
        _, orig, res = run_single(s, "watermark-only", P3, wm(style=st))
        q, tot = diff_quadrants(orig, res)
        c.ok(tot > 0, f"style {st} stamps something ({tot:.4f} of pixels)")

    # ── credit line only: small, opposite corner (bottom-left) ──
    _, orig, res = run_single(s, "watermark-only", P3, wm(text="", credit="TL: QA"))
    q, tot = diff_quadrants(orig, res)
    c.ok(q[2] > 0 and q[0] == q[1] == q[3] == 0, f"credit alone lands bottom-left {q}")

    # ── Compress Output → the download is a real JPEG named .jpg ──
    s.set_toggle("compressOut", True)
    tid, orig, res = run_single(s, "watermark-only", P3, wm())
    name, data = s.download("#downloadBtn")
    c.eq(name, "translated_3.jpg", "compressed download is named .jpg")
    c.ok(data[:3] == b"\xff\xd8\xff", "compressed download has JPEG bytes")
    s.set_toggle("compressOut", False)

    # ── transcript on a page with no text ──
    p.click('.tab[data-tab="compare"]')
    p.evaluate("document.getElementById('downloadTranscript').click()")
    c.ok(s.visible("#errorSection") and "Nothing to download" in s.text("#errorMsg"),
         "transcript download on a no-text page explains itself")
    p.click("#retryBtn")
    c.ok(s.visible("#resultSection"), "Try Again goes back to the result")

    # ── Rotate only: cw on two pages ──
    p.click("#newBtn")
    s.set_workflow("rotate-pages")
    c.ok(s.visible("#rotatePanel"), "turn picker shows for Rotate only")
    s.select("rotateTurn", "cw")
    s.upload([P3, P4])
    s.wait_preview()
    s.go()
    s.wait_all_done(2, timeout=120000)
    import cv2
    for i, src in enumerate([P3, P4]):
        t = task_of(s.strip_status()[i]["src"])
        o = cv2.imread(src)
        r = decode(fetch(f"/api/result/{t}"))
        c.ok(np.array_equal(r, cv2.rotate(o, cv2.ROTATE_90_CLOCKWISE)), f"page {i + 1} turned 90° right exactly")
    c.eq(s.text("#compLabelRight"), "Turned", "compare label for Rotate only")
    for turn, op in (("ccw", cv2.ROTATE_90_COUNTERCLOCKWISE), ("180", cv2.ROTATE_180)):
        p.click("#newBtn")
        s.set_workflow("rotate-pages")
        s.select("rotateTurn", turn)
        s.upload([P3])
        s.wait_preview()
        s.go()
        s.wait_all_done(1, timeout=60000)
        r = decode(fetch(f"/api/result/{s.active_task()}"))
        c.ok(np.array_equal(r, cv2.rotate(cv2.imread(P3), op)), f"turn {turn} exact")
    # A typed watermark is NOT stamped by the fix-up workflow.
    c.ok(np.array_equal(r, cv2.rotate(cv2.imread(P3), cv2.ROTATE_180)), "rotate output carries no watermark")

    # ── Cut out pages (batch): a photo is cut, a full-frame page passes through ──
    p.click("#newBtn")
    s.set_workflow("cut-pages")
    s.upload([os.path.join(inputs, "photo4.jpg"), P3])
    s.wait_preview()
    s.go()
    s.wait_all_done(2, timeout=300000)
    st = s.strip_status()          # name-sorted: 3.jpg first, then photo4.jpg
    photo = cv2.imread(os.path.join(inputs, "photo4.jpg"))
    cut = decode(fetch(f"/api/result/{task_of(st[1]['src'])}"))
    c.ok(cut.shape[0] < photo.shape[0] * 0.9 and cut.shape[1] < photo.shape[1] * 0.9,
         f"photo {photo.shape[:2]} cut down to the page {cut.shape[:2]}")
    c.ok(cut.shape[0] > photo.shape[0] * 0.6, "the cut keeps most of the page")
    through = decode(fetch(f"/api/result/{task_of(st[0]['src'])}"))
    c.ok(np.array_equal(through, cv2.imread(P3)),
         f"a scan that fills the frame passes through untouched ({through.shape[:2]})")
    s.click_chip(0)
    c.ok("passed through untouched" in s.text("#progressMsg") or s.visible("#pageResult"),
         "untouched page is still a finished page")
    c.eq(s.text("#compLabelRight"), "Page only", "compare label for Cut out pages")
    log = open(os.path.join(ROOT, "app.log"), errors="replace").read() if os.path.exists(os.path.join(ROOT, "app.log")) else ""
    c.ok("Traceback" not in log.split("[cutpage]")[-1][:2000] if "[cutpage]" in log else True,
         "cut-out does not trace back in the server log")

    # ── Clean — no key ──
    p.click("#newBtn")
    s.set_workflow("local-clean")
    s.set_toggle("hdUpscale", False)
    s.set_value("watermark", "@qa")
    s.upload([P3])
    s.wait_preview()
    s.go()
    s.wait_all_done(1, timeout=300000)
    tid = s.active_task()
    res = decode(fetch(f"/api/result/{tid}"))
    o = cv2.imread(P3)
    c.eq(res.shape[:2], o.shape[:2], "Clean — no key keeps the page size")
    c.ok(np.array_equal(decode(fetch(f"/api/result/{tid}?watermark=0")), res) is False,
         "Clean — no key stamps the typed watermark and keeps a clean twin")
    twin = decode(fetch(f"/api/result/{tid}?watermark=0"))
    c.ok(np.percentile(twin, 90) == 255 and np.percentile(twin, 10) <= np.percentile(o, 10),
         "cleaned page keeps white paper and puts the black back into the ink")
    c.eq(s.text("#compLabelRight"), "Cleaned", "compare label for Clean — no key")
    c.eq(s.text('.tab[data-tab="translated"]'), "Clean", "tab label for Clean — no key")

c.finish(s)
