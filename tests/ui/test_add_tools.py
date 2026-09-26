"""Browser test of the editor's ADD family of tools, against a real server.

Covers: ＋ Add, Vertical translate, Type text, Lasso translate, Point
translate, Find missed text, the Details-list editing of added items, the
transcript buttons, Undo after a failed read, and tool chains (Type → Type →
Edit, Type → Resize → font, Point (fails) → Type in the same spot, Add → Undo
→ Type, page switch and back).

Runs WITHOUT an API key on purpose: every OCR/translate call must fail
loudly and leave the editor usable. Type text has to work end to end (the
typed English lands in the output image, in the chosen font, at the drawn
box) because it needs no key at all.

Run:  /path/to/python tests/ui/test_add_tools.py   (skips if no server)
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from add_harness import (Editor, Results, PAGE_IMG, RERENDER_TIMEOUT, changed_fraction,
                     fetch_result, ink_fraction, skip_unless_server)

FONT_A = FONT_B = None
PAGE2 = os.environ.get("MT_PAGE2", os.path.join(os.path.dirname(PAGE_IMG), "4.jpg"))


def near(a, b, tol=3):
    return all(abs(int(x) - int(y)) <= tol for x, y in zip(a, b))


def poly_near(a, b, tol=3):
    return len(a) == len(b) and all(near(p, q, tol) for p, q in zip(a, b))


def wait_response(ed, part, n_before, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        rs = [r for r in ed.responses if part in r[0]]
        if len(rs) > n_before:
            return rs[-1]
        # Not time.sleep: the sync Playwright client only delivers events
        # while it is inside one of its own calls.
        ed.page.wait_for_timeout(100)
    return None


def count_responses(ed, part):
    return len([r for r in ed.responses if part in r[0]])


def settle_ime(ed, timeout=5000):
    """True when the keyboard dialog comes up (within timeout), else False."""
    try:
        ed.ime_wait(timeout)
        return True
    except Exception:
        return False


def ink_angle(before, after, x, y, w, h):
    """Orientation (deg) of the pixels that changed inside a rect — for a
    tilted strip of text this is the strip's angle."""
    import cv2
    ra = before[y:y + h, x:x + w].astype(np.int16)
    rb = after[y:y + h, x:x + w].astype(np.int16)
    diff = (np.abs(ra - rb).max(axis=2) > 60) & (rb.mean(axis=2) < 110)
    ys, xs = np.nonzero(diff)
    if len(xs) < 50:
        return None, int(len(xs))
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    (_, _), (rw, rh), ang = cv2.minAreaRect(pts)
    if rw < rh:
        ang += 90.0
    while ang > 90:
        ang -= 180
    while ang <= -90:
        ang += 180
    return float(ang), int(len(xs))


def main():
    skip_unless_server()
    from playwright.sync_api import sync_playwright
    R = Results()
    with sync_playwright() as pw:
        ed = Editor(pw)
        try:
            run(ed, R)
        finally:
            errs = ed.errors()
            R.check("no JS exceptions during the whole run", not errs["js"], "; ".join(errs["js"])[:600])
            R.check("no console errors during the whole run", not errs["console"],
                    "; ".join(errs["console"])[:600])
            ed.close()
    sys.exit(R.summary())


def run(ed, R):
    p = ed.page
    ed.open()
    # Two pages so a page switch can be tested; both go through Clean (no key).
    p.click('.wf-card[data-wf="clean"]')
    p.set_input_files("#fileInput", [PAGE_IMG, PAGE2] if os.path.exists(PAGE2) else [PAGE_IMG])
    p.click("#goBtn")
    p.wait_for_selector("#pageResult", state="visible", timeout=400_000)
    p.wait_for_function(
        "() => { const i = document.getElementById('transFull'); return i.complete && i.naturalWidth > 0; }",
        timeout=400_000)
    src = p.get_attribute("#transFull", "src")
    tid = src.split("/api/result/")[1].split("?")[0]
    ed.task_id = tid
    # Let the second page finish too: a CPU box cleaning page 2 answers the
    # editor's calls seconds late, and that is not what is under test here.
    if p.locator(".pg-chip").count() >= 2:
        p.wait_for_selector(".pg-chip:nth-child(2) .pg-dot.done", timeout=400_000)
    W, H = ed.dims()
    R.check("page cleaned and shown", W > 0 and H > 0, f"{W}x{H}")
    base0 = fetch_result(tid, ed.rev())
    # Per-item fonts must differ from the page font, or a font change is invisible.
    page_font = p.input_value('#fontSelect')
    global FONT_A, FONT_B
    FONT_A, FONT_B = [f for f in ('Anton-Regular.ttf', 'PirataOne-Regular.ttf', 'Orbitron.ttf') if f != page_font][:2]
    FAR = (620, 1300, 300, 150)        # bottom-right face: must never change

    # ── 1. ＋ Add with the Claude engine and no key ─────────────────────
    ed.select_tool("add")
    n = count_responses(ed, "/api/ocr-translate/")
    ed.drag_box(750, 1050, 900, 1240)
    resp = wait_response(ed, "/api/ocr-translate/", n)
    body = ed.last_request("/api/ocr-translate/")
    R.check("Add: request carries the drawn bbox", body and near(body["bbox"], [750, 1050, 150, 190]),
            f"{body and body.get('bbox')}")
    R.check("Add: server refuses without a key", resp and resp[1] == 400, f"{resp and resp[:2]}")
    ime = settle_ime(ed)
    msg = (ed.ime_msg() if ime else "") + " | " + ed.hint()
    R.check("Add: the user is told WHY it failed (key)", "key" in msg.lower(), msg)
    if ime:
        ed.ime_cancel()
    oc = ed.overlay_counts()
    R.check("Add: no ghost box after the failure", oc["add"] == 0 and oc["cover"] == 0 and oc["polys"] == 0, str(oc))
    R.check("Add: tool still selected and usable", ed.tool() == "add")

    # ── 2. Type text A and B, then Edit A ───────────────────────────────
    ed.select_tool("type-add")
    ed.drag_box(70, 700, 190, 820)
    ed.ime_wait()
    R.check("Type text: IME opens for the typed original", "Type the original" in ed.ime_title(), ed.ime_title())
    ed.ime_use("テスト", "HELLO")
    R.check("Type text A: placed", "Added" in ed.hint(), ed.hint())
    ed.drag_box(465, 660, 595, 780)
    ed.ime_wait()
    ed.ime_use("二", "WORLD")
    R.check("Type text B: placed", "Added" in ed.hint(), ed.hint())

    ed.select_tool("edit")
    oc = ed.overlay_counts()
    R.check("Edit tool shows both typed boxes", oc["move"] == 2, str(oc))
    cx, cy = ed.to_client(130, 760)
    ed.ensure_visible(130, 760)
    cx, cy = ed.to_client(130, 760)
    p.mouse.click(cx, cy)
    p.wait_for_selector(".edit-pop", timeout=5000)
    R.check("Edit A: popover holds A's text", p.input_value(".edit-pop-text") == "HELLO", p.input_value(".edit-pop-text"))
    p.fill(".edit-pop-text", "HELLO A")
    n = count_responses(ed, "/api/rerender/")
    p.click(".epop-save")
    resp = wait_response(ed, "/api/rerender/", n, timeout=RERENDER_TIMEOUT / 1000)
    p.wait_for_function("() => !document.getElementById('editApply').disabled", timeout=RERENDER_TIMEOUT)
    body = ed.last_request("/api/rerender/")
    added = {a["id"]: a for a in (body or {}).get("added", [])}
    R.check("Chain Type→Type→Edit: both items sent, right text",
            set(added) == {"m1", "m2"} and added["m1"]["translation"] == "HELLO A"
            and added["m2"]["translation"] == "WORLD", str(added))
    R.check("Chain Type→Type→Edit: ids unique", len(added) == len(body.get("added", [])))
    R.check("re-render succeeded", resp and resp[1] == 200, str(resp))
    p.wait_for_function(
        "() => { const i = document.getElementById('transFull'); return i.complete && i.naturalWidth > 0; }",
        timeout=RERENDER_TIMEOUT)
    out1 = fetch_result(tid, ed.rev())
    ia = ink_fraction(out1, 70, 700, 120, 120)
    ib = ink_fraction(out1, 465, 660, 130, 120)
    R.check("Type text: typed English is drawn in box A", ia > 0.02, f"ink={ia:.3f}")
    R.check("Type text: typed English is drawn in box B", ib > 0.02, f"ink={ib:.3f}")
    R.check("Type text: far-away art untouched", changed_fraction(base0, out1, *FAR) < 0.01,
            f"changed={changed_fraction(base0, out1, *FAR):.4f}")

    # ── 3. Details list: textarea + per-item font, then Apply ───────────
    rows = ed.details_rows()
    arows = [r for r in rows if r["added"]]
    R.check("Details: both typed items listed", [r["id"] for r in arows] == ["m1", "m2"], str(rows))
    R.check("Details: A shows the edited text", any(r["id"] == "m1" and r["text"] == "HELLO A" for r in arows))
    p.select_option('.tl-font[data-id="m1"]', FONT_A)
    p.fill('.tl-edit[data-id="m2"]', "WORLD 2")
    n = count_responses(ed, "/api/rerender/")
    p.click("#applyBtn")
    resp = wait_response(ed, "/api/rerender/", n, timeout=RERENDER_TIMEOUT / 1000)
    p.wait_for_function("() => !document.getElementById('applyBtn').disabled", timeout=RERENDER_TIMEOUT)
    body = ed.last_request("/api/rerender/")
    added = {a["id"]: a for a in body.get("added", [])}
    R.check("Details: font pick for an added item is sent", body.get("fonts", {}).get("m1") == FONT_A,
            str(body.get("fonts")))
    R.check("Details: textarea edit of an added item is sent", added.get("m2", {}).get("translation") == "WORLD 2",
            str(added.get("m2")))
    p.wait_for_function(
        "() => { const i = document.getElementById('transFull'); return i.complete && i.naturalWidth > 0; }",
        timeout=RERENDER_TIMEOUT)
    out2 = fetch_result(tid, ed.rev())
    R.check("Details: font change actually changes the drawn text",
            changed_fraction(out1, out2, 70, 700, 120, 120) > 0.005,
            f"changed={changed_fraction(out1, out2, 70, 700, 120, 120):.4f}")

    # ── 4. Chain: Type text C → Resize → Edit font ───────────────────────
    ed.select_tool("type-add")
    ed.drag_box(480, 1060, 630, 1210)
    ed.ime_wait()
    ed.ime_use("三", "THREE")
    ed.select_tool("resize")
    oc = ed.overlay_counts()
    R.check("Resize tool shows the 3 typed boxes", oc["resize"] == 3, str(oc))
    ed.ensure_visible(555, 1135)
    ax, ay = ed.to_client(555, 1135)
    bx, by = ed.to_client(575, 1145)
    p.mouse.move(ax, ay); p.mouse.down(); p.mouse.move(bx, by, steps=6); p.mouse.up()
    ed.select_tool("edit")
    ed.ensure_visible(575, 1145)
    ex, ey = ed.to_client(575, 1145)
    p.mouse.click(ex, ey)
    p.wait_for_selector(".edit-pop", timeout=5000)
    R.check("Edit after Resize: popover opens on the MOVED box", p.input_value(".edit-pop-text") == "THREE",
            p.input_value(".edit-pop-text"))
    p.select_option(".epop-font", FONT_B)
    n = count_responses(ed, "/api/rerender/")
    p.click(".epop-save")
    resp = wait_response(ed, "/api/rerender/", n, timeout=RERENDER_TIMEOUT / 1000)
    p.wait_for_function("() => !document.getElementById('editApply').disabled", timeout=RERENDER_TIMEOUT)
    body = ed.last_request("/api/rerender/")
    bx3 = body.get("boxes", {}).get("m3")
    R.check("Chain Type→Resize→font: resized box kept", bx3 and near(bx3, [500, 1070, 150, 150], 6), str(bx3))
    R.check("Chain Type→Resize→font: font kept", body.get("fonts", {}).get("m3") == FONT_B, str(body.get("fonts")))
    R.check("Chain Type→Resize→font: earlier font (A) not lost", body.get("fonts", {}).get("m1") == FONT_A)
    added = {a["id"]: a for a in body.get("added", [])}
    R.check("Chain Type→Resize→font: translation kept", added.get("m3", {}).get("translation") == "THREE")
    p.wait_for_function(
        "() => { const i = document.getElementById('transFull'); return i.complete && i.naturalWidth > 0; }",
        timeout=RERENDER_TIMEOUT)

    # ── 5. Chain: Type text D → Add (fails) → Undo → Type text E ────────
    ed.select_tool("type-add")
    ed.drag_box(200, 1080, 330, 1180)
    ed.ime_wait()
    ed.ime_use("四", "FOUR")
    rows = [r["id"] for r in ed.details_rows() if r["added"]]
    R.check("Type text D placed (4 items)", rows == ["m1", "m2", "m3", "m4"], str(rows))
    ed.select_tool("add")
    n = count_responses(ed, "/api/ocr-translate/")
    ed.drag_box(200, 1200, 330, 1280)
    wait_response(ed, "/api/ocr-translate/", n)
    if settle_ime(ed):
        ed.ime_cancel()
    p.click("#undoBtn")
    rows = [r["id"] for r in ed.details_rows() if r["added"]]
    R.check("Chain Add(fails)→Undo: ONE undo removes D (the failed Add left no undo entry)",
            rows == ["m1", "m2", "m3"], str(rows))
    ed.select_tool("type-add")
    ed.drag_box(200, 1080, 330, 1180)
    ed.ime_wait()
    ed.ime_use("五", "FIVE")
    rows = ed.details_rows()
    ids = [r["id"] for r in rows if r["added"]]
    R.check("Chain Add→Undo→Type: E placed with a unique id", len(ids) == 4 and len(set(ids)) == 4
            and any(r["text"] == "FIVE" for r in rows), str(ids))

    # ── 6. Point translate (no key): polygon in image coords, scrolled ───
    strip = [(60, 1330), (330, 1270), (345, 1320), (75, 1380)]
    ed.select_tool("pen-add")
    n = count_responses(ed, "/api/ocr-translate/")
    ed.click_points(strip)
    resp = wait_response(ed, "/api/ocr-translate/", n)
    body = ed.last_request("/api/ocr-translate/")
    R.check("Point translate: poly matches the clicked points (page scrolled)",
            body and body.get("poly") and poly_near(body["poly"], strip), f"{body and body.get('poly')}")
    R.check("Point translate: bbox is the poly's bounds",
            body and near(body["bbox"], [60, 1270, 285, 110]), f"{body and body.get('bbox')}")
    ime = settle_ime(ed)
    msg = (ed.ime_msg() if ime else "") + " | " + ed.hint()
    R.check("Point translate: told why it failed", "key" in msg.lower(), msg)
    if ime:
        ed.ime_cancel()
    oc = ed.overlay_counts()
    R.check("Point translate: no ghost polygon after the failure", oc["polys"] == 0 and oc["pen"] == 0, str(oc))
    # then Type text in the same spot
    ed.select_tool("type-add")
    ed.drag_box(60, 1270, 345, 1380)
    ed.ime_wait()
    ed.ime_use("六", "SIX")
    rows = ed.details_rows()
    ids = [r["id"] for r in rows if r["added"]]
    R.check("Chain Point(fails)→Type: exactly one new item, no half-made one",
            len(ids) == 5 and len(set(ids)) == 5 and sum(1 for r in rows if r["text"] == "SIX") == 1, str(ids))
    ed.select_tool("edit")
    oc = ed.overlay_counts()
    R.check("Chain Point(fails)→Type: overlay shows 5 boxes and no leftover shape",
            oc["move"] == 5 and oc["polys"] == 0, str(oc))

    # ── 7. Lasso translate (no key) ──────────────────────────────────────
    lasso = [(420, 200), (520, 190), (560, 260), (500, 320), (430, 300)]
    ed.select_tool("lasso-add")
    n = count_responses(ed, "/api/ocr-translate/")
    ed.lasso(lasso)
    resp = wait_response(ed, "/api/ocr-translate/", n)
    body = ed.last_request("/api/ocr-translate/")
    xs = [q[0] for q in lasso]; ys = [q[1] for q in lasso]
    R.check("Lasso translate: bbox is the outline's bounds",
            body and near(body["bbox"], [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)], 4),
            f"{body and body.get('bbox')}")
    pl = (body or {}).get("poly") or []
    R.check("Lasso translate: outline sent as poly so OCR reads only inside it",
            len(pl) >= 3 and all(min(xs) - 4 <= q[0] <= max(xs) + 4 and min(ys) - 4 <= q[1] <= max(ys) + 4 for q in pl),
            f"{len(pl)} pts")
    if settle_ime(ed):
        ed.ime_cancel()
    oc = ed.overlay_counts()
    R.check("Lasso translate: no ghost outline after the failure", oc["polys"] == 0, str(oc))

    # ── 8. Offline engine: server has no OCR — must say so ───────────────
    ed.set_engine("local")
    ed.select_tool("vert-add")
    n = count_responses(ed, "/api/ocr-translate/")
    ed.drag_box(880, 30, 960, 230)
    resp = wait_response(ed, "/api/ocr-translate/", n)
    ime = settle_ime(ed)
    msg = (ed.ime_msg() if ime else "") + " | " + (ed.ime_title() if ime else "") + " | " + ed.hint()
    R.check("Vertical (offline): IME opens with the reason (OCR unavailable)",
            ime and "ocr" in msg.lower(), msg)
    if ime:
        ed.ime_use("縦", "VERT")
    n = count_responses(ed, "/api/rerender/")
    ed.select_tool("edit")
    ed.apply()
    body = ed.last_request("/api/rerender/")
    added = {a["id"]: a for a in body.get("added", [])}
    vid = next((k for k, v in added.items() if v["translation"] == "VERT"), None)
    R.check("Vertical: placed sideways (rotation -90 sent)", vid and body.get("rotations", {}).get(vid) == -90,
            f"{vid} {body.get('rotations')}")

    # ── 9. Point translate (offline) end to end, scaled-down display ────
    p.set_viewport_size({"width": 640, "height": 800})
    p.wait_for_timeout(300)
    ed.select_tool("pen-add")
    n = count_responses(ed, "/api/ocr-translate/")
    # a strip across the (now white) big bubble, ~-33 deg: the only ink that
    # changes there is the text itself, so its angle can be measured.
    strip2 = [(750, 1180), (890, 1090), (905, 1115), (765, 1205)]
    ed.click_points(strip2)
    resp = wait_response(ed, "/api/ocr-translate/", n)
    body = ed.last_request("/api/ocr-translate/")
    R.check("Point translate: poly correct on a scaled-down display",
            body and body.get("poly") and poly_near(body["poly"], strip2, 4), f"{body and body.get('poly')}")
    ed.ime_wait()
    ed.ime_use("ドドド", "RUMBLE RUMBLE")
    p.set_viewport_size({"width": 1280, "height": 900})
    before = fetch_result(tid, ed.rev())
    ed.select_tool("edit")
    ed.apply()
    body = ed.last_request("/api/rerender/")
    added = {a["id"]: a for a in body.get("added", [])}
    pid = next((k for k, v in added.items() if v["translation"] == "RUMBLE RUMBLE"), None)
    R.check("Point translate: poly kept on the placed item", pid and poly_near(added[pid].get("poly") or [], strip2, 4),
            f"{pid} {added.get(pid, {}).get('poly')}")
    after = fetch_result(tid, ed.rev())
    ang, npx = ink_angle(before, after, 740, 1080, 175, 135)
    R.check("Point translate: text drawn inside the tilted strip", npx > 60, f"{npx} px")
    R.check("Point translate: tilted strip gives tilted text (~-33°)", ang is not None and -45 <= ang <= -20,
            f"angle={ang}")
    # Chain: Point translate → Edit its wording (tilt untouched) → the tilt
    # from the outline must survive; a saved "0°" would freeze it upright.
    ed.select_tool("edit")
    ed.ensure_visible(828, 1148)
    ex, ey = ed.to_client(828, 1148)
    p.mouse.click(ex, ey)
    p.wait_for_selector(".edit-pop", timeout=5000)
    R.check("Chain Point→Edit: popover opens on the point-translated item",
            p.input_value(".edit-pop-text") == "RUMBLE RUMBLE", p.input_value(".edit-pop-text"))
    p.fill(".edit-pop-text", "RUMBLE!")
    n = count_responses(ed, "/api/rerender/")
    p.click(".epop-save")
    wait_response(ed, "/api/rerender/", n, timeout=RERENDER_TIMEOUT / 1000)
    p.wait_for_function("() => !document.getElementById('editApply').disabled", timeout=RERENDER_TIMEOUT)
    body = ed.last_request("/api/rerender/")
    R.check("Chain Point→Edit: no tilt override written for an untouched slider",
            pid not in (body.get("rotations") or {}), str(body.get("rotations")))
    R.check("Chain Point→Edit: the click opened the editor, not deleted the erase outline",
            any(isinstance(c, dict) and poly_near(c.get("poly") or [], strip2, 4) for c in body.get("covers", [])),
            str(body.get("covers"))[:200])
    p.wait_for_function(
        "() => { const i = document.getElementById('transFull'); return i.complete && i.naturalWidth > 0; }",
        timeout=RERENDER_TIMEOUT)
    after2 = fetch_result(tid, ed.rev())
    ang2, npx2 = ink_angle(before, after2, 740, 1080, 175, 135)
    R.check("Chain Point→Edit: text still tilted after the edit", ang2 is not None and -45 <= ang2 <= -20,
            f"angle={ang2} ({npx2} px)")

    # ── 10. Find missed text ─────────────────────────────────────────────
    ed.set_engine("claude")
    p.click("#rescanBtn")
    p.wait_for_timeout(300)
    R.check("Find missed (no key): asks for the key, button live",
            "key" in ed.hint().lower() and not p.is_disabled("#rescanBtn"), ed.hint())
    ed.set_engine("local")
    n = count_responses(ed, "/api/rescan/")
    p.click("#rescanBtn")
    resp = wait_response(ed, "/api/rescan/", n, timeout=300)
    p.wait_for_function("() => !document.getElementById('rescanBtn').disabled", timeout=300_000)
    h = ed.hint()
    R.check("Find missed (offline): says clearly that OCR is unavailable",
            ("ocr" in h.lower()) and "Smart Detection" not in h, h)
    R.check("Find missed (offline): button restored", p.text_content("#rescanBtn").strip().endswith("Find missed text"))

    # ── 11. Transcript ───────────────────────────────────────────────────
    ed.go_tab("details")
    p.click("#copyTranscript")
    p.wait_for_timeout(300)
    try:
        clip = p.evaluate("navigator.clipboard.readText()")
    except Exception as e:
        clip = f"<clipboard error {e}>"
    R.check("Transcript copy: typed items included", "HELLO A" in clip and "テスト" in clip, clip[:200])
    with p.expect_download(timeout=10_000) as dl:
        p.click("#downloadTranscript")
    path = dl.value.path()
    txt = open(path, encoding="utf-8").read() if path else ""
    R.check("Transcript download: .txt has the lines", "WORLD 2" in txt and dl.value.suggested_filename.endswith("_translation.txt"),
            f"{dl.value.suggested_filename} {txt[:120]!r}")

    # ── 12. Page switch and back keeps everything ────────────────────────
    chips = p.locator(".pg-chip")
    if chips.count() >= 2:
        p.wait_for_selector(".pg-chip:nth-child(2) .pg-dot.done", timeout=400_000)
        before_rows = ed.details_rows()
        chips.nth(1).locator(".pg-thumb").click()
        p.wait_for_function(
            f"() => !document.getElementById('transFull').src.includes('{tid}')")
        R.check("Page 2 (clean, nothing added) has no Details tab and no rows",
                not ed.details_tab_visible() and not ed.details_rows())
        ed.select_tool("edit")
        R.check("Page 2 shows none of page 1's boxes", ed.overlay_counts()["move"] == 0, str(ed.overlay_counts()))
        chips.nth(0).locator(".pg-thumb").click()
        p.wait_for_function(
            f"() => document.getElementById('transFull').src.includes('{tid}')")
        p.wait_for_function(
            "() => { const i = document.getElementById('transFull'); return i.complete && i.naturalWidth > 0; }")
        after_rows = ed.details_rows()
        R.check("Back on page 1: added rows, text and fonts intact", after_rows == before_rows,
                f"{before_rows} != {after_rows}")
        ed.select_tool("edit")
        ed.apply()
        body = ed.last_request("/api/rerender/")
        added = {a["id"]: a for a in body.get("added", [])}
        R.check("After switch: poly, bbox, translation, font still sent",
                pid in added and poly_near(added[pid].get("poly") or [], strip2, 4)
                and near(added["m1"]["bbox"], [70, 700, 120, 120]) and added["m1"]["translation"] == "HELLO A"
                and body.get("fonts", {}).get("m1") == FONT_A
                and body.get("fonts", {}).get("m3") == FONT_B,
                f"added={list(added)} fonts={body.get('fonts')}")
    else:
        R.check("second page available for the switch test", False, "only one page")


if __name__ == "__main__":
    main()
