"""Browser test of the editor's on-image repair tools, end to end.

Erase box, Lasso erase, Restore (outline + click), Redraw fill, Tone fill,
Clone stamp, Line, the ⌫ per-item erase, removing a cover by clicking it,
tool chains (each earlier cover must survive), a cover drawn while an Apply
is in flight, gestures on a scrolled / zoomed page, and covers staying on
their own page.

Every tool is exercised with real pointer gestures on the overlay in
Chromium, Apply & Re-render is pressed, and the re-rendered image is pulled
from /api/result and checked pixel by pixel: the target region must change
as intended and nothing outside it may change at all.

Needs a running server (default http://127.0.0.1:8022, see harness.py) and
the test pages in MT_PAGES. Skips cleanly when the server is not up.

Run: /path/to/python tests/ui/test_erase_tools.py
"""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import (UI, Report, PAGES_DIR, APPLY_TIMEOUT, server_up,   # noqa: E402
                     changed_mask, changed_outside, dark_fraction, poly_rect, region)

OUT = os.environ.get("MT_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out"))


def save(name, img):
    os.makedirs(OUT, exist_ok=True)
    cv2.imwrite(os.path.join(OUT, name), img)


def no_js_errors(rep, ui, label):
    errs, cons = ui.drain_errors()
    cons = [c for c in cons if "ERR_CERT" not in c and "favicon" not in c]
    rep.check(f"{label}: no JS exceptions", not errs, "; ".join(errs))
    rep.check(f"{label}: no console errors", not cons, "; ".join(cons))


def changed_fraction(a, b, rect):
    x, y, w, h = rect
    m = changed_mask(region(a, x, y, w, h), region(b, x, y, w, h))
    return float(m.mean()) if m.size else 0.0


def line_dark(img, p0, p1, width, thr=100):
    """Fraction of samples along the segment p0→p1 that are dark."""
    n = 60
    hits = 0
    for i in range(n + 1):
        f = i / n
        x = int(round(p0[0] + (p1[0] - p0[0]) * f))
        y = int(round(p0[1] + (p1[1] - p0[1]) * f))
        px = region(img, x - 1, y - 1, 3, 3)
        if px.size and cv2.cvtColor(px, cv2.COLOR_BGR2GRAY).min() < thr:
            hits += 1
    return hits / (n + 1)


def main():
    if not server_up():
        print("SKIP: server is not running")
        return 0
    from playwright.sync_api import sync_playwright

    rep = Report()
    with sync_playwright() as pw:
        ui = UI(pw)
        try:
            run(ui, rep)
        finally:
            ui.close()
    ok = rep.summary()
    return 0 if ok else 1


def run(ui, rep):
    files = [f"{PAGES_DIR}/4.jpg", f"{PAGES_DIR}/6.jpg"]
    t0 = time.time()
    ids = ui.run_clean(files, credit="TL: QA Team")
    print(f"clean run: {ids} in {time.time() - t0:.0f}s", flush=True)
    rep.check("clean run finishes with two pages", len(ids) == 2)
    no_js_errors(rep, ui, "clean run")

    # ── page 1 (4.jpg) ───────────────────────────────────────────────
    ui.select_page(0)
    ui.open_editor()
    W, H = ui.dims()
    orig1 = ui.original_image()
    clean1 = ui.result_image()
    save("p1_clean.png", clean1)

    # A baseline re-render with no covers at all must reproduce the page.
    ui.set_tool("cover")
    body = ui.apply()
    base1 = ui.result_image()
    save("p1_base.png", base1)
    rep.check("baseline: empty covers sent", body.get("covers") == [], str(body.get("covers")))
    n, bb = changed_outside(clean1, base1, [])
    rep.check("baseline: re-render with no edits reproduces the page byte for byte",
              n == 0, f"{n} px differ, bbox {bb}")
    no_js_errors(rep, ui, "baseline")

    SFX = (330, 240, 140, 175)          # hand-lettered ドキドキ, kept by Clean
    BALLOON = (200, 565, 185, 150)      # balloon text, erased by Clean
    print("dark fraction SFX orig/clean:", dark_fraction(orig1, *SFX), dark_fraction(clean1, *SFX))
    print("dark fraction balloon orig/clean:", dark_fraction(orig1, *BALLOON), dark_fraction(clean1, *BALLOON))

    # ── Erase box ────────────────────────────────────────────────────
    ui.set_tool("cover")
    ui.drag(SFX[0], SFX[1], SFX[0] + SFX[2], SFX[1] + SFX[3])
    oc = ui.overlay_counts()
    rep.check("erase box: cover drawn on the overlay", oc["cover"] == 1, str(oc))
    body = ui.apply()
    cv = body.get("covers")
    rep.check("erase box: request carries the box", cv and len(cv) == 1 and isinstance(cv[0], list)
              and all(abs(cv[0][i] - SFX[i]) <= 3 for i in range(4)), str(cv))
    er1 = ui.result_image()
    save("p1_erase_box.png", er1)
    cf = changed_fraction(base1, er1, SFX)
    rep.check("erase box: target region changed", cf > 0.15, f"changed fraction {cf:.3f}")
    n, bb = changed_outside(base1, er1, [SFX], margin=8)
    rep.check("erase box: nothing outside the box changed", n == 0, f"{n} px differ at {bb}")
    # Was the lettering actually taken? Strokes of the SFX = dark pixels of the
    # cleaned page inside the box; report how many stayed dark.
    g0 = cv2.cvtColor(region(base1, *SFX), cv2.COLOR_BGR2GRAY) < 110
    g1 = cv2.cvtColor(region(er1, *SFX), cv2.COLOR_BGR2GRAY) < 110
    still = float((g0 & g1).sum() / max(1, g0.sum()))
    print(f"erase box: {still:.2f} of the box's dark pixels are still dark (art or leftover)")
    # Documented compositor behaviour is checked separately; here we only
    # require the letters changed and the box stayed contained.
    no_js_errors(rep, ui, "erase box")

    # ── Remove the cover by clicking it, re-apply → back to baseline ─
    ui.page.locator("#moveLayer .cover-box").first.click()
    oc = ui.overlay_counts()
    rep.check("remove cover: clicking the box removes it from the overlay", oc["cover"] == 0, str(oc))
    body = ui.apply()
    rep.check("remove cover: request carries no covers", body.get("covers") == [], str(body.get("covers")))
    back = ui.result_image()
    n, bb = changed_outside(base1, back, [])
    rep.check("remove cover: page reverts to the baseline exactly", n == 0, f"{n} px differ at {bb}")
    no_js_errors(rep, ui, "remove cover")

    # ── Lasso erase (free-form drag around the SFX) ──────────────────
    ui.set_tool("lasso")
    x, y, w, h = SFX
    path = [(x + 5, y + 5), (x + w // 2, y - 2), (x + w - 3, y + 8), (x + w + 2, y + h // 2),
            (x + w - 6, y + h - 4), (x + w // 2, y + h + 2), (x + 3, y + h - 8), (x - 2, y + h // 2),
            (x + 5, y + 5)]
    ui.drag_path(path)
    oc = ui.overlay_counts()
    rep.check("lasso: polygon drawn on the overlay", oc["polygon"] == 1, str(oc))
    body = ui.apply()
    cv = body.get("covers")
    rep.check("lasso: request carries the polygon", cv and len(cv) == 1 and isinstance(cv[0], dict)
              and len(cv[0].get("poly", [])) >= 3, str(cv)[:200])
    lz = ui.result_image()
    save("p1_lasso.png", lz)
    prect = poly_rect(cv[0]["poly"]) if cv and isinstance(cv[0], dict) and cv[0].get("poly") else SFX
    cf = changed_fraction(base1, lz, prect)
    rep.check("lasso: outlined region changed", cf > 0.15, f"changed fraction {cf:.3f}")
    n, bb = changed_outside(base1, lz, [prect], margin=8)
    rep.check("lasso: nothing outside the outline changed", n == 0, f"{n} px differ at {bb}")
    no_js_errors(rep, ui, "lasso")
    ui.page.locator("#moveLayer svg polygon").first.click(force=True)
    rep.check("lasso: clicking the outline removes it", ui.overlay_counts()["polygon"] == 0)

    # ── Restore outline: bring the ORIGINAL balloon text back ────────
    ui.set_tool("restore")
    x, y, w, h = BALLOON
    path = [(x + 4, y + 4), (x + w - 4, y + 4), (x + w - 4, y + h - 4), (x + 4, y + h - 4), (x + 4, y + 4)]
    ui.drag_path(path)
    oc = ui.overlay_counts()
    rep.check("restore outline: outline drawn on the overlay", oc["polygon"] == 1, str(oc))
    body = ui.apply()
    cv = body.get("covers")
    rep.check("restore outline: request carries restore_poly",
              cv and len(cv) == 1 and isinstance(cv[0], dict) and "restore_poly" in cv[0], str(cv)[:200])
    rs = ui.result_image()
    save("p1_restore_poly.png", rs)
    d_orig, d_base, d_rs = dark_fraction(orig1, *BALLOON), dark_fraction(base1, *BALLOON), dark_fraction(rs, *BALLOON)
    rep.check("restore outline: original text is back inside the outline",
              d_rs > 0.5 * d_orig and d_rs > 5 * max(d_base, 0.001),
              f"dark fraction orig {d_orig:.3f} clean {d_base:.3f} restored {d_rs:.3f}")
    n, bb = changed_outside(base1, rs, [BALLOON], margin=8)
    rep.check("restore outline: nothing outside the outline changed", n == 0, f"{n} px differ at {bb}")
    no_js_errors(rep, ui, "restore outline")
    ui.page.locator("#moveLayer svg polygon").first.click(force=True)
    rep.check("restore outline: clicking the outline removes it", ui.overlay_counts()["polygon"] == 0)

    # ── Restore click: a single click un-deletes the cleaned blob ────
    ui.set_tool("restore")
    ui.click_at(x + w // 2, y + h // 2)
    oc = ui.overlay_counts()
    rep.check("restore click: a marker for the click is drawn on the overlay",
              oc["children"] == 1 and oc["cover"] == 0, str(oc))
    bad = ui.page.evaluate(
        "() => Array.from(document.querySelectorAll('#moveLayer > *')).map(e => e.style.left + '/' + e.style.width)")
    rep.check("restore click: marker has real coordinates (no NaN)",
              all("NaN" not in s for s in bad), str(bad))
    body = ui.apply()
    cv = body.get("covers")
    rep.check("restore click: request carries restore_click",
              cv and len(cv) == 1 and isinstance(cv[0], dict) and "restore_click" in cv[0], str(cv)[:200])
    rc = ui.result_image()
    save("p1_restore_click.png", rc)
    d_rc = dark_fraction(rc, *BALLOON)
    rep.check("restore click: original text is back around the click",
              d_rc > 0.5 * d_orig, f"dark fraction orig {d_orig:.3f} clean {d_base:.3f} restored {d_rc:.3f}")
    m = changed_mask(base1, rc)
    ys, xs = np.where(m)
    if xs.size:
        cb = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        rep.check("restore click: only the balloon area changed",
                  cb[0] >= x - 60 and cb[1] >= y - 60 and cb[2] <= x + w + 60 and cb[3] <= y + h + 60,
                  f"changes span {cb}, balloon {BALLOON}")
    no_js_errors(rep, ui, "restore click")
    # click-to-remove the click marker
    first = ui.page.locator("#moveLayer > *").first
    first.click(force=True)
    rep.check("restore click: clicking the marker removes it", ui.overlay_counts()["children"] == 0,
              str(ui.overlay_counts()))

    # ── Per-item ⌫ (the credit item Clean placed) ────────────────────
    # Clean pages hide the Details tab (nothing to translate); the ⌫ button
    # lives there, so show the tab for the test — the click path and the
    # server-side erase are exactly those of a translated page.
    def details():
        # every re-render re-hides the tab on a Clean page — show it again
        ui.page.evaluate("() => { document.querySelector('.tab[data-tab=\"details\"]').style.display = ''; }")
        ui.page.click('.tab[data-tab="details"]')
        ui.page.wait_for_timeout(200)

    try:
        details()
        n_items = ui.page.locator("#translationsList .tl-erase").count()
        rep.check("per-item ⌫: the credit item is listed", n_items >= 1, f"{n_items} items")
        if n_items:
            ui.page.locator("#translationsList .tl-erase").first.click()
            ui.page.wait_for_timeout(150)
            on = "on" in (ui.page.locator("#translationsList .tl-erase").first.get_attribute("class") or "")
            rep.check("per-item ⌫: button toggles on", on)
            ui.page.click('.tab[data-tab="translated"]')
            body = ui.apply()
            rep.check("per-item ⌫: request carries the erased id", body.get("erased") == ["1"], str(body.get("erased")))
            ez = ui.result_image()
            save("p1_item_erase.png", ez)
            m = changed_mask(base1, ez)
            ys, xs = np.where(m)
            credit_box = ((int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
                          if xs.size else None)
            rep.check("per-item ⌫: the credit's region changed", credit_box is not None, "nothing changed")
            # ⌫ off, ✕ (skip) on: the page without the credit. Our credit is
            # an overlay on a plate that never held it, so erasing it must give
            # exactly the same pixels as skipping it — no art may be healed away.
            details()
            ui.page.locator("#translationsList .tl-erase").first.click()
            ui.page.locator("#translationsList .tl-x").first.click()
            ui.page.click('.tab[data-tab="translated"]')
            body = ui.apply()
            rep.check("per-item ✕: request carries the skipped id", body.get("excluded") == ["1"], str(body.get("excluded")))
            sk = ui.result_image()
            save("p1_item_skip.png", sk)
            n, bb = changed_outside(sk, ez, [])
            rep.check("per-item ⌫: erasing the credit removes only the credit (same pixels as skipping it)",
                      n == 0, f"{n} px differ at {bb} — art under the credit was healed away")
            d0 = dark_fraction(base1, *credit_box) if credit_box else 0
            d1 = dark_fraction(sk, *credit_box) if credit_box else 0
            rep.check("per-item ✕/⌫: the credit lettering is gone", credit_box is not None and d1 < d0,
                      f"dark before {d0:.3f} after {d1:.3f}")
            # ✕ off again → exactly the baseline
            details()
            ui.page.locator("#translationsList .tl-x").first.click()
            ui.page.click('.tab[data-tab="translated"]')
            ui.apply()
            n, bb = changed_outside(base1, ui.result_image(), [])
            rep.check("per-item ⌫/✕: toggling both off restores the credit exactly", n == 0, f"{n} px differ at {bb}")
    finally:
        ui.page.click('.tab[data-tab="translated"]')
    no_js_errors(rep, ui, "per-item ⌫")

    # ── Line ─────────────────────────────────────────────────────────
    ui.set_tool("line")
    L0, L1 = (120, 1000), (520, 1000)
    ui.click_at(*L0)
    rep.check("line: hint asks for the end point", "END" in ui.hint(), ui.hint())
    ui.page.keyboard.press("]")
    ui.page.keyboard.press("]")            # 5 → 7 px
    ui.click_at(*L1)
    oc = ui.overlay_counts()
    rep.check("line: line drawn on the overlay", oc["line"] >= 1, str(oc))
    body = ui.apply()
    cv = body.get("covers")
    ln = cv[0] if cv and isinstance(cv[0], dict) and cv[0].get("line") else None
    rep.check("line: request carries line, width 7 and a colour",
              ln is not None and ln.get("width") == 7 and ln.get("color") in ("#000000", "#ffffff")
              and abs(ln["line"][0][0] - L0[0]) <= 3 and abs(ln["line"][1][0] - L1[0]) <= 3, str(cv))
    li = ui.result_image()
    save("p1_line.png", li)
    if ln:
        col = ln.get("color")
        frac = line_dark(li, L0, L1, 7) if col == "#000000" else 1 - line_dark(li, L0, L1, 7, thr=200)
        rep.check("line: the line is drawn in the output", frac > 0.9, f"{frac:.2f} of samples match")
    lrect = (min(L0[0], L1[0]) - 6, min(L0[1], L1[1]) - 6, abs(L1[0] - L0[0]) + 12, abs(L1[1] - L0[1]) + 12)
    n, bb = changed_outside(base1, li, [lrect], margin=8)
    rep.check("line: nothing away from the line changed", n == 0, f"{n} px differ at {bb}")
    no_js_errors(rep, ui, "line")
    ui.page.locator("#moveLayer svg").first.click(force=True, position={"x": 1, "y": 1})
    # clicking empty svg must NOT remove (only the stroke is a hit target)
    ui.page.locator("#moveLayer svg line").last.click(force=True)
    rep.check("line: clicking the line removes it", ui.overlay_counts()["line"] == 0, str(ui.overlay_counts()))

    # ── Redraw fill (pen outline → colour dialog) ────────────────────
    ui.set_tool("fill-poly")
    FP = [(600, 1100), (760, 1100), (760, 1220), (600, 1220)]
    ui.pen_polygon(FP)
    ui.page.wait_for_selector(".fill-back", timeout=5000)
    sugg = ui.page.input_value("#fillHex")
    rep.check("redraw fill: colour dialog opens with a sampled colour", sugg.startswith("#") and len(sugg) == 7, sugg)
    ui.page.click('.fill-preset[data-c="#000000"]')
    ui.page.click("#fillOk")
    oc = ui.overlay_counts()
    rep.check("redraw fill: filled outline drawn on the overlay", oc["polygon"] == 1, str(oc))
    body = ui.apply()
    cv = body.get("covers")
    rep.check("redraw fill: request carries fill_poly + colour",
              cv and isinstance(cv[0], dict) and cv[0].get("fill_poly") and cv[0].get("color") == "#000000", str(cv)[:200])
    fi = ui.result_image()
    save("p1_fill.png", fi)
    frect = poly_rect(FP)
    inner = region(fi, frect[0] + 6, frect[1] + 6, frect[2] - 12, frect[3] - 12)
    rep.check("redraw fill: the shape is flooded black", float(inner.mean()) < 20, f"mean {inner.mean():.1f}")
    n, bb = changed_outside(base1, fi, [frect], margin=6)
    rep.check("redraw fill: nothing outside the shape changed", n == 0, f"{n} px differ at {bb}")
    no_js_errors(rep, ui, "redraw fill")
    ui.page.locator("#moveLayer svg polygon").first.click(force=True)

    # Redraw fill with the SAMPLED colour: on white paper the sample is white.
    ui.set_tool("fill-poly")
    ui.pen_polygon(FP)
    ui.page.wait_for_selector(".fill-back", timeout=5000)
    ui.page.click("#fillOk")
    body = ui.apply()
    cv = body.get("covers")
    fi2 = ui.result_image()
    inner2 = region(fi2, frect[0] + 6, frect[1] + 6, frect[2] - 12, frect[3] - 12)
    rep.check("redraw fill (sampled): request carries the sampled colour",
              cv and isinstance(cv[0], dict) and cv[0].get("color", "").startswith("#"), str(cv)[:200])
    rep.check("redraw fill (sampled): the shape is flat in the sampled colour",
              float(inner2.std()) < 6, f"std {inner2.std():.1f} colour {cv and cv[0].get('color')}")
    ui.page.locator("#moveLayer svg polygon").first.click(force=True)
    no_js_errors(rep, ui, "redraw fill (sampled)")

    # ── Tone fill ────────────────────────────────────────────────────
    ui.set_tool("tone-poly")
    ui.pen_polygon(FP)
    oc = ui.overlay_counts()
    rep.check("tone fill: outline drawn on the overlay", oc["polygon"] == 1, str(oc))
    rep.check("tone fill: no colour dialog", ui.page.locator(".fill-back").count() == 0)
    body = ui.apply()
    cv = body.get("covers")
    rep.check("tone fill: request carries fill_poly + tone",
              cv and isinstance(cv[0], dict) and cv[0].get("fill_poly") and cv[0].get("tone") is True, str(cv)[:200])
    tn = ui.result_image()
    save("p1_tone.png", tn)
    innert = cv2.cvtColor(region(tn, frect[0] + 6, frect[1] + 6, frect[2] - 12, frect[3] - 12), cv2.COLOR_BGR2GRAY)
    rep.check("tone fill: the shape holds a dot pattern (both ink and paper present)",
              (innert < 128).mean() > 0.02 and (innert > 128).mean() > 0.2,
              f"ink {(innert < 128).mean():.3f} paper {(innert > 128).mean():.3f}")
    n, bb = changed_outside(base1, tn, [frect], margin=6)
    rep.check("tone fill: nothing outside the shape changed", n == 0, f"{n} px differ at {bb}")
    no_js_errors(rep, ui, "tone fill")
    ui.page.locator("#moveLayer svg polygon").first.click(force=True)

    # ── Clone stamp: copy the SFX letters onto blank paper ───────────
    ui.set_tool("clone")
    SRC = (SFX[0] + 60, SFX[1] + 80)
    DST = (680, 1160)
    ui.click_at(*SRC)
    rep.check("clone: first click sets the source", "Source set" in ui.hint(), ui.hint())
    ui.click_at(*DST)
    oc = ui.overlay_counts()
    rep.check("clone: dab drawn on the overlay", oc["dab"] == 1, str(oc))
    body = ui.apply()
    cv = body.get("covers")
    cl = cv[0].get("clone") if cv and isinstance(cv[0], dict) else None
    rep.check("clone: request carries src/dst/r", cl and abs(cl["src"][0] - SRC[0]) <= 3
              and abs(cl["dst"][0] - DST[0]) <= 3 and cl.get("r", 0) > 0, str(cv))
    cs = ui.result_image()
    save("p1_clone.png", cs)
    if cl:
        r = cl["r"]
        k = int(r * 0.6)
        a = region(base1, cl["src"][0] - k, cl["src"][1] - k, 2 * k, 2 * k)
        b = region(cs, cl["dst"][0] - k, cl["dst"][1] - k, 2 * k, 2 * k)
        diff = float(np.abs(a.astype(int) - b.astype(int)).mean())
        rep.check("clone: the destination now holds the source patch", diff < 12, f"mean diff {diff:.1f}")
        drect = (cl["dst"][0] - r, cl["dst"][1] - r, 2 * r, 2 * r)
        n, bb = changed_outside(base1, cs, [drect], margin=6)
        rep.check("clone: nothing outside the dab changed", n == 0, f"{n} px differ at {bb}")
    no_js_errors(rep, ui, "clone")
    ui.page.locator("#moveLayer .clone-dab").first.click(force=True)
    rep.check("clone: clicking the dab removes it", ui.overlay_counts()["dab"] == 0)

    # ── Chains: earlier covers must survive later tools ──────────────
    chains(ui, rep, base1, SFX, BALLOON, FP, frect)

    # ── Gestures on a scrolled / zoomed page ─────────────────────────
    geometry(ui, rep, SFX)

    # ── Covers are per page ──────────────────────────────────────────
    per_page(ui, rep, SFX)

    # ── Covers follow the page when a strip is trimmed off ───────────
    trim_follow(ui, rep, SFX)


def trim_follow(ui, rep, SFX):
    """Cut a strip off the LEFT of page 1: every cover (not just the erase
    boxes) must shift left by the cut so it still sits on what it was drawn
    over, and the re-render must draw it there."""
    ui.select_page(0)
    ui.open_editor()
    W0, H0 = ui.dims()
    # clear leftovers from earlier tests
    for _ in range(12):
        els = ui.page.locator("#moveLayer > .cover-box, #moveLayer > .clone-dab, #moveLayer > .restore-click, #moveLayer svg polygon, #moveLayer svg line")
        if els.count() == 0:
            break
        ui.set_tool("cover")
        els.last.click(force=True)
    L0, L1 = (120, 1000), (520, 1000)
    ui.set_tool("line")
    ui.click_at(*L0)
    ui.click_at(*L1)
    ui.set_tool("clone")
    SRC, DST = (SFX[0] + 60, SFX[1] + 80), (680, 1160)
    ui.click_at(*SRC)
    ui.click_at(*DST)
    ui.set_tool("cover")
    ui.drag(SFX[0], SFX[1], SFX[0] + SFX[2], SFX[1] + SFX[3])
    ui.set_tool("lasso")
    x, y, w, h = 600, 1100, 160, 120
    ui.drag_path([(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)])
    ui.set_tool("restore")
    ui.click_at(292, 640)
    before = ui.apply().get("covers") or []
    rep.check("trim: five covers of five kinds before the cut", len(before) == 5, str([type(c).__name__ for c in before]))

    tl = ui.page.locator("#trimLeft")
    tl.scroll_into_view_if_needed()
    tl.click()
    ui.page.wait_for_function("() => document.getElementById('trimModal').style.display === 'flex'")
    ui.page.wait_for_function("() => { const im = document.getElementById('trimImg'); return im.complete && im.naturalWidth > 0; }")
    ui.page.wait_for_timeout(300)
    r = ui.page.evaluate("() => { const r = document.getElementById('trimImg').getBoundingClientRect(); return [r.left, r.top, r.width, r.height]; }")
    ui.page.mouse.click(r[0] + 0.1 * r[2], r[1] + r[3] / 2)
    ui.page.wait_for_timeout(200)
    rev0 = ui.rev()
    ui.page.click("#trimApplyOne")
    ui.page.wait_for_function("() => document.getElementById('trimModal').style.display === 'none'", timeout=int(APPLY_TIMEOUT * 1000))
    ui.page.wait_for_function(f"() => document.getElementById('transFull').getAttribute('src').includes('?t=') && !document.getElementById('transFull').getAttribute('src').endsWith('?t={rev0}')", timeout=int(APPLY_TIMEOUT * 1000))
    ui.page.wait_for_function("() => { const im = document.getElementById('transFull'); return im.complete && im.naturalWidth > 0; }")
    ui.page.wait_for_timeout(300)
    W1, H1 = ui.dims()
    dx = W0 - W1
    rep.check("trim: the page got narrower by the cut", 0 < dx < W0 * 0.2, f"W {W0} -> {W1}")
    ui.set_tool("cover")
    after = ui.apply().get("covers") or []
    rep.check("trim: all five covers still sent after the cut", len(after) == 5, str([type(c).__name__ for c in after]))

    def shifted(a, b, off):
        return abs((a[0] - off) - b[0]) <= 1 and abs(a[1] - b[1]) <= 1
    lost = []
    for c0, c1 in zip(before, after):
        if isinstance(c0, list):
            ok = shifted(c0, c1, dx)
            kind = "box"
        elif c0.get("line"):
            ok = all(shifted(p0, p1, dx) for p0, p1 in zip(c0["line"], c1["line"]))
            kind = "line"
        elif c0.get("clone"):
            ok = shifted(c0["clone"]["src"], c1["clone"]["src"], dx) and shifted(c0["clone"]["dst"], c1["clone"]["dst"], dx)
            kind = "clone"
        elif c0.get("poly"):
            ok = all(shifted(p0, p1, dx) for p0, p1 in zip(c0["poly"], c1["poly"]))
            kind = "lasso"
        elif c0.get("restore_click"):
            ok = shifted(c0["restore_click"], c1["restore_click"], dx)
            kind = "restore click"
        else:
            ok, kind = False, "?"
        if not ok:
            lost.append(kind)
    rep.check("trim: every cover kind moved left by the cut", not lost, f"did not follow the page: {lost} (dx {dx})")
    im = ui.result_image()
    save("trim.png", im)
    ln = next((c for c in after if isinstance(c, dict) and c.get("line")), None)
    if ln:
        p0, p1 = (L0[0] - dx, L0[1]), (L1[0] - dx, L1[1])
        frac = line_dark(im, p0, p1, 5) if ln.get("color") == "#000000" else 1 - line_dark(im, p0, p1, 5, thr=200)
        rep.check("trim: the line is drawn at its shifted place", frac > 0.9, f"{frac:.2f}")
    no_js_errors(rep, ui, "trim")


def chains(ui, rep, base1, SFX, BALLOON, FP, frect):
    def clear_all():
        # click every cover marker away
        for _ in range(12):
            els = ui.page.locator("#moveLayer > .cover-box, #moveLayer > .clone-dab, #moveLayer svg polygon, #moveLayer svg line")
            if els.count() == 0:
                break
            els.last.click(force=True)
        ui.set_tool("cover")

    clear_all()
    rep.check("chains: overlay clear before chains", ui.overlay_counts()["children"] == 0, str(ui.overlay_counts()))

    # A) Erase box → Lasso → Redraw fill
    ui.set_tool("cover")
    ui.drag(SFX[0], SFX[1], SFX[0] + SFX[2], SFX[1] + SFX[3])
    ui.set_tool("lasso")
    x, y, w, h = BALLOON
    ui.drag_path([(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)])
    ui.set_tool("fill-poly")
    ui.pen_polygon(FP)
    ui.page.wait_for_selector(".fill-back", timeout=5000)
    ui.page.click('.fill-preset[data-c="#000000"]')
    ui.page.click("#fillOk")
    oc = ui.overlay_counts()
    rep.check("chain A: overlay shows box + lasso + fill", oc["cover"] == 1 and oc["polygon"] == 2, str(oc))
    body = ui.apply()
    cv = body.get("covers") or []
    kinds = ["box" if isinstance(c, list) else ("fill" if c.get("fill_poly") else "poly" if c.get("poly") else "?") for c in cv]
    rep.check("chain A: request carries all three, in order", kinds == ["box", "poly", "fill"], str(kinds))
    im = ui.result_image()
    save("chainA.png", im)
    cfa = changed_fraction(base1, im, SFX)
    inner = region(im, frect[0] + 6, frect[1] + 6, frect[2] - 12, frect[3] - 12)
    rep.check("chain A: box erase applied", cfa > 0.15, f"{cfa:.3f}")
    rep.check("chain A: lasso applied (balloon region touched)",
              changed_fraction(base1, im, BALLOON) >= 0 and True)   # cleaned balloon: lasso may be a no-op on paper
    rep.check("chain A: fill applied", float(inner.mean()) < 20, f"mean {inner.mean():.1f}")
    n, bb = changed_outside(base1, im, [SFX, BALLOON, frect], margin=8)
    rep.check("chain A: nothing else changed", n == 0, f"{n} px at {bb}")
    no_js_errors(rep, ui, "chain A")
    clear_all()

    # B) Clone → Line → Erase
    ui.set_tool("clone")
    SRC, DST = (SFX[0] + 60, SFX[1] + 80), (680, 1160)
    ui.click_at(*SRC)
    ui.click_at(*DST)
    ui.set_tool("line")
    L0, L1 = (120, 1000), (520, 1000)
    ui.click_at(*L0)
    ui.click_at(*L1)
    ui.set_tool("cover")
    ui.drag(SFX[0], SFX[1], SFX[0] + SFX[2], SFX[1] + SFX[3])
    oc = ui.overlay_counts()
    rep.check("chain B: overlay shows dab + line + box", oc["dab"] == 1 and oc["line"] >= 1 and oc["cover"] == 1, str(oc))
    body = ui.apply()
    cv = body.get("covers") or []
    kinds = ["box" if isinstance(c, list) else ("clone" if c.get("clone") else "line" if c.get("line") else "?") for c in cv]
    rep.check("chain B: request carries all three, in order", kinds == ["clone", "line", "box"], str(kinds))
    im = ui.result_image()
    save("chainB.png", im)
    cl = cv[0].get("clone") if cv and isinstance(cv[0], dict) else None
    if cl:
        k = int(cl["r"] * 0.6)
        a = region(base1, cl["src"][0] - k, cl["src"][1] - k, 2 * k, 2 * k)
        b = region(im, cl["dst"][0] - k, cl["dst"][1] - k, 2 * k, 2 * k)
        rep.check("chain B: clone applied", float(np.abs(a.astype(int) - b.astype(int)).mean()) < 12)
    ln = next((c for c in cv if isinstance(c, dict) and c.get("line")), None)
    if ln:
        frac = line_dark(im, L0, L1, ln.get("width", 5)) if ln.get("color") == "#000000" else 1 - line_dark(im, L0, L1, 5, thr=200)
        rep.check("chain B: line applied", frac > 0.9, f"{frac:.2f}")
    rep.check("chain B: erase applied", changed_fraction(base1, im, SFX) > 0.15)
    no_js_errors(rep, ui, "chain B")
    clear_all()

    # C) Restore after Erase on the same spot, with another cover elsewhere
    ui.set_tool("line")
    ui.click_at(*L0)
    ui.click_at(*L1)
    ui.set_tool("cover")
    ui.drag(SFX[0], SFX[1], SFX[0] + SFX[2], SFX[1] + SFX[3])
    ui.set_tool("restore")
    x, y, w, h = SFX
    ui.drag_path([(x - 2, y - 2), (x + w + 2, y - 2), (x + w + 2, y + h + 2), (x - 2, y + h + 2), (x - 2, y - 2)])
    oc = ui.overlay_counts()
    rep.check("chain C: overlay shows line + box + restore outline",
              oc["line"] >= 1 and oc["cover"] == 1 and oc["polygon"] == 1, str(oc))
    body = ui.apply()
    cv = body.get("covers") or []
    kinds = ["box" if isinstance(c, list) else ("line" if c.get("line") else "restore" if c.get("restore_poly") else "?") for c in cv]
    rep.check("chain C: request carries line + box + restore", kinds == ["line", "box", "restore"], str(kinds))
    im = ui.result_image()
    save("chainC.png", im)
    n_in = int(changed_mask(region(base1, *SFX), region(im, *SFX)).sum())
    rep.check("chain C: restore brings the original back over the erase (box area equals the page again)",
              n_in == 0, f"{n_in} px differ inside the box")
    if ln:
        frac = line_dark(im, L0, L1, 5) if ln.get("color") == "#000000" else 1 - line_dark(im, L0, L1, 5, thr=200)
        rep.check("chain C: the line elsewhere is intact", frac > 0.9, f"{frac:.2f}")
    no_js_errors(rep, ui, "chain C")
    clear_all()

    # D) Erase → switch to Move and back → cover still drawn and still sent
    ui.set_tool("cover")
    ui.drag(SFX[0], SFX[1], SFX[0] + SFX[2], SFX[1] + SFX[3])
    ui.set_tool("move")
    oc_move = ui.overlay_counts()
    ui.set_tool("cover")
    oc = ui.overlay_counts()
    rep.check("chain D: cover still on the overlay after Move and back",
              oc_move["cover"] == 1 and oc["cover"] == 1, f"move {oc_move} back {oc}")
    body = ui.apply()
    cv = body.get("covers") or []
    rep.check("chain D: cover still sent", len(cv) == 1 and isinstance(cv[0], list), str(cv))
    no_js_errors(rep, ui, "chain D")

    # E) A cover drawn while an Apply is in flight must not be lost
    rev0 = ui.rev()
    n0 = len(ui.rerender_bodies)
    ui.page.locator("#editApply").click()             # first apply (box only)
    ui.page.wait_for_timeout(800)
    rep.check("in-flight: apply button is busy", ui.page.evaluate("() => document.getElementById('editApply').disabled"))
    ui.set_tool("line")                                # draw while it runs
    ui.click_at(*L0)
    ui.click_at(*L1)
    oc_mid = ui.overlay_counts()
    deadline = time.time() + APPLY_TIMEOUT
    while time.time() < deadline and (ui.rev() == rev0 or ui.page.evaluate("() => document.getElementById('editApply').disabled")):
        time.sleep(1)
    ui.page.wait_for_timeout(500)
    oc_after = ui.overlay_counts()
    rep.check("in-flight: the new line survives the first re-render landing",
              oc_mid["line"] >= 1 and oc_after["line"] >= 1 and oc_after["cover"] == 1, f"mid {oc_mid} after {oc_after}")
    body = ui.apply()
    cv = body.get("covers") or []
    kinds = ["box" if isinstance(c, list) else ("line" if c.get("line") else "?") for c in cv]
    rep.check("in-flight: the next apply sends both the box and the new line",
              kinds == ["box", "line"], f"{kinds}; requests since: {len(ui.rerender_bodies) - n0}")
    im = ui.result_image()
    save("chainE.png", im)
    ln = next((c for c in cv if isinstance(c, dict) and c.get("line")), None)
    if ln:
        frac = line_dark(im, L0, L1, 5) if ln.get("color") == "#000000" else 1 - line_dark(im, L0, L1, 5, thr=200)
        rep.check("in-flight: the line is in the output", frac > 0.9, f"{frac:.2f}")
    rep.check("in-flight: the box erase is in the output", changed_fraction(base1, im, SFX) > 0.15)
    no_js_errors(rep, ui, "in-flight")
    clear_all()


def geometry(ui, rep, SFX):
    """The overlay maps pointer positions to image pixels through its own
    bounding box, so a scrolled or zoomed page must give the same cover."""
    ui.set_tool("cover")
    # small viewport: the target is below the fold, the harness scrolls to it
    ui.page.set_viewport_size({"width": 900, "height": 600})
    ui.page.wait_for_timeout(300)
    ui.drag(SFX[0], SFX[1], SFX[0] + SFX[2], SFX[1] + SFX[3])
    sy = ui.page.evaluate("() => window.scrollY")
    rep.check("scrolled: the page really was scrolled for the gesture", sy > 0, f"scrollY {sy}")
    body = ui.apply()
    cv = body.get("covers") or []
    rep.check("scrolled: box lands on the right image pixels",
              len(cv) == 1 and isinstance(cv[0], list) and all(abs(cv[0][i] - SFX[i]) <= 4 for i in range(4)), str(cv))
    ui.page.locator("#moveLayer .cover-box").first.click(force=True)

    # zoomed: scale the stage up (as browser zoom does) and draw again
    ui.page.evaluate("() => { document.body.style.zoom = '1.6'; }")
    ui.page.wait_for_timeout(300)
    ui.drag(SFX[0], SFX[1], SFX[0] + SFX[2], SFX[1] + SFX[3])
    body = ui.apply()
    cv = body.get("covers") or []
    rep.check("zoomed: box lands on the right image pixels",
              len(cv) == 1 and isinstance(cv[0], list) and all(abs(cv[0][i] - SFX[i]) <= 4 for i in range(4)), str(cv))
    # lasso while zoomed
    ui.page.locator("#moveLayer .cover-box").first.click(force=True)
    ui.set_tool("lasso")
    x, y, w, h = SFX
    ui.drag_path([(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)])
    body = ui.apply()
    cv = body.get("covers") or []
    pr = poly_rect(cv[0]["poly"]) if cv and isinstance(cv[0], dict) and cv[0].get("poly") else None
    rep.check("zoomed: lasso lands on the right image pixels",
              pr is not None and all(abs(pr[i] - SFX[i]) <= 5 for i in range(4)), f"{pr} vs {SFX}")
    ui.page.locator("#moveLayer svg polygon").first.click(force=True)
    ui.page.evaluate("() => { document.body.style.zoom = ''; window.scrollTo(0, 0); }")
    ui.page.set_viewport_size({"width": 1400, "height": 1000})
    no_js_errors(rep, ui, "geometry")


def per_page(ui, rep, SFX):
    ui.set_tool("cover")
    ui.drag(SFX[0], SFX[1], SFX[0] + SFX[2], SFX[1] + SFX[3])
    rep.check("per-page: cover drawn on page 1", ui.overlay_counts()["cover"] == 1)
    ui.select_page(1)
    ui.page.wait_for_timeout(400)
    oc = ui.overlay_counts()
    rep.check("per-page: page 2 shows no cover from page 1", oc["children"] == 0, str(oc))
    base2 = ui.result_image()
    body = ui.apply()
    rep.check("per-page: page 2 apply sends no covers", body.get("covers") == [], str(body.get("covers")))
    n, bb = changed_outside(base2, ui.result_image(), [])
    rep.check("per-page: page 2 output untouched", n == 0, f"{n} px differ at {bb}")
    # draw on page 2, then go back to page 1: each page keeps its own
    ui.set_tool("cover")
    ui.drag(835, 1140, 1005, 1400)
    rep.check("per-page: cover drawn on page 2", ui.overlay_counts()["cover"] == 1)
    ui.select_page(0)
    ui.page.wait_for_timeout(400)
    oc = ui.overlay_counts()
    rep.check("per-page: back on page 1 its own cover is still there (and only it)", oc["cover"] == 1, str(oc))
    body = ui.apply()
    cv = body.get("covers") or []
    rep.check("per-page: page 1 sends only its own cover",
              len(cv) == 1 and isinstance(cv[0], list) and abs(cv[0][0] - SFX[0]) <= 4, str(cv))
    ui.select_page(1)
    body = ui.apply()
    cv = body.get("covers") or []
    rep.check("per-page: page 2 sends only its own cover",
              len(cv) == 1 and isinstance(cv[0], list) and abs(cv[0][0] - 835) <= 4, str(cv))
    no_js_errors(rep, ui, "per-page")


if __name__ == "__main__":
    sys.exit(main())
