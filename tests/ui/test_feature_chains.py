"""Features used one after another must not clobber each other's settings or
per-page state. Each chain reports pass/fail with what got lost."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import (Check, Session, build_inputs, decode, fetch, page_paths,
                      skip_unless_server, zip_entries)

skip_unless_server()
inputs = build_inputs()
c = Check("feature chains")
P3, P4 = page_paths("3.jpg", "4.jpg")
O3, O4 = cv2.imread(P3), cv2.imread(P4)

WM = {"watermark": "@chain", "wmStyle": "bold", "wmPlace": "tl", "wmSize": "l",
      "wmOpacity": "35", "credit": "TL: Chain"}
OTHER = {"fontSelect": None, "pageFinish": "restore", "textCase": "keep",
         "fontVariety": "expressive", "mangaTitle": "One Piece", "glossary": "ゾロ = Zoro"}
TOGGLES = {"hdUpscale": True, "compressOut": True, "styleFonts": True, "wmTileToo": True,
           "maxQuality": False, "translateSfx": True}


def apply_settings(s):
    first_font = s.page.evaluate("document.querySelectorAll('#fontSelect option')[1].value")
    OTHER["fontSelect"] = first_font
    for k, v in WM.items():
        s.select(k, v) if k.startswith("wm") else s.set_value(k, v)
    for k, v in OTHER.items():
        s.select(k, v) if k in ("fontSelect", "pageFinish", "textCase", "fontVariety") else s.set_value(k, v)
    for k, v in TOGGLES.items():
        s.set_toggle(k, v)


def snapshot(s):
    out = {k: s.value(k) for k in list(WM) + list(OTHER)}
    out.update({k: s.checked(k) for k in TOGGLES})
    out["workflow"] = s.page.evaluate("document.querySelector('.wf-card.active').dataset.wf")
    return out


def expected():
    out = dict(WM); out.update(OTHER); out.update(TOGGLES)
    return out


def lost(before, after):
    return {k: (before.get(k), after.get(k)) for k in before if before.get(k) != after.get(k)}


def save_zip(s):
    name, data = s.download("#savePagesBtn")
    entries, zf = zip_entries(data)
    return [e[0] for e in entries], {n: decode(zf.read(n)) for n, _, _ in entries}


def wait_unlocked(s, btn):
    s.page.wait_for_function(f"() => !document.getElementById('{btn}').disabled", timeout=300000)


def trim(s, side, frac_pos):
    p = s.page
    p.click("#trim" + side.capitalize())
    p.wait_for_selector("#trimModal", state="visible")
    p.wait_for_function("() => document.getElementById('trimImg').naturalWidth > 0")
    box = p.locator("#trimImg").bounding_box()
    if side in ("left", "right"):
        p.mouse.click(box["x"] + box["width"] * frac_pos, box["y"] + box["height"] / 2)
    else:
        p.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] * frac_pos)
    ro = s.text("#trimReadout")
    p.click("#trimApplyOne")
    p.wait_for_selector("#trimModal", state="hidden", timeout=60000)
    return float(ro.split("(")[1].split("%")[0]) / 100


with Session() as s:
    p = s.page
    p.wait_for_function("() => document.querySelectorAll('#fontSelect option').length > 1", timeout=15000)
    s.record_requests()
    apply_settings(s)
    base = snapshot(s)

    # ── Chain 1: Rotate (↻) → Trim → Save pages as ZIP ──
    s.set_workflow("watermark-only")
    s.upload([P3, P4])
    s.wait_preview()
    s.go()
    s.wait_all_done(2, timeout=120000)
    s.click_chip(0)
    s.set_toggle("orientAll", False)
    p.click("#orientRight")
    wait_unlocked(s, "orientRight")
    frac = trim(s, "top", 0.12)
    names, imgs = save_zip(s)
    turned = cv2.rotate(O3, cv2.ROTATE_90_CLOCKWISE)
    cut = round(frac * turned.shape[0])
    want = turned[cut:]
    ok1 = names == ["3.png", "4.jpg"] and imgs["3.png"].shape == want.shape and np.array_equal(imgs["3.png"], want)
    ok2 = np.array_equal(imgs["4.jpg"], O4)
    c.ok(ok1, f"CHAIN rotate→trim→zip: page 1 is rotated AND trimmed ({imgs['3.png'].shape[:2]} vs {want.shape[:2]})")
    c.ok(ok2, "CHAIN rotate→trim→zip: page 2 untouched")
    snap = snapshot(s)
    c.ok(lost(base, {**snap, "workflow": base["workflow"]}) == {}, f"CHAIN rotate→trim→zip: settings intact, lost={lost(base, snap)}")

    # ── Chain 2: watermark fields → Clean — no key → Watermark only (diff bodies) ──
    p.click("#newBtn")
    s.set_workflow("local-clean")
    s.upload([P3])
    s.wait_preview()
    s.go()
    s.wait_all_done(1, timeout=300000)
    after_clean = snapshot(s)
    p.click("#newBtn")
    s.set_workflow("watermark-only")
    s.upload([P3])
    s.wait_preview()
    s.go()
    s.wait_all_done(1, timeout=120000)
    after_wm = snapshot(s)
    req_clean = [f for u, f in s.requests if u.endswith("/api/localclean")][-1]
    req_wm = [f for u, f in s.requests if u.endswith("/api/rawify")][-1]
    want_wm = {"watermark": "@chain", "wm_style": "bold+tile", "wm_place": "tl", "wm_size": "l",
               "wm_opacity": "35", "credit": "TL: Chain"}
    d_clean = {k: (req_clean.get(k), v) for k, v in want_wm.items() if req_clean.get(k) != v}
    d_wm = {k: (req_wm.get(k), v) for k, v in want_wm.items() if req_wm.get(k) != v}
    c.ok(not d_clean, f"CHAIN wm→clean→wm: Clean request carries the typed watermark fields {d_clean}")
    c.ok(not d_wm, f"CHAIN wm→clean→wm: Watermark-only request carries the same fields {d_wm}")
    c.eq(req_clean.get("hd"), "true", "CHAIN wm→clean→wm: Clean sends the HD toggle")
    c.eq(req_wm.get("style"), "none", "CHAIN wm→clean→wm: Watermark-only asks for no raw effect")
    leak = {k for k in req_wm if k in ("hd", "clean_only", "font", "finish", "turn")}
    c.ok(not leak, f"CHAIN wm→clean→wm: no Clean-only fields leak into the Watermark-only body {leak}")
    same = {k: (req_clean.get(k), req_wm.get(k)) for k in ("compress", "max_quality", "gpu_cap") if req_clean.get(k) != req_wm.get(k)}
    c.ok(not same, f"CHAIN wm→clean→wm: shared options identical across both bodies {same}")
    l1 = lost(base, {**after_clean, "workflow": base["workflow"]})
    l2 = lost(base, {**after_wm, "workflow": base["workflow"]})
    c.ok(not l1 and not l2, f"CHAIN wm→clean→wm: fields unchanged between runs (after clean {l1}, after wm {l2})")
    res = decode(fetch(f"/api/result/{s.active_task()}"))
    c.ok(not np.array_equal(res, O3), "CHAIN wm→clean→wm: the stamp is actually on the page")

    # ── Chain 3: Remove BG (cut-out) → Rotate ──
    p.click("#newBtn")
    s.set_workflow("watermark-only")
    s.upload([os.path.join(inputs, "photo4.jpg")])
    s.wait_preview()
    s.go()
    s.wait_all_done(1, timeout=120000)
    p.click("#cutoutBtn")
    wait_unlocked(s, "cutoutBtn")
    _, imgs = save_zip(s)
    cut_img = imgs["photo4.png"]
    photo = cv2.imread(os.path.join(inputs, "photo4.jpg"))
    c.ok(cut_img.shape[0] < photo.shape[0] * 0.9, f"CHAIN cutout→rotate: cut-out happened {photo.shape[:2]} → {cut_img.shape[:2]}")
    p.click("#orientRight")
    wait_unlocked(s, "orientRight")
    _, imgs2 = save_zip(s)
    c.ok(np.array_equal(imgs2["photo4.png"], cv2.rotate(cut_img, cv2.ROTATE_90_CLOCKWISE)),
         f"CHAIN cutout→rotate: rotation applied to the CUT page, not the photo ({imgs2['photo4.png'].shape[:2]})")
    snap = snapshot(s)
    c.ok(lost(base, {**snap, "workflow": base["workflow"]}) == {}, f"CHAIN cutout→rotate: settings intact, lost={lost(base, snap)}")

    # ── Chain 4: switching the workflow card must not change other settings ──
    p.click("#newBtn")
    cards = p.evaluate("[...document.querySelectorAll('.wf-card')].map(c => c.dataset.wf)")
    drift = {}
    for wf in cards:
        s.set_workflow(wf)
        snap = snapshot(s)
        d = lost(base, {**snap, "workflow": base["workflow"]})
        if d:
            drift[wf] = d
    c.ok(not drift, f"CHAIN card switching: no setting changed by any card {drift}")
    ls_after = p.evaluate("JSON.stringify(Object.fromEntries(Object.keys(localStorage).filter(k => k.startsWith('manga_') && k !== 'manga_workflow').map(k => [k, localStorage.getItem(k)])))")
    s.set_workflow("scan-translate")
    ls_after2 = p.evaluate("JSON.stringify(Object.fromEntries(Object.keys(localStorage).filter(k => k.startsWith('manga_') && k !== 'manga_workflow').map(k => [k, localStorage.getItem(k)])))")
    c.eq(ls_after, ls_after2, "CHAIN card switching: localStorage (bar the card itself) untouched")

    # ── Chain 5: reload mid-way → everything comes back ──
    s.set_workflow("local-clean")
    s.upload([P3])
    s.wait_preview()
    s.go()
    s.wait_all_done(1, timeout=300000)
    before_reload = snapshot(s)
    s.reload()
    p.wait_for_function("() => document.querySelectorAll('#fontSelect option').length > 1", timeout=15000)
    p.wait_for_timeout(300)
    after_reload = snapshot(s)
    c.ok(lost(before_reload, after_reload) == {}, f"CHAIN reload: every setting restored exactly, lost={lost(before_reload, after_reload)}")
    c.eq(after_reload["workflow"], "local-clean", "CHAIN reload: workflow card restored")
    exp = expected()
    c.ok(all(after_reload[k] == v for k, v in exp.items()), f"CHAIN reload: values are the ones set at the start {lost(exp, after_reload)}")

c.finish(s)
