"""Browser test of the editor's Move / Resize / Edit / Undo / Redo tools.

Runs the real page in Chromium against a running server, translates two
real pages through it (see stub_llm.py for running without an API key),
then exercises each tool with real mouse gestures and checks three things
after every Apply & Re-render: the request body the page sent, the pixels of
the result image, and the UI state afterwards. Also checks that nothing
done on page 1 leaks onto page 2, and that tools used in sequence keep each
other's work.

    python tests/ui/stub_llm.py 8123 &
    ANTHROPIC_BASE_URL=http://127.0.0.1:8123 PORT=8021 python app.py &
    python tests/ui/test_editor_tools.py

Skips (exit 0) when no server is up.
"""
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from editor_harness import (PAGES, Browser, Report, dark_count, fetch_result,   # noqa: E402
                     region_diff, skip_unless_ready)

TOL = 4          # px tolerance on screen / image geometry


def near(a, b, tol=TOL):
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


def center(rect):
    return rect[0] + rect[2] / 2, rect[1] + rect[3] / 2


def handle_center(br, id_, d):
    return br.page.evaluate("""([id, d]) => {
        const b = [...document.querySelectorAll('#moveLayer .resize-box')]
          .find(b => ((b.querySelector('.move-tag')||{}).textContent||'').replace('#','') === String(id));
        const h = b.querySelector('.rsz-' + d).getBoundingClientRect();
        return [h.left + h.width / 2, h.top + h.height / 2]; }""", [str(id_), d])


def pick_items(br, n):
    """The n largest numbered item boxes on the overlay (ids as strings)."""
    boxes = [b for b in br.overlay_boxes() if b["id"].isdigit()]
    boxes.sort(key=lambda b: -(b["img"][2] * b["img"][3]))
    return [b["id"] for b in boxes[:n]]


def main():
    skip_unless_ready()
    if len(PAGES) < 2:
        print("SKIP: need two pages in MT_PAGES for the isolation checks")
        return 0
    rep = Report()
    br = Browser(headed=bool(__import__("os").environ.get("MT_HEADED")))
    try:
        t0 = time.time()
        tasks = br.translate_pages(PAGES[:2])
        print(f"translated {len(tasks)} pages in {time.time() - t0:.0f}s: {tasks}", flush=True)
        rep.check("two pages translated", len(tasks) == 2)
        br.tab("translated")
        br.set_tool("move")
        ids = pick_items(br, 8)
        rep.check("page has enough numbered items on the overlay", len(ids) >= 8, str(ids))
        A, B, C, D, E, F, G, H = (ids + ["?"] * 8)[:8]
        task1, _ = br.active_task()
        sx, sy = br.image_scale()

        with rep.section("MOVE"):
            br.set_tool("move")
            br.scroll_to_box(A)
            a0 = br.box_by_id(A)
            cx, cy = center(a0["rect"])
            br.drag(cx, cy, cx + 60, cy + 30)
            a1 = br.box_by_id(A)
            exp_off = [60 * sx, 30 * sy]
            rep.check("move: box follows the drag on screen",
                      near(a1["img"][:2], [a0["img"][0] + exp_off[0], a0["img"][1] + exp_off[1]]), f"{a0['img']} -> {a1['img']}")
            before = fetch_result(*br.active_task())
            body = br.apply()
            rep.check("move: request carries the offset",
                      A in body.get("offsets", {}) and near(body["offsets"][A], exp_off), str(body.get("offsets")))
            after = fetch_result(*br.active_task())
            old_bb = a0["img"]
            new_bb = [old_bb[0] + exp_off[0], old_bb[1] + exp_off[1], old_bb[2], old_bb[3]]
            union = [min(old_bb[0], new_bb[0]), min(old_bb[1], new_bb[1]),
                     old_bb[2] + abs(exp_off[0]), old_bb[3] + abs(exp_off[1])]
            rep.check("move: result image changed where the text moved",
                      region_diff(before, after, union, pad=10) > 0.5,
                      f"diff={region_diff(before, after, union, pad=10):.2f}")
            a2 = br.box_by_id(A)
            rep.check("move: box stays at the moved spot after re-render",
                      a2 and near(a2["img"][:2], new_bb[:2]), f"{a2 and a2['img']} vs {new_bb}")

        with rep.section("RESIZE"):
            br.set_tool("resize")
            br.scroll_to_box(B)
            b0 = br.box_by_id(B)
            hx, hy = handle_center(br, B, "se")
            br.drag(hx, hy, hx + 50, hy + 30)
            b1 = br.box_by_id(B)
            exp_box = [b0["img"][0], b0["img"][1], b0["img"][2] + 50 * sx, b0["img"][3] + 30 * sy]
            rep.check("resize: SE handle grows the box on screen", near(b1["img"], exp_box), f"{b0['img']} -> {b1['img']}")
            before = fetch_result(*br.active_task())
            body = br.apply()
            rep.check("resize: request carries the new box",
                      B in body.get("boxes", {}) and near(body["boxes"][B], exp_box), f"{body.get('boxes')} vs {exp_box}")
            rep.check("resize: no stray offset for a resized item", B not in body.get("offsets", {}), str(body.get("offsets")))
            after = fetch_result(*br.active_task())
            rep.check("resize: result image changed inside the box",
                      region_diff(before, after, exp_box, pad=6) > 0.3, f"diff={region_diff(before, after, exp_box, pad=6):.2f}")
            b2 = br.box_by_id(B)
            rep.check("resize: box keeps its new size after re-render",
                      b2 and near(b2["img"], exp_box), f"{b2 and b2['img']} vs {exp_box}")
            # drag the middle to move it
            b2 = br.box_by_id(B)
            cx, cy = center(b2["rect"])
            br.drag(cx, cy, cx + 40, cy + 20)
            b3 = br.box_by_id(B)
            exp_box2 = [exp_box[0] + 40 * sx, exp_box[1] + 20 * sy, exp_box[2], exp_box[3]]
            rep.check("resize: dragging the middle moves the box on screen", near(b3["img"], exp_box2), f"{b2['img']} -> {b3['img']}")
            body = br.apply()
            rep.check("resize: moved box is sent as the new box",
                      B in body.get("boxes", {}) and near(body["boxes"][B], exp_box2), f"{body.get('boxes', {}).get(B)} vs {exp_box2}")
            # the W handle must not let the box drift once it hits the minimum width
            br.scroll_to_box(B)
            b4 = br.box_by_id(B)
            hx, hy = handle_center(br, B, "w")
            br.drag(hx, hy, hx + b4["rect"][2] + 80, hy)
            b5 = br.box_by_id(B)
            rep.check("resize: W handle past the minimum keeps the right edge in place",
                      abs((b5["img"][0] + b5["img"][2]) - (b4["img"][0] + b4["img"][2])) <= TOL,
                      f"right edge {b4['img'][0] + b4['img'][2]:.0f} -> {b5['img'][0] + b5['img'][2]:.0f}")
            br.page.click("#undoBtn")            # put B back to the moved/resized box
            # Resize > Apply > Undo > Apply must bring the DETECTED box back: the
            # server used to store the hand-resized box as the item's box, so once
            # applied a resize could never be undone.
            br.page.click("#undoBtn"); br.page.click("#undoBtn")
            body = br.apply()
            rep.check("resize: undo after Apply sends no box override", B not in body.get("boxes", {}), str(body.get("boxes")))
            b6 = br.box_by_id(B)
            rep.check("resize: undo after Apply restores the detected box on the overlay",
                      b6 and near(b6["img"], b0["img"]), f"{b6 and b6['img']} vs {b0['img']}")
            br.page.click("#redoBtn"); br.page.click("#redoBtn")
            body = br.apply()
            rep.check("resize: redo after that brings the resized box back",
                      B in body.get("boxes", {}) and near(body["boxes"][B], exp_box2), f"{body.get('boxes', {}).get(B)} vs {exp_box2}")

        with rep.section("EDIT"):
            br.set_tool("edit")
            br.scroll_to_box(C)
            c0 = br.box_by_id(C)
            br.page.mouse.click(*center(c0["rect"]))
            br.page.wait_for_selector(".edit-pop", timeout=3000)
            rep.check("edit: popover shows the current text",
                      br.page.input_value(".edit-pop .edit-pop-text") == f"LINE {C}", br.page.input_value(".edit-pop .edit-pop-text"))
            br.page.fill(".edit-pop .edit-pop-text", "HELLO WORLD")
            br.page.click('.edit-pop .clr-opt[data-c="white"]')
            br.page.click('.edit-pop .epop-ra[data-a="-45"]')
            br.page.evaluate("""() => { const r = document.querySelector('.edit-pop .epop-fs');
                r.value = 150; r.dispatchEvent(new Event('input', {bubbles: true})); }""")
            rep.check("edit: size readout follows the slider",
                      br.page.inner_text(".edit-pop .epop-fs-val").strip() == "150%", br.page.inner_text(".edit-pop .epop-fs-val"))
            rep.check("edit: tilt readout follows the quick button",
                      br.page.inner_text(".edit-pop .epop-rot-val").strip() == "-45°", br.page.inner_text(".edit-pop .epop-rot-val"))
            br.page.select_option(".edit-pop .epop-font", "Bangers-Regular.ttf")
            before = fetch_result(*br.active_task())
            body = br.apply(click=".edit-pop .epop-save")
            rep.check("edit: popover closes on Save", br.page.query_selector(".edit-pop") is None)
            rep.check("edit: text sent", body.get("edits", {}).get(C) == "HELLO WORLD", str(body.get("edits", {}).get(C)))
            rep.check("edit: colour sent", body.get("colors", {}).get(C) == "white", str(body.get("colors")))
            rep.check("edit: tilt sent", body.get("rotations", {}).get(C) == -45, str(body.get("rotations")))
            rep.check("edit: size sent", body.get("font_scales", {}).get(C) == 1.5, str(body.get("font_scales")))
            rep.check("edit: font sent", body.get("fonts", {}).get(C) == "Bangers-Regular.ttf", str(body.get("fonts")))
            after = fetch_result(*br.active_task())
            rep.check("edit: result image changed inside the box",
                      region_diff(before, after, c0["img"], pad=20) > 0.5, f"diff={region_diff(before, after, c0['img'], pad=20):.2f}")
            det = br.details()
            rep.check("edit: Details row shows the new text", det.get(C) == "HELLO WORLD", str(det.get(C)))
            rep.check("edit: Details font dropdown follows",
                      br.page.evaluate("id => document.querySelector('.tl-font[data-id=\"' + id + '\"]').value", C) == "Bangers-Regular.ttf")
            rep.check("edit: Details size readout follows",
                      br.page.evaluate("id => document.querySelector('.tl-fsb[data-id=\"' + id + '\"]').parentElement.querySelector('.tl-fsv').textContent", C) == "150%")
            rep.check("edit: Details colour buttons follow",
                      br.page.evaluate("id => document.querySelector('.tl-color[data-id=\"' + id + '\"] .tl-clr.on').dataset.c", C) == "white")
            # Cancel must not keep a half-made change (colour is picked before Save)
            br.scroll_to_box(C)
            br.page.mouse.click(*center(br.box_by_id(C)["rect"]))
            br.page.wait_for_selector(".edit-pop")
            br.page.click('.edit-pop .clr-opt[data-c="black"]')
            br.page.click(".edit-pop .epop-cancel")
            body = br.apply()
            rep.check("edit: Cancel discards a colour picked in the popover",
                      body.get("colors", {}).get(C) == "white", str(body.get("colors", {}).get(C)))

            # Empty
            br.scroll_to_box(D)
            d0 = br.box_by_id(D)
            before = fetch_result(*br.active_task())
            br.page.mouse.click(*center(d0["rect"]))
            br.page.wait_for_selector(".edit-pop")
            body = br.apply(click=".edit-pop .epop-erase")
            rep.check("empty: request marks the item erased", D in body.get("erased", []), str(body.get("erased")))
            after = fetch_result(*br.active_task())
            rep.check("empty: lettering is gone and the balloon stays light",
                      dark_count(after, d0["img"], pad=4) < dark_count(before, d0["img"], pad=4),
                      f"dark {dark_count(before, d0['img'], pad=4)} -> {dark_count(after, d0['img'], pad=4)}"
                      " (a RISE means the balloon itself was inpainted away)")
            rep.check("empty: Details row is marked erased",
                      br.page.evaluate("id => !![...document.querySelectorAll('.tl-item.erased .tl-id')].find(e => e.textContent === '#' + id)", D))
            rep.check("empty: emptied item has no box to edit/move", br.box_by_id(D) is None)

        with rep.section("UNDO / REDO"):
            br.set_tool("move")
            br.scroll_to_box(E)
            e0 = br.box_by_id(E)
            cx, cy = center(e0["rect"])
            br.drag(cx, cy, cx + 50, cy + 25)
            e_moved = [e0["img"][0] + 50 * sx, e0["img"][1] + 25 * sy]
            br.page.click("#undoBtn")
            e1 = br.box_by_id(E)
            rep.check("undo: box goes back after a move", near(e1["img"][:2], e0["img"][:2]), f"{e0['img']} vs {e1['img']}")
            br.page.click("#redoBtn")
            e2 = br.box_by_id(E)
            rep.check("redo: box moves again", near(e2["img"][:2], e_moved), f"{e2['img']} vs {e_moved}")
            body = br.apply()
            rep.check("chain move>undo>redo: offset survives to Apply",
                      E in body.get("offsets", {}) and near(body["offsets"][E], [50 * sx, 25 * sy]), str(body.get("offsets", {}).get(E)))
            # keyboard undo
            br.scroll_to_box(E)
            e3 = br.box_by_id(E)
            cx, cy = center(e3["rect"])
            br.drag(cx, cy, cx - 30, cy)
            br.page.keyboard.press("Control+z")
            e4 = br.box_by_id(E)
            rep.check("undo: Ctrl+Z undoes a move", near(e4["img"][:2], e3["img"][:2]), f"{e3['img']} vs {e4['img']}")
            br.page.keyboard.press("Control+y")
            e5 = br.box_by_id(E)
            rep.check("redo: Ctrl+Y redoes it", near(e5["img"][0:1], [e3["img"][0] - 30 * sx]), f"{e5['img']}")
            br.page.keyboard.press("Control+z")
            # A click that does not drag must not eat an undo step
            e6 = br.box_by_id(E)
            br.page.mouse.click(*center(e6["rect"]))
            br.page.click("#undoBtn")
            e7 = br.box_by_id(E)
            rep.check("undo: a click without a drag is not an undo step (Undo still reverts the real move)",
                      near(e7["img"][:2], e0["img"][:2]), f"{e7['img']} vs {e0['img']}")
            br.page.click("#redoBtn")
            # undo of an Edit-popover Save: it must revert the Save and ONLY the
            # Save — the move made just before it has to survive.
            e8 = br.box_by_id(E)
            cx, cy = center(e8["rect"])
            br.drag(cx, cy, cx + 10, cy + 10)
            e_after_drag = br.box_by_id(E)["img"][:2]
            br.set_tool("edit")
            br.page.mouse.click(*center(br.box_by_id(C)["rect"]))
            br.page.wait_for_selector(".edit-pop")
            br.page.fill(".edit-pop .edit-pop-text", "SECOND TEXT")
            br.page.click('.edit-pop .epop-ra[data-a="0"]')
            body = br.apply(click=".edit-pop .epop-save")
            rep.check("edit: second save sends the new text", body.get("edits", {}).get(C) == "SECOND TEXT")
            br.page.click("#undoBtn")
            det = br.details()
            rep.check("undo: reverts an Edit-popover Save (text)", det.get(C) == "HELLO WORLD", str(det.get(C)))
            e9 = br.box_by_id(E)
            rep.check("undo: reverting the Save keeps the move made before it",
                      e9 and near(e9["img"][:2], e_after_drag), f"{e9 and e9['img']} vs {e_after_drag}")
            body = br.apply()
            rep.check("undo: reverted text and tilt are what gets applied",
                      body.get("edits", {}).get(C) == "HELLO WORLD" and body.get("rotations", {}).get(C) == -45,
                      f"{body.get('edits', {}).get(C)} / {body.get('rotations', {}).get(C)}")
            br.page.click("#redoBtn")
            det = br.details()
            rep.check("redo: re-applies the Edit-popover Save", det.get(C) == "SECOND TEXT", str(det.get(C)))
            br.page.click("#undoBtn")
            # Ctrl+Z inside a text field must be the field's own undo, not the page's
            br.tab("details")
            ta = br.page.locator(f'.tl-edit[data-id="{E}"]')
            ta.click()
            ta.type(" EXTRA")
            br.page.evaluate("document.getElementById('editHint').textContent = 'probe'")
            br.page.keyboard.press("Control+z")
            rep.check("undo: Ctrl+Z inside a textarea does not fire the page undo",
                      br.hint() == "probe", br.hint())
            br.page.evaluate("id => { const t = document.querySelector('.tl-edit[data-id=\"' + id + '\"]'); t.value = 'LINE ' + id; t.dispatchEvent(new Event('input')); }", E)
            br.tab("translated")

        with rep.section("TYPE TEXT (added item)"):
            br.set_tool("type-add")
            img_rect = br.image_rect()
            x0, y0 = img_rect[0] + 30, img_rect[1] + 30
            br.drag(x0, y0, x0 + 180, y0 + 70)
            br.page.wait_for_selector(".ime-box", state="visible", timeout=3000)
            br.page.fill(".ime-src", "テスト")
            br.page.fill(".ime-out", "TYPED TEXT")
            br.page.click(".ime-use")
            rep.check("type text: hint confirms the placement", br.hint().startswith("Added!"), br.hint())
            br.set_tool("move")          # the add tools draw no item boxes; Move does
            added = [b for b in br.overlay_boxes() if "added-box" in b["cls"]]
            rep.check("type text: added box appears on the overlay", len(added) == 1, str(len(added)))
            add_bb = added[0]["img"] if added else [0, 0, 1, 1]
            before = fetch_result(*br.active_task())
            body = br.apply()
            rep.check("type text: request carries the added item",
                      len(body.get("added", [])) == 1 and body["added"][0]["translation"] == "TYPED TEXT"
                      and near(body["added"][0]["bbox"], add_bb, 3), str(body.get("added")))
            after = fetch_result(*br.active_task())
            # Any change will do: over dark art the lettering is white with a
            # halo, over a light patch it is black, so a dark-pixel count is
            # not a fair measure here.
            rep.check("type text: lettering lands in the result",
                      region_diff(before, after, add_bb, pad=6) > 1.0, f"diff={region_diff(before, after, add_bb, pad=6):.2f}")
            aid = body["added"][0]["id"] if body.get("added") else "m1"
            # move / resize / edit the added item
            br.set_tool("move")
            m0 = br.added_box()
            cx, cy = center(m0["rect"])
            br.drag(cx, cy, cx + 40, cy + 20)
            body = br.apply()
            rep.check("added: move sends an offset for it",
                      aid in body.get("offsets", {}) and near(body["offsets"][aid], [40 * sx, 20 * sy]), str(body.get("offsets", {}).get(aid)))
            br.set_tool("resize")
            m1 = br.added_box()
            rep.check("added: resize box shows it where it was moved to",
                      near(m1["img"][:2], [add_bb[0] + 40 * sx, add_bb[1] + 20 * sy]), f"{m1['img']} vs {add_bb} + offset")
            hx, hy = br.page.evaluate("""() => { const h = document.querySelector('#moveLayer .resize-box.added-box .rsz-se').getBoundingClientRect();
                return [h.left + h.width / 2, h.top + h.height / 2]; }""")
            br.drag(hx, hy, hx + 30, hy + 10)
            body = br.apply()
            rep.check("added: resize sends a box for it", aid in body.get("boxes", {}), str(body.get("boxes", {}).get(aid)))
            eff = body.get("boxes", {}).get(aid, [0, 0, 0, 0])
            off = body.get("offsets", {}).get(aid, [0, 0])
            rep.check("added: box + offset still puts it where it was moved to, not twice as far",
                      near([eff[0] + off[0], eff[1] + off[1]], [add_bb[0] + 40 * sx, add_bb[1] + 20 * sy]),
                      f"box {eff} + offset {off} vs {[add_bb[0] + 40 * sx, add_bb[1] + 20 * sy]}")
            br.set_tool("edit")
            m2 = br.added_box()
            br.page.mouse.click(*center(m2["rect"]))
            br.page.wait_for_selector(".edit-pop")
            rep.check("added: popover has Delete, no Empty",
                      br.page.query_selector(".edit-pop .epop-erase") is None and br.page.inner_text(".edit-pop .epop-remove").strip() == "Delete")
            br.page.fill(".edit-pop .edit-pop-text", "TYPED TWO")
            body = br.apply(click=".edit-pop .epop-save")
            rep.check("added: edited text is sent", body.get("added", [{}])[0].get("translation") == "TYPED TWO", str(body.get("added")))

        with rep.section("CHAINS"):
            # (a) Resize F, then Edit its text, then Move it
            br.set_tool("resize")
            br.scroll_to_box(F)
            f0 = br.box_by_id(F)
            hx, hy = handle_center(br, F, "se")
            br.drag(hx, hy, hx + 40, hy + 30)
            f_box = [f0["img"][0], f0["img"][1], f0["img"][2] + 40 * sx, f0["img"][3] + 30 * sy]
            br.set_tool("edit")
            fe = br.box_by_id(F)
            rep.check("chain a: Edit-tool box shows the resized geometry before Apply",
                      fe and near(fe["img"], f_box), f"{fe and fe['img']} vs {f_box}")
            br.scroll_to_box(F)
            br.page.mouse.click(*center(br.box_by_id(F)["rect"]))
            br.page.wait_for_selector(".edit-pop")
            br.page.fill(".edit-pop .edit-pop-text", "CHAIN A")
            body = br.apply(click=".edit-pop .epop-save")
            rep.check("chain a: resize survives the Edit save",
                      F in body.get("boxes", {}) and near(body["boxes"][F], f_box), f"{body.get('boxes', {}).get(F)} vs {f_box}")
            br.set_tool("move")
            br.scroll_to_box(F)
            fm = br.box_by_id(F)
            rep.check("chain a: Move-tool box shows the resized geometry",
                      fm and near(fm["img"], f_box), f"{fm and fm['img']} vs {f_box}")
            cx, cy = center(fm["rect"])
            br.drag(cx, cy, cx + 30, cy + 20)
            before = fetch_result(*br.active_task())
            body = br.apply()
            lost = [k for k, ok in [("boxes", F in body.get("boxes", {}) and near(body["boxes"][F], f_box)),
                                    ("edits", body.get("edits", {}).get(F) == "CHAIN A"),
                                    ("offsets", F in body.get("offsets", {}) and near(body["offsets"][F], [30 * sx, 20 * sy]))] if not ok]
            rep.check("chain a: resize > edit > move — box, text and offset all sent", not lost, f"lost: {lost}")
            after = fetch_result(*br.active_task())
            rep.check("chain a: image changed around the item",
                      region_diff(before, after, f_box, pad=int(30 * sx) + 10) > 0.3)

            # (b) Edit font on G, then Resize G
            br.set_tool("edit")
            br.scroll_to_box(G)
            br.page.mouse.click(*center(br.box_by_id(G)["rect"]))
            br.page.wait_for_selector(".edit-pop")
            br.page.select_option(".edit-pop .epop-font", "Anton-Regular.ttf")
            body = br.apply(click=".edit-pop .epop-save")
            rep.check("chain b: font saved", body.get("fonts", {}).get(G) == "Anton-Regular.ttf")
            br.set_tool("resize")
            br.scroll_to_box(G)
            g0 = br.box_by_id(G)
            hx, hy = handle_center(br, G, "se")
            br.drag(hx, hy, hx + 30, hy + 20)
            body = br.apply()
            lost = [k for k, ok in [("fonts", body.get("fonts", {}).get(G) == "Anton-Regular.ttf"),
                                    ("boxes", G in body.get("boxes", {}))] if not ok]
            rep.check("chain b: edit font > resize — font and box both sent", not lost, f"lost: {lost}")

            # (d) per-item font / colour / size on H, switch page and back, apply
            br.set_tool("edit")
            br.scroll_to_box(H)
            br.page.mouse.click(*center(br.box_by_id(H)["rect"]))
            br.page.wait_for_selector(".edit-pop")
            br.page.select_option(".edit-pop .epop-font", "Orbitron.ttf")
            br.page.click('.edit-pop .clr-opt[data-c="black"]')
            br.page.evaluate("""() => { const r = document.querySelector('.edit-pop .epop-fs');
                r.value = 80; r.dispatchEvent(new Event('input', {bubbles: true})); }""")
            body = br.apply(click=".edit-pop .epop-save")
            br.select_page(2)
            br.select_page(1)
            br.page.wait_for_function("document.querySelectorAll('#moveLayer .move-box').length > 0")
            body = br.apply()
            lost = [k for k, ok in [("fonts", body.get("fonts", {}).get(H) == "Orbitron.ttf"),
                                    ("colors", body.get("colors", {}).get(H) == "black"),
                                    ("font_scales", body.get("font_scales", {}).get(H) == 0.8)] if not ok]
            rep.check("chain d: font/colour/size survive a page switch", not lost, f"lost: {lost}")
            rep.check("chain d: Details rows still show them after the switch",
                      br.page.evaluate("id => document.querySelector('.tl-font[data-id=\"' + id + '\"]').value", H) == "Orbitron.ttf"
                      and br.page.evaluate("id => document.querySelector('.tl-color[data-id=\"' + id + '\"] .tl-clr.on').dataset.c", H) == "black")

            # (e) switching tools must not reset anything
            for t in ["cover", "add", "lasso", "resize", "edit", "move", None, "move"]:
                br.set_tool(t)
            body = br.apply()
            expect = {
                "offsets[A]": A in body.get("offsets", {}) and near(body["offsets"][A], exp_off),
                "boxes[B]": B in body.get("boxes", {}),
                "edits[C]": body.get("edits", {}).get(C) == "HELLO WORLD",
                "colors[C]": body.get("colors", {}).get(C) == "white",
                "rotations[C]": body.get("rotations", {}).get(C) == -45,
                "fonts[C]": body.get("fonts", {}).get(C) == "Bangers-Regular.ttf",
                "font_scales[C]": body.get("font_scales", {}).get(C) == 1.5,
                "erased[D]": D in body.get("erased", []),
                "added": any(a.get("translation") == "TYPED TWO" for a in body.get("added", [])),
                "offsets[E]": E in body.get("offsets", {}),
                "boxes[F]": F in body.get("boxes", {}), "edits[F]": body.get("edits", {}).get(F) == "CHAIN A",
                "fonts[G]": body.get("fonts", {}).get(G) == "Anton-Regular.ttf",
                "fonts[H]": body.get("fonts", {}).get(H) == "Orbitron.ttf",
            }
            lost = [k for k, ok in expect.items() if not ok]
            rep.check("chain e: switching tools keeps every in-progress value", not lost, f"lost: {lost}")

            # (f) toggling a tool off/on must not double up handlers
            br.set_tool("move")
            base_layer = br.listener_counts("#moveLayer")
            for _ in range(3):
                br.set_tool(None); br.set_tool("move")
            layer = br.listener_counts("#moveLayer")
            rep.check("chain f: overlay listener count is stable across tool toggles", layer == base_layer, f"{base_layer} -> {layer}")
            br.scroll_to_box(A)
            sel = "#moveLayer .move-box"
            cnt = br.listener_counts(sel)
            rep.check("chain f: one pointerdown handler per box", cnt.get("pointerdown") == 1, str(cnt))
            a3 = br.box_by_id(A)
            cx, cy = center(a3["rect"])
            br.drag(cx, cy, cx + 20, cy + 10)
            br.page.click("#undoBtn")
            a4 = br.box_by_id(A)
            rep.check("chain f: one drag = one undo step", near(a4["img"][:2], a3["img"][:2]), f"{a3['img']} vs {a4['img']}")

        with rep.section("PAGE ISOLATION"):
            page1_boxes = br.overlay_boxes()
            br.select_page(2)
            task2, _ = br.active_task()
            rep.check("isolation: page 2 shows its own result image", task2 == tasks[1] and task2 != task1, f"{task2} vs {tasks}")
            br.page.wait_for_function("document.querySelectorAll('#moveLayer .move-box').length > 0", timeout=5000)
            p2_boxes = br.overlay_boxes()
            rep.check("isolation: no added box from page 1 on page 2", not any("added-box" in b["cls"] for b in p2_boxes))
            same = [b for b in p2_boxes if any(b["id"] == q["id"] and near(b["img"], q["img"], 1) for q in page1_boxes)]
            rep.check("isolation: page 2's boxes are not page 1's boxes", not same, f"{len(same)} identical boxes")
            det2 = br.details()
            rep.check("isolation: page 2 Details has none of page 1's text",
                      not any(v in ("HELLO WORLD", "SECOND TEXT", "CHAIN A") for v in det2.values()) and all(v.startswith("LINE ") for v in det2.values()),
                      str(det2)[:300])
            rep.check("isolation: page 2 has no erased row", br.page.evaluate("document.querySelectorAll('.tl-item.erased').length") == 0)
            rep.check("isolation: page 2 fonts / sizes / colours are untouched",
                      br.page.evaluate("[...document.querySelectorAll('.tl-font')].every(s => s.value === '')")
                      and br.page.evaluate("[...document.querySelectorAll('.tl-fsv')].every(s => s.textContent === '100%')")
                      and br.page.evaluate("[...document.querySelectorAll('.tl-color')].every(g => g.querySelector('.tl-clr.on').dataset.c === 'auto')"))
            n_before = len(br.rerenders)
            body = br.apply()
            rep.check("isolation: page 2's Apply goes to page 2's task", br.rerenders[n_before][0] == task2)
            leaks = [k for k in ("offsets", "boxes", "fonts", "colors", "rotations", "font_scales") if body.get(k)]
            leaks += [k for k in ("erased", "excluded", "added", "covers", "glows", "fits") if body.get(k)]
            rep.check("isolation: page 2's request carries none of page 1's state", not leaks, f"leaked: {leaks} {[body.get(k) for k in leaks]}")
            rep.check("isolation: page 2's edits are its own lines",
                      all(v.startswith("LINE ") for v in body.get("edits", {}).values()), str(body.get("edits"))[:200])
            # back to page 1: everything still there
            br.select_page(1)
            br.page.wait_for_function("document.querySelectorAll('#moveLayer .move-box').length > 0", timeout=5000)
            det1 = br.details()
            rep.check("isolation: page 1 Details still has its edits", det1.get(C) == "HELLO WORLD" and det1.get(F) == "CHAIN A", f"{det1.get(C)} / {det1.get(F)}")
            a5 = br.box_by_id(A)
            rep.check("isolation: page 1 overlay keeps its moved box", a5 and near(a5["img"][:2], [old_bb[0] + exp_off[0], old_bb[1] + exp_off[1]]), f"{a5 and a5['img']}")
            # apply in flight while switching pages: page 2's rows must not be
            # harvested into page 1's items
            n = len(br.rerenders)
            br.page.wait_for_function("!document.getElementById('editApply').disabled", timeout=br.APPLY_TIMEOUT_S * 1000)
            with br.page.expect_response(lambda r: "/api/rerender/" in r.url and r.request.method == "POST", timeout=br.APPLY_TIMEOUT_S * 1000):
                br.page.click("#editApply")
                br.select_page(2)
            time.sleep(0.5)
            br.select_page(1)
            det1 = br.details()
            rep.check("isolation: switching pages during a re-render does not pull page 2's text into page 1",
                      det1.get(C) == "HELLO WORLD" and det1.get(F) == "CHAIN A", f"{det1.get(C)} / {det1.get(F)}")
            body = br.apply()
            rep.check("isolation: page 1's next request still has its own text",
                      body.get("edits", {}).get(C) == "HELLO WORLD" and body.get("edits", {}).get(F) == "CHAIN A")
        rep.check("no browser dialogs popped up", not br.dialogs, str(br.dialogs))
    finally:
        code = rep.finish(br)
        br.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
