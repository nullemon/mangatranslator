"""Browser test of the Edit popover's "Tilt the whole box".

Turning it on for a line draws its move/resize box turned at the line's
tilt, a drag on the turned box's right-hand handle lengthens it ALONG the
slant (not in screen directions), and Apply & Re-render sends the setting.

Needs a running server (default http://127.0.0.1:8022, see harness.py) and
the test pages in MT_PAGES. Skips cleanly when the server is not up.

Run: /path/to/python tests/ui/test_tilt_box.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import UI, Report, PAGES_DIR, server_up   # noqa: E402


def run(ui, rep):
    ids = ui.run_clean([f"{PAGES_DIR}/4.jpg"], credit="TL: QA Team")
    rep.check("clean run finishes", len(ids) == 1)
    ui.open_editor()

    ui.set_tool("edit")
    ui.page.locator("#moveLayer .move-box").first.click(force=True)
    ui.page.wait_for_timeout(200)
    tb = ui.page.locator(".epop-tb")
    rep.check("popover has 'Tilt the whole box'", tb.count() == 1)
    tb.check()
    ui.page.fill(".epop-rot-num", "20")
    ui.page.locator(".epop-save").click()
    ui.page.wait_for_timeout(200)

    ui.set_tool("resize")
    box = ui.page.locator("#moveLayer .resize-box").first
    tr = box.evaluate("e => e.style.transform")
    rep.check("the resize box is drawn turned", "rotate(20deg)" in tr, tr)

    b0 = box.evaluate("e => [e.style.left, e.style.top, e.style.width, e.style.height]")
    handle = ui.page.locator("#moveLayer .resize-box .rsz-e").first
    handle.scroll_into_view_if_needed()
    ui.page.wait_for_timeout(150)
    h = handle.bounding_box()
    cx, cy = h["x"] + h["width"] / 2, h["y"] + h["height"] / 2
    # drag 60 screen px along the box's own +x direction (20° clockwise)
    ex, ey = cx + 60 * math.cos(math.radians(20)), cy + 60 * math.sin(math.radians(20))
    ui.page.mouse.move(cx, cy)
    ui.page.mouse.down()
    for k in range(1, 9):
        ui.page.mouse.move(cx + (ex - cx) * k / 8, cy + (ey - cy) * k / 8)
    ui.page.mouse.up()
    ui.page.wait_for_timeout(150)
    b1 = box.evaluate("e => [e.style.left, e.style.top, e.style.width, e.style.height]")
    f = lambda v: float(v.rstrip("%"))              # noqa: E731
    grew_w = f(b1[2]) - f(b0[2])
    grew_h = f(b1[3]) - f(b0[3])
    rep.check("dragging the turned box's side handle lengthens it along the slant",
              grew_w > 0.5 and abs(grew_h) < 0.2, f"width +{grew_w:.2f}% height {grew_h:+.2f}%")

    body = ui.apply()
    tbs = body.get("tilt_boxes") or []
    rots = body.get("rotations") or {}
    rep.check("Apply sends the tilted box and its angle",
              len(tbs) == 1 and rots.get(str(tbs[0])) == 20, f"{tbs} {rots}")
    errs, _cons = ui.drain_errors()
    rep.check("no JS exceptions", not errs, str(errs))


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
    return 0 if rep.summary() else 1


if __name__ == "__main__":
    sys.exit(main())
