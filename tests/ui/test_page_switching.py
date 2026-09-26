"""Switching pages during and after processing, and while a re-render is in
flight: the result image, labels and tabs must always be the active page's.
Uses the Clean (remove text) workflow — the only no-key one whose pages can
be re-rendered — so this takes a few minutes on CPU (LaMa per page)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import (Check, Session, page_paths, skip_unless_server, task_of)

skip_unless_server()
c = Check("page switching")
P3, P4, P5 = page_paths("3.jpg", "4.jpg", "5.jpg")

with Session() as s:
    p = s.page
    s.set_workflow("clean")
    s.set_toggle("hdUpscale", False)
    s.upload([P3, P4, P5])
    s.wait_preview()
    s.go()

    # ── during processing: look at a queued page while page 1 runs ──
    p.wait_for_function("() => (document.getElementById('progressMsg').textContent || '') !== ''")
    s.click_chip(2)
    c.ok(s.strip_status()[2]["active"], "third chip active while page 1 processes")
    msg0 = s.text("#progressMsg")
    p.wait_for_timeout(2500)
    msg1 = s.text("#progressMsg")
    c.ok(not s.visible("#pageResult") and s.visible("#pageProcessing"), "queued page shows a waiting state")
    c.ok(msg1 in ("Starting...", "Queued", msg0), f"queued page's message is its own, not page 1's ({msg1!r})")

    # wait for page 1 to finish while page 3 stays selected
    t0 = time.time()
    while time.time() - t0 < 400 and s.strip_status()[0]["status"] != "done":
        p.wait_for_timeout(500)
    c.eq(s.strip_status()[0]["status"], "done", "page 1 finished")
    c.ok(not s.visible("#pageResult"), "finishing page 1 does not hijack the view of page 3")
    c.ok(s.strip_status()[2]["active"], "page 3 still the active chip")

    # switch to the finished page
    s.click_chip(0)
    t1 = task_of(s.strip_status()[0]["src"])
    c.eq(s.active_task(), t1, "result image is page 1's task")
    c.eq(s.text("#compLabelRight"), "Cleaned", "Clean (remove text) pane label")
    c.eq(s.text('.tab[data-tab="translated"]'), "Clean", "Clean (remove text) tab label")
    c.ok(not s.visible('.tab[data-tab="details"]'), "no Details tab for a cleaned page")
    orig_src = p.get_attribute("#origImg", "src") or ""
    c.ok(t1 in orig_src, "original pane is page 1's too")

    s.wait_all_done(3, timeout=600000)
    st = s.strip_status()
    tasks = [task_of(x["src"]) for x in st]
    c.eq(len(set(tasks)), 3, "three distinct tasks")

    # ── after processing: every chip shows its own page ──
    for i in range(3):
        s.click_chip(i)
        c.eq(s.active_task(), tasks[i], f"chip {i + 1} → its own result")
        c.ok(tasks[i] in (p.get_attribute("#origFull", "src") or ""), f"chip {i + 1} → its own original")

    # ── a hidden Details tab must not stay the open panel ──
    p.evaluate("document.querySelector('.tab[data-tab=\"details\"]').click()")
    s.click_chip(1)
    active_panel = p.evaluate("document.querySelector('.tab-panel.active').id")
    c.eq(active_panel, "panel-compare", "switching to a no-text page falls back to Compare when Details was open")
    c.ok(p.evaluate("document.querySelector('.tab.active').dataset.tab") == "compare", "the Compare tab is lit")

    # ── re-render in flight, then switch pages ──
    s.click_chip(0)
    rev_before = st[0]["src"]
    s.record_requests()
    p.evaluate("document.getElementById('applyBtn').click()")      # re-render page 1
    p.wait_for_timeout(150)
    s.click_chip(2)                                                 # switch while it runs
    c.eq(s.active_task(), tasks[2], "page 3 shown immediately after switching")
    p.wait_for_function("() => document.getElementById('applyBtn').textContent !== 'Re-rendering...'",
                        timeout=300000)
    p.wait_for_timeout(300)
    c.eq(s.active_task(), tasks[2], "re-render completion leaves page 3 on screen")
    st2 = s.strip_status()
    c.ok(st2[0]["src"] != rev_before and tasks[0] in st2[0]["src"], "page 1's strip thumbnail was refreshed (rev bumped)")
    c.ok(st2[2]["active"] and not st2[0]["active"], "active chip is still page 3")
    c.ok(any("/api/rerender/" + tasks[0] in u for u, _ in s.requests), "the re-render went to page 1's task")
    c.ok(not s.visible("#errorSection"), "no error shown")

c.finish(s)
