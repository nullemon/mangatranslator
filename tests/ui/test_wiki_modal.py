"""The 📋 Paste from wiki… glossary modal, from Settings and from inside the
Train / Manage profile modal."""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import BASE, Check, Session, skip_unless_server

skip_unless_server()
c = Check("wiki glossary modal")

WIKI = ("Gomu Gomu no Pistol (ゴムゴムの銃 (ピストル) Gomu Gomu no Pisutoru?)\n"
        "Rokudo no Tsuji (六道の辻 Rokudō no Tsuji?, literally \"Six Paths Crossroads\")\n"
        "覇王色 = Conqueror's Haki\n")


def api(method, path, body=None):
    req = urllib.request.Request(BASE + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read() or b"{}")


with Session() as s:
    p = s.page
    s.set_value("glossary", "ルフィ = Luffy")

    # ── open from Settings ──
    p.click('.gloss-wiki-btn[data-gloss-target="glossary"]')
    c.ok(s.visible("#wikiModal"), "modal opens")
    c.ok(p.evaluate("document.activeElement && document.activeElement.id") == "wikiInput", "textarea focused")
    c.ok(not s.visible("#wikiResult"), "no stale result on open")

    # extract on empty
    p.click("#wikiExtract")
    c.ok("Paste something" in s.text("#wikiStatus"), "empty paste explains itself")

    p.fill("#wikiInput", WIKI)
    p.click("#wikiExtract")
    p.wait_for_selector("#wikiResult", state="visible", timeout=30000)
    found = s.text("#wikiFound")
    lines = s.value("wikiLines")
    c.ok(found.startswith("Found ") and "3 names" in found, f"{found!r}")
    c.ok("覇王色" in lines and "Conqueror" in lines and "六道の辻" in lines, f"lines: {lines!r}")

    # edit one line, then the count follows
    s.set_value("wikiLines", lines + "\nゾロ = Zoro")
    c.ok("4 names" in s.text("#wikiFound"), "count follows edits")

    p.click("#wikiAdd")
    c.ok(not s.visible("#wikiModal"), "Add closes the modal")
    g = s.value("glossary")
    c.ok(g.startswith("ルフィ = Luffy") and "ゾロ = Zoro" in g and "覇王色 = Conqueror's Haki" in g, f"merged: {g!r}")
    c.eq(p.evaluate("localStorage.getItem('manga_glossary')"), g, "merged glossary persisted")
    c.ok(p.evaluate("document.activeElement.id") == "glossary", "focus lands on the glossary")

    # adding the same again → nothing new, modal stays open
    p.click('.gloss-wiki-btn[data-gloss-target="glossary"]')
    p.fill("#wikiInput", "ゾロ = Zoro")
    p.click("#wikiExtract")
    p.wait_for_selector("#wikiResult", state="visible", timeout=30000)
    p.click("#wikiAdd")
    c.ok(s.visible("#wikiModal") and "Nothing new" in s.text("#wikiNote"), "duplicate names are reported, not re-added")
    c.eq(s.value("glossary"), g, "glossary unchanged by a duplicate add")

    # a changed English for a known name updates in place
    p.fill("#wikiInput", "ゾロ = Roronoa Zoro")
    p.click("#wikiExtract")
    p.wait_for_selector("#wikiResult", state="visible", timeout=30000)
    p.click("#wikiAdd")
    g2 = s.value("glossary")
    c.ok("ゾロ = Roronoa Zoro" in g2 and g2.count("ゾロ") == 1, f"known name updated in place: {g2!r}")

    # ── three ways to close ──
    p.click('.gloss-wiki-btn[data-gloss-target="glossary"]')
    p.keyboard.press("Escape")
    c.ok(not s.visible("#wikiModal"), "Escape closes")
    p.click('.gloss-wiki-btn[data-gloss-target="glossary"]')
    p.click("#wikiClose")
    c.ok(not s.visible("#wikiModal"), "✕ closes")
    p.click('.gloss-wiki-btn[data-gloss-target="glossary"]')
    p.mouse.click(4, 4)                       # backdrop, outside the card
    c.ok(not s.visible("#wikiModal"), "backdrop click closes")
    p.click('.gloss-wiki-btn[data-gloss-target="glossary"]')
    p.click(".wiki-card h3")                  # inside the card: must stay open
    c.ok(s.visible("#wikiModal"), "clicking inside the card keeps it open")
    p.fill("#wikiInput", "ロビン = Robin")       # the Close button sits with the results
    p.click("#wikiExtract")
    p.wait_for_selector("#wikiResult", state="visible", timeout=30000)
    p.click("#wikiCancel")
    c.ok(not s.visible("#wikiModal"), "Close button closes without adding")
    c.ok("ロビン" not in s.value("glossary"), "Close discards the extracted names")

    # ── the copy inside Train / Manage ──
    api("POST", "/api/profile/qa-wiki-test", {"name": "QA Wiki Test", "style_guide": "", "honorifics": "",
                                              "sfx_policy": "", "glossary": [{"term": "ナミ", "translation": "Nami", "notes": ""}]})
    try:
        p.click("#trainBtn")
        p.wait_for_selector("#trainModal", state="visible")
        s.select("trainProfileSel", "qa-wiki-test")
        p.wait_for_selector("#trainResult", state="visible", timeout=15000)
        c.ok("ナミ = Nami" in s.value("trainGloss"), "profile glossary loaded")
        p.click('.gloss-wiki-btn[data-gloss-target="trainGloss"]')
        c.ok(s.visible("#wikiModal"), "modal opens over the Train modal")
        p.fill("#wikiInput", WIKI)
        p.click("#wikiExtract")
        p.wait_for_selector("#wikiResult", state="visible", timeout=30000)
        p.click("#wikiAdd")
        tg = s.value("trainGloss")
        c.ok(tg.startswith("ナミ = Nami") and "覇王色 = Conqueror's Haki" in tg, f"merged into the profile glossary: {tg!r}")
        c.ok("Save profile" in s.text("#trainMeta"), "reminded to save the profile")
        c.ok(s.visible("#trainModal") and not s.visible("#wikiModal"), "Train modal stays open, wiki modal closed")
        c.ok(p.evaluate("document.activeElement.id") == "trainGloss", "focus on the profile glossary")
        # Escape closes only the wiki modal, not Train underneath it
        p.click('.gloss-wiki-btn[data-gloss-target="trainGloss"]')
        p.keyboard.press("Escape")
        c.ok(s.visible("#trainModal") and not s.visible("#wikiModal"), "Escape closes the wiki modal only")
        # the main glossary was not touched by the train-side paste
        c.eq(s.value("glossary"), g2, "settings glossary untouched by the profile paste")
        p.click("#trainClose")
        c.ok(not s.visible("#trainModal"), "Train modal closes")
    finally:
        api("DELETE", "/api/profile/qa-wiki-test")

c.finish(s)
