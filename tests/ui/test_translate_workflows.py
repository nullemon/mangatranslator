"""The translate workflows end to end, in a real browser, against the real
pipeline — with the translation APIs replaced by the stub servers.

Start the stubs and the app (the app must point at them):

    python tests/ui/stub_llm.py 8131 &
    python tests/ui/stub_gemini.py 8132 &
    ANTHROPIC_BASE_URL=http://127.0.0.1:8131 GEMINI_BASE_URL=http://127.0.0.1:8132 \\
        PORT=8031 python app.py &

then run all sections, or name some:

    MT_URL=http://127.0.0.1:8031 python tests/ui/test_translate_workflows.py
    MT_URL=http://127.0.0.1:8031 python tests/ui/test_translate_workflows.py user editor

Sections:
  user      the user's own settings (Gemini Pro, natural, Japanese -> English,
            title, missing font, Font per mood, Untouched finish, watermark +
            credit, Maximum Quality) on two pages, Scan -> Translate
  editor    the Details list on that page: edit, per-line font, colour, size,
            text-size slider, Skip / un-Skip, ⌫ / un-⌫, reading-order
            re-translate, transcript, Download / Download All
  combo     Claude, Raw -> Translate, every other setting flipped (literal,
            Spanish, keep case, restore finish, Smart Detection, SFX,
            compress, keep site marks, replace watermark, tiled mark, GPU cap,
            SBS, style box, wiki glossary, a trained profile, an uploaded
            font) and a setting changed between two pages of the batch
  sequence  everything switched back off, cards switched, a reload: no value
            from the previous runs may leak into the next request or prompt
  errors    Offline engine, a bad key, quota, an empty reply, a retired model
            — each ends with a clear message, never a stuck page
  webtoon   two slices in Webtoon mode -> one long page, credit included
  upscale   HD Upscale and One-by-one reach the pipeline (HTTP, small crop)
  pieces    SBS cut regions and a webtoon strip over HTTP: credit placed, and
            a key that fails every piece fails the page with the reason

Each translate takes a minute or more on CPU (LaMa); the whole file runs in
roughly 15-25 minutes. Skips cleanly when the app or a stub is not running.
"""
import io
import json
import os
import re
import sys
import time
import urllib.request
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import (BASE, Check, Session, decode, fetch,  # noqa: E402
                      skip_unless_server, task_of)

GEMINI = os.environ.get("MT_GEMINI_STUB", "http://127.0.0.1:8132")
CLAUDE = os.environ.get("MT_CLAUDE_STUB", "http://127.0.0.1:8131")
CH = os.environ.get("MT_CHAPTER", "/tmp/claude-0/ch/raw")
OPD = os.environ.get("MT_PAGES", "/tmp/claude-0/op")
FONTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "fonts")

skip_unless_server()


def _up(url):
    try:
        with urllib.request.urlopen(url + "/__log", timeout=3):
            return True
    except Exception:
        return False


for _u in (GEMINI, CLAUDE):
    if not _up(_u):
        print(f"SKIP: no stub at {_u} (start tests/ui/stub_gemini.py / stub_llm.py)")
        sys.exit(0)

P73, P74 = os.path.join(CH, "p-0073.jpg"), os.path.join(CH, "p-0074.jpg")
P3, P4 = os.path.join(OPD, "3.jpg"), os.path.join(OPD, "4.jpg")
for _p in (P73, P74, P3, P4):
    if not os.path.exists(_p):
        print(f"SKIP: test page {_p} is missing")
        sys.exit(0)

SECTIONS = sys.argv[1:] or ["user", "editor", "combo", "sequence", "errors", "webtoon",
                            "upscale", "pieces"]
c = Check("translate workflows")
STATE = {}


# ── helpers ────────────────────────────────────────────────────────────
def stub_reset(url):
    urllib.request.urlopen(urllib.request.Request(url + "/__reset", data=b"{}",
                                                  method="POST"), timeout=5)


def stub_log(url):
    with urllib.request.urlopen(url + "/__log", timeout=5) as r:
        return json.loads(r.read())


def status(tid):
    with urllib.request.urlopen(f"{BASE}/api/status/{tid}", timeout=10) as r:
        return json.loads(r.read())


def translate_reqs(s):
    return [f for u, f in s.requests if u.endswith("/api/translate")]


def all_task_ids(s):
    return [task_of(x["src"]) for x in s.strip_status()]


def run(s, paths, timeout=900000, allow_error=False):
    """Upload, press Go, wait until every page is done (or errored)."""
    if s.visible("#newBtn"):
        s.page.click("#newBtn")
    s.upload(paths)
    s.wait_preview()
    s.record_requests()
    s.go()
    return s.wait_all_done(len(paths), timeout=timeout, allow_error=allow_error)


def page_task(s, i=0, n=1):
    if n > 1:
        s.click_chip(i)
    s.page.wait_for_function("""() => { const i = document.getElementById('transFull');
        return i.complete && i.naturalWidth > 0; }""", timeout=60000)
    return s.active_task()          # the task id


def diff_box(a, b, thresh=40):
    """Bounding box (x0, y0, x1, y1) of the pixels that differ, or None."""
    d = np.any(np.abs(a.astype(int) - b.astype(int)) > thresh, axis=2)
    ys, xs = np.nonzero(d)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def region_change(a, b, bbox, pad=0):
    x, y, w, h = [int(v) for v in bbox]
    H, W = a.shape[:2]
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(np.abs(a[y0:y1, x0:x1].astype(int) - b[y0:y1, x0:x1].astype(int)).mean())


def untouched_mask(shape, items, extra=(), pad=40):
    """True where no item (and no extra box) could have touched the page."""
    H, W = shape[:2]
    m = np.ones((H, W), bool)
    for it in items:
        x, y, w, h = [int(v) for v in it["bbox"]]
        m[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad] = False
    for x0, y0, x1, y1 in extra:
        m[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = False
    return m


def wait_apply(s, timeout=400000):
    p = s.page
    p.wait_for_function(
        "() => document.getElementById('applyBtn').textContent !== 'Re-rendering...'",
        timeout=timeout)
    p.wait_for_timeout(400)


def set_wm(s, text, style="clean", place="br", size="m", opacity="40", tile_too=False, credit=""):
    s.set_value("watermark", text)
    s.select("wmStyle", style)
    s.select("wmPlace", place)
    s.select("wmSize", size)
    s.set_value("wmOpacity", opacity, "change")
    s.set_toggle("wmTileToo", tile_too)
    s.set_value("credit", credit)


def set_options(s, **kw):
    defaults = dict(smartMode=False, oneByOne=False, webtoonMode=False,
                    translateSfx=False, maxQuality=False, hdUpscale=False,
                    compressOut=False, removeWatermark=True,
                    replaceWatermark=False, sbsMode=False, styleFonts=False)
    defaults.update(kw)
    for k, v in defaults.items():
        s.set_toggle(k, v)


# ── 1. the user's own settings ─────────────────────────────────────────
def section_user(s):
    p = s.page
    print("\n== user: Gemini Pro, the user's exact settings, 2 pages ==")
    stub_reset(GEMINI)
    # The saved font from the user's own machine is not installed here: it
    # must fall back to Auto-detect, not break the page or the request.
    p.evaluate("localStorage.setItem('manga_font', 'animeace2_reg.ttf')")
    s.reload()
    p.wait_for_function("() => document.querySelectorAll('#fontSelect option').length > 1",
                        timeout=15000)
    c.eq(s.value("fontSelect"), "", "a saved font that isn't installed falls back to Auto-detect")

    s.set_workflow("scan-translate")
    s.select("engine", "gemini")
    s.set_value("apiKey", "stub-key")
    s.select("model", "gemini-pro-latest")
    s.select("transStyle", "natural")
    s.select("sourceLang", "Japanese")
    s.select("targetLang", "English")
    s.set_value("mangaTitle", "One PIECE")
    s.set_value("stylePrompt", "")
    s.set_value("glossary", "")
    s.select("profileSelect", "")
    s.select("textCase", "upper")
    s.set_toggle("styleFonts", True)
    s.select("fontVariety", "pro")
    s.select("pageFinish", "off")
    set_wm(s, "read One Piece on kaisuki.com/g/onepiece", "clean", "br", "m", "40",
           credit="Translations by wonpe4ce")
    set_options(s, maxQuality=True, styleFonts=True)
    s.select("gpuCap", "100")

    run(s, [P73, P74])
    reqs = translate_reqs(s)
    c.eq(len(reqs), 2, "two translate requests")
    want = {"provider": "gemini", "model": "gemini-pro-latest", "source_lang": "Japanese",
            "target_lang": "English", "text_case": "upper", "finish": "off",
            "style_fonts": "pro", "max_quality": "true", "smart_mode": "false",
            "one_by_one": "false", "webtoon": "false", "translate_sfx": "false",
            "compress": "false", "remove_watermark": "true", "replace_watermark": "false",
            "upscale": "false", "enhance": "false", "font": "", "gpu_cap": "100",
            "watermark": "read One Piece on kaisuki.com/g/onepiece", "wm_style": "clean",
            "wm_place": "br", "wm_size": "m", "wm_opacity": "40",
            "credit": "Translations by wonpe4ce", "api_key": "stub-key"}
    for i, r in enumerate(reqs):
        bad = {k: (r.get(k), v) for k, v in want.items() if r.get(k) != v}
        c.ok(not bad, f"page {i + 1} request carries every setting {bad}")
        st = r.get("style_prompt", "")
        c.ok('SERIES: this page is from "One PIECE"' in st, f"page {i + 1}: manga title sent")
        c.ok("profile" not in r, f"page {i + 1}: no profile sent when None is picked")

    log = stub_log(GEMINI)
    c.ok(log and all(e["model"] == "gemini-pro-latest" for e in log),
         "every Gemini call used the picked model")
    thinks = [(e["config"] or {}).get("thinkingConfig") for e in log]
    c.eq(thinks[0], {"thinkingBudget": 0}, "the ladder starts at thinking off")
    c.eq(thinks[1], {"thinkingLevel": "low"}, "Gemini Pro's refusal steps to thinkingLevel")
    c.ok(all(t == {"thinkingLevel": "low"} for t in thinks[2:]),
         f"the accepted rung is remembered for the rest of the run {thinks[2:]}")
    prompts = [e["prompt"] for e in log if e["kind"] in ("regions", "texts", "free_text")]
    c.ok(prompts and all('"One PIECE"' in x for x in prompts), "every prompt carries the title")
    c.ok(all("Japanese" in x and "English" in x for x in prompts),
         "prompts ask for Japanese -> English")
    c.ok(all("right-to-left" in x for x in prompts if "numbered regions" in x),
         "manga (not webtoon) reading order")

    tids = []
    for i, path in enumerate((P73, P74)):
        tid = page_task(s, i, 2)
        tids.append(tid)
        t = status(tid)
        res = t["result"]
        items = res["items"]
        c.ok(t["max_quality"] and t["finish"] == "off" and t["style_fonts"] == "pro"
             and t["font_path"] is None, f"page {i + 1}: server kept the settings")
        c.ok(res["output_path"].endswith(".png"),
             f"page {i + 1}: Maximum Quality saves the page lossless ({res['output_path']})")
        c.ok(t.get("cost_note", "").startswith("≈ $"), f"page {i + 1}: cost recorded {t.get('cost_note')!r}")
        credit = [it for it in items if it.get("type") == "credit"]
        c.ok(len(credit) == 1 and credit[0]["translation"] == "Translations by wonpe4ce"
             and credit[0].get("credit"), f"page {i + 1}: the credit is on the page")
        lines = [it for it in items if it.get("type") != "credit"]
        c.ok(lines and all(re.fullmatch(r"LINE \d+", it["translation"]) for it in lines),
             f"page {i + 1}: {len(lines)} lines translated")
        c.ok(all(it.get("tone") == "dialogue" for it in lines),
             f"page {i + 1}: the translator's tone reaches the items (Font per mood)")

        out = decode(fetch(f"/api/result/{tid}"))
        clean = decode(fetch(f"/api/result/{tid}?watermark=0"))
        orig = decode(fetch(f"/api/original/{tid}"))
        c.eq(out.shape, orig.shape, f"page {i + 1}: full resolution kept")
        H, W = out.shape[:2]
        wm = diff_box(out, clean)
        c.ok(wm is not None, f"page {i + 1}: watermark stamped")
        if wm:
            x0, y0, x1, y1 = wm
            c.ok(x0 > 2 and y0 > 2 and x1 < W - 3 and y1 < H - 3,
                 f"page {i + 1}: the whole watermark is on the page {wm}")
            c.ok(x1 - x0 > W * 0.45, f"page {i + 1}: the full URL is there, not cut short {wm}")
            STATE.setdefault("wm_boxes", []).append(wm)
        cx, cy, cw, ch = credit[0]["bbox"]
        c.ok(region_change(clean, orig, credit[0]["bbox"]) > 1.0,
             f"page {i + 1}: credit drawn at {credit[0]['bbox']}")
        # Untouched finish: away from the lettering, the watermark and the
        # credit, the page is the original pixels.
        m = untouched_mask(out.shape, items, [wm] if wm else [])
        dev = np.abs(clean.astype(int) - orig.astype(int)).max(axis=2)[m]
        c.ok(dev.size > 0.3 * H * W and float(dev.mean()) < 0.5,
             f"page {i + 1}: 'Untouched' leaves the rest of the page as it was "
             f"(mean diff {float(dev.mean()):.3f} over {dev.size / (H * W):.0%})")

    # cost readout on screen for the active page
    s.click_chip(0)
    c.ok(s.visible("#pageCost") and "≈ $" in s.text("#pageCost"),
         f"cost shown under the result: {s.text('#pageCost')!r}")
    STATE["user_tids"] = tids


# ── 2. the Details list and exports on the user's page ─────────────────
def section_editor(s):
    p = s.page
    print("\n== editor: Details list, reading order, transcript, downloads ==")
    if "user_tids" not in STATE:
        print("  (runs after 'user')")
        return
    s.click_chip(0)
    tid = s.active_task()
    t = status(tid)
    items = [it for it in t["result"]["items"] if it.get("type") != "credit"]
    ids = [str(it["id"]) for it in items]
    by = {str(it["id"]): it for it in items}
    before = decode(fetch(f"/api/result/{tid}?watermark=0"))
    base = decode(fetch(f"/api/original/{tid}"))
    p.click('.tab[data-tab="details"]')
    p.wait_for_selector("#panel-details.active")
    rows = p.evaluate("document.querySelectorAll('#translationsList .tl-item').length")
    c.eq(rows, len(t["result"]["items"]), "one Details row per item (credit included)")

    e_id, f_id, w_id, z_id, k_id, x_id = ids[:6]
    # edit text
    p.fill(f'.tl-edit[data-id="{e_id}"]', "HELLO QA")
    # per-line font
    p.select_option(f'.tl-font[data-id="{f_id}"]', "Bangers-Regular.ttf")
    # colour: white
    p.click(f'.tl-color[data-id="{w_id}"] .tl-clr[data-c="white"]')
    # size A+
    p.click(f'.tl-fsb[data-id="{z_id}"][data-d="1"]')
    # skip one, erase one
    p.click(f'.tl-x[data-id="{k_id}"]')
    p.click(f'.tl-erase[data-id="{x_id}"]')
    s.set_value("fontScale", "1.2")
    s.record_requests()
    p.click("#applyBtn")
    wait_apply(s)
    rr = [f for u, f in s.requests if "/api/rerender/" in u]
    c.eq(len(rr), 1, "one re-render request")
    body = rr[0] if rr else {}
    c.eq(body.get("edits", {}).get(e_id), "HELLO QA", "edited text sent")
    c.eq(body.get("fonts"), {f_id: "Bangers-Regular.ttf"}, "per-line font sent")
    c.eq(body.get("colors", {}).get(w_id), "white", "per-line colour sent")
    c.eq(body.get("font_scales", {}).get(z_id), 1.1, "per-line size sent")
    c.eq(body.get("excluded"), [k_id], "skip sent")
    c.eq(body.get("erased"), [x_id], "erase sent")
    c.eq(body.get("font_scale"), 1.2, "text-size slider sent")
    after = decode(fetch(f"/api/result/{tid}?watermark=0"))
    for rid, what in ((e_id, "edited line"), (f_id, "font pick"), (w_id, "white text"),
                      (z_id, "bigger line")):
        c.ok(region_change(after, before, by[rid]["bbox"], 6) > 0.5,
             f"{what} re-rendered in its box (#{rid})")
    c.ok(region_change(after, base, by[k_id]["bbox"]) < region_change(before, base, by[k_id]["bbox"]),
         f"skipped bubble #{k_id} shows the original again")
    t2 = status(tid)
    stored = {str(it["id"]): it for it in t2["result"]["items"]}
    c.eq(stored[k_id]["translation"], by[k_id]["translation"],
         "a skipped line keeps its translation on the server")
    c.eq(stored[x_id]["translation"], by[x_id]["translation"],
         "an erased line keeps its translation on the server")
    c.eq(stored[e_id].get("tone"), "dialogue", "tone survives a re-render")
    c.ok(any(it.get("credit") for it in t2["result"]["items"]), "credit flag survives a re-render")

    # un-skip / un-erase: the text comes back
    p.click(f'.tl-x[data-id="{k_id}"]')
    p.click(f'.tl-erase[data-id="{x_id}"]')
    c.eq(p.input_value(f'.tl-edit[data-id="{k_id}"]'), by[k_id]["translation"],
         "un-skipping shows the line's text again")
    c.eq(p.input_value(f'.tl-edit[data-id="{x_id}"]'), by[x_id]["translation"],
         "un-erasing shows the line's text again")
    s.record_requests()
    p.click("#applyBtn")
    wait_apply(s)
    body = [f for u, f in s.requests if "/api/rerender/" in u][-1]
    c.ok(body.get("excluded") == [] and body.get("erased") == [], "nothing skipped or erased now")
    c.eq(body["edits"].get(k_id), by[k_id]["translation"], "un-skipped line re-sent with its text")
    again = decode(fetch(f"/api/result/{tid}?watermark=0"))
    c.ok(region_change(again, base, by[k_id]["bbox"]) > 1.0, f"un-skipped line #{k_id} typeset again")
    c.ok(region_change(again, after, by[x_id]["bbox"]) > 1.0, f"un-erased line #{x_id} typeset again")
    wm = diff_box(decode(fetch(f"/api/result/{tid}")), again)
    c.ok(wm is not None, "watermark re-stamped after a re-render")

    # transcript
    s.context.grant_permissions(["clipboard-read", "clipboard-write"], origin=BASE)
    p.click("#copyTranscript")
    p.wait_for_timeout(300)
    clip = p.evaluate("navigator.clipboard.readText()")
    c.ok("HELLO QA" in clip and "LINE" in clip, "transcript copied with the edited line")
    name, data = s.download("#downloadTranscript")
    txt = data.decode("utf-8")
    c.ok(name.endswith("_translation.txt") and "HELLO QA" in txt, f"transcript .txt ({name})")
    # a skipped line is not in the transcript
    p.click(f'.tl-x[data-id="{k_id}"]')
    p.click("#copyTranscript")
    p.wait_for_timeout(300)
    clip2 = p.evaluate("navigator.clipboard.readText()")
    c.ok(clip2.count("→") == clip.count("→") - 1, "a skipped line drops out of the transcript")
    p.click(f'.tl-x[data-id="{k_id}"]')

    # reading order → re-translate
    stub_reset(GEMINI)
    s.record_requests()
    p.click("#orderBtn")
    p.wait_for_selector(".ro-back")
    p.click("#roGo")
    p.wait_for_selector(".ro-back", state="detached", timeout=120000)
    wait_apply(s)
    ro = [f for u, f in s.requests if "/api/retranslate-ordered/" in u]
    c.ok(ro and ro[0].get("provider") == "gemini" and ro[0].get("model") == "gemini-pro-latest",
         "reading-order re-translate uses the picked engine and model")
    c.ok(ro and "One PIECE" in ro[0].get("style_prompt", ""), "and the manga title")
    kinds = [e["kind"] for e in stub_log(GEMINI)]
    c.eq(kinds, ["texts"], "one ordered text-translate call")

    # downloads
    name, data = s.download("#downloadBtn")
    c.ok(name.endswith(".png") and np.array_equal(decode(data), decode(fetch(f"/api/result/{tid}"))),
         f"Download This Page = the stamped page ({name})")
    name, data = s.download("#downloadCleanBtn")
    c.ok("(no watermark)" in name and np.array_equal(
        decode(data), decode(fetch(f"/api/result/{tid}?watermark=0"))), f"No-watermark download ({name})")
    name, data = s.download("#zipBtn", timeout=300000)
    zf = zipfile.ZipFile(io.BytesIO(data))
    names = sorted(zf.namelist())
    c.eq(len(names), 2, f"Download All holds both pages {names}")
    for n_, tid_ in zip(names, STATE["user_tids"]):
        c.ok(np.array_equal(decode(zf.read(n_)), decode(fetch(f"/api/result/{tid_}"))),
             f"{n_} is that page's stamped result")


# ── 3. every other setting, Claude, and a change mid-batch ─────────────
def section_combo(s):
    p = s.page
    print("\n== combo: Claude, Raw -> Translate, everything else flipped ==")
    stub_reset(CLAUDE)
    s.set_workflow("raw-translate")
    s.select("engine", "claude")
    s.set_value("apiKey", "sk-ant-stub")
    s.select("model", "claude-opus-4-6")
    s.select("transStyle", "literal")
    s.select("sourceLang", "Japanese")
    s.select("targetLang", "Spanish")
    s.set_value("mangaTitle", "")
    s.set_value("stylePrompt", "Keep honorifics QA-STYLE STUB-TONES")
    # glossary through the wiki paste
    s.set_value("glossary", "")
    p.click('.gloss-wiki-btn[data-gloss-target="glossary"]')
    p.fill("#wikiInput", "ゾロ = Zoro\nナミ = Nami")
    p.click("#wikiExtract")
    p.wait_for_selector("#wikiAdd:not([disabled])")
    p.click("#wikiAdd")
    c.ok("ゾロ = Zoro" in s.value("glossary"), "wiki names merged into the glossary")

    # a trained profile: learn from two 'translated' pages via the stub
    p.click("#trainBtn")
    p.wait_for_selector("#trainModal", state="visible")
    p.fill("#trainName", "QA Series")
    p.set_input_files("#trainFiles", [P3, P4])
    p.click("#trainLearn")
    p.wait_for_function("() => /Learned from|failed/.test(document.getElementById('trainStatus').textContent)",
                        timeout=120000)
    c.ok("Learned from 2 of 2" in s.text("#trainStatus"), f"profile learned: {s.text('#trainStatus')!r}")
    c.ok("ルフィ = Luffy" in s.value("trainGloss"), "learned glossary shown for review")
    p.fill("#trainGloss", s.value("trainGloss") + "\nサンジ = Sanji")
    p.click("#trainSave")
    p.wait_for_function("() => /Saved/.test(document.getElementById('trainStatus').textContent)")
    p.click("#trainClose")
    c.eq(s.value("profileSelect"), "qa-series", "saved profile selected for translation")
    prof = json.loads(fetch("/api/profile/qa-series"))
    c.ok(any(g["translation"] == "Sanji" for g in prof["glossary"]), "reviewed edit saved")

    # a font added with "+ Add"
    src = os.path.join(FONTS, "ComicNeue-Bold.ttf")
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_inputs")
    os.makedirs(tmp, exist_ok=True)
    qa_font = os.path.join(tmp, "QAUploadFont.ttf")
    with open(src, "rb") as a, open(qa_font, "wb") as b:
        b.write(a.read())
    p.set_input_files("#fontUpload", qa_font)
    p.wait_for_function("() => document.getElementById('fontSelect').value === 'QAUploadFont.ttf'",
                        timeout=15000)
    c.ok(True, "uploaded font selected")

    s.select("textCase", "keep")
    s.set_toggle("styleFonts", True)
    s.select("fontVariety", "expressive")
    s.select("pageFinish", "restore")
    set_wm(s, "@qa-combo", "tile", "tl", "s", "60", tile_too=True, credit="TL: QA")
    set_options(s, smartMode=False, sbsMode=True, translateSfx=True, compressOut=True,
                 removeWatermark=False, replaceWatermark=True, styleFonts=True)
    s.select("gpuCap", "60")

    # page 2's request is built when page 2 starts; change settings while
    # page 1 runs and page 2 must carry the new values, page 1 the old.
    if s.visible("#newBtn"):
        p.click("#newBtn")
    s.upload([P3, P4])
    s.wait_preview()
    s.record_requests()
    s.go()
    p.wait_for_function("() => (window.__mtReqs || []).filter(r => r[0].endsWith('/api/translate')).length >= 1")
    s.select("gpuCap", "80")
    s.select("wmPlace", "br")
    s.wait_all_done(2, timeout=900000)
    reqs = translate_reqs(s)
    c.eq(len(reqs), 2, "two translate requests")
    r1, r2 = reqs[0], reqs[1]
    want = {"provider": "claude", "model": "claude-opus-4-6", "target_lang": "Spanish",
            "text_case": "keep", "finish": "restore", "style_fonts": "expressive",
            "smart_mode": "true", "translate_sfx": "true", "compress": "true",
            "remove_watermark": "false", "replace_watermark": "true",
            "font": "QAUploadFont.ttf", "wm_style": "tile", "wm_size": "s",
            "wm_opacity": "60", "credit": "TL: QA", "profile": "qa-series",
            "max_quality": "false", "enhance": "false"}
    bad = {k: (r1.get(k), v) for k, v in want.items() if r1.get(k) != v}
    c.ok(not bad, f"page 1 request carries every setting {bad}")
    c.ok(r1.get("gpu_cap") == "60" and r1.get("wm_place") == "tl", "page 1: the values at its start")
    c.ok(r2.get("gpu_cap") == "80" and r2.get("wm_place") == "br",
         "page 2: the values changed mid-batch, nothing stale")
    st = r1.get("style_prompt", "")
    c.ok("QA-STYLE" in st and "ゾロ = Zoro" in st and "faithfully" in st,
         "style box, glossary and the Literal approach all sent")

    log = stub_log(CLAUDE)
    c.ok(log and all(e["model"] == "claude-opus-4-6" for e in log), "Claude got the picked model")
    kinds = [e["kind"] for e in log]
    c.ok("smart" in kinds, f"SBS page turns on the AI-vision (smart) pass {kinds}")
    pr = [e["prompt"] for e in log if e["kind"] in ("smart", "regions", "texts", "free_text")]
    c.ok(pr and all("SERIES STYLE PROFILE" in x and "ルフィ → Luffy" in x and "サンジ → Sanji" in x
                    for x in pr), "the trained profile is in every prompt")
    c.ok(all("Spanish" in x for x in pr), "target language Spanish in every prompt")
    c.ok(all("DO translate them" in x for x in pr if x.startswith("You are an expert manga page analyzer")
             and "Carefully examine" in x), "SFX: translate them")

    for i in range(2):
        tid = page_task(s, i, 2)
        t = status(tid)
        res = t["result"]
        c.ok(t["font_path"] == "fonts/QAUploadFont.ttf", f"page {i + 1}: uploaded font used")
        c.ok(res["output_path"].endswith(".jpg"), f"page {i + 1}: Compress Output gives a JPEG")
        c.ok(t["text_case"] == "keep" and t["style_fonts"] == "expressive" and t["finish"] == "restore",
             f"page {i + 1}: server kept case / mood fonts / finish")
        c.ok(any(it.get("type") == "credit" and it["translation"] == "TL: QA" for it in res["items"]),
             f"page {i + 1}: credit on the page")
        tones = {it.get("tone") for it in res["items"] if it.get("type") != "credit"}
        c.ok({"shout", "dialogue"} <= tones,
             f"page {i + 1}: Smart Detection passes the model's voices on for the mood fonts {tones}")
        out = decode(fetch(f"/api/result/{tid}"))
        clean = decode(fetch(f"/api/result/{tid}?watermark=0"))
        H, W = out.shape[:2]
        d = np.any(np.abs(out.astype(int) - clean.astype(int)) > 12, axis=2)
        quads = [d[:H // 2, :W // 2].mean(), d[:H // 2, W // 2:].mean(),
                 d[H // 2:, :W // 2].mean(), d[H // 2:, W // 2:].mean()]
        c.ok(min(quads) > 0.002, f"page {i + 1}: tiled mark covers every quarter {np.round(quads, 4)}")
        orig = decode(fetch(f"/api/original/{tid}"))
        m = untouched_mask(out.shape, res["items"])
        dev = np.abs(clean.astype(int) - orig.astype(int)).max(axis=2)[m]
        c.ok(float(dev.mean()) > 3, f"page {i + 1}: 'Restore' finish changed the page (mean {float(dev.mean()):.1f})")
    STATE["combo_done"] = True


# ── 4. back to plain: nothing stale may leak ───────────────────────────
def section_sequence(s):
    p = s.page
    print("\n== sequence: settings off again, cards switched, reload ==")
    stub_reset(GEMINI)
    s.select("engine", "gemini")
    s.set_value("apiKey", "stub-key")
    s.select("model", "gemini-flash-latest")
    s.select("targetLang", "English")
    s.select("transStyle", "natural")
    s.set_value("mangaTitle", "")
    s.set_value("stylePrompt", "")
    s.set_value("glossary", "")
    s.select("profileSelect", "")
    s.select("fontSelect", "")
    s.select("textCase", "upper")
    s.select("pageFinish", "clean")
    set_wm(s, "", credit="")
    set_options(s)
    s.select("gpuCap", "100")
    # switch cards around, then reload: the plain settings must stick
    for wf in ("clean", "raw-scan-translate", "watermark-only", "scan-translate"):
        s.set_workflow(wf)
    s.reload()
    c.eq(p.evaluate("document.querySelector('.wf-card.active').dataset.wf"), "scan-translate",
         "workflow survives the reload")
    c.eq(s.value("profileSelect"), "", "profile stays None after reload")
    c.eq(s.value("watermark"), "", "watermark stays empty after reload")
    c.ok(not s.checked("sbsMode") and not s.checked("translateSfx") and not s.checked("compressOut"),
         "toggles stay off after reload")
    run(s, [P3])
    r = translate_reqs(s)[0]
    for k in ("watermark", "credit", "profile", "style_prompt", "wm_style", "cut_regions"):
        c.ok(k not in r, f"no stale {k} in the request ({r.get(k)!r})")
    c.ok(r["smart_mode"] == "false" and r["translate_sfx"] == "false" and r["compress"] == "false"
         and r["replace_watermark"] == "false" and r["remove_watermark"] == "true"
         and r["style_fonts"] == "false" and r["font"] == "" and r["gpu_cap"] == "100"
         and r["text_case"] == "upper" and r["finish"] == "clean" and r["target_lang"] == "English",
         f"every option back to plain {r}")
    log = stub_log(GEMINI)
    pr = " ".join(e["prompt"] for e in log)
    for leak in ("SERIES STYLE PROFILE", "ゾロ", "QA-STYLE", "Spanish", "faithfully", "One PIECE"):
        c.ok(leak not in pr, f"nothing from earlier runs in the prompts ({leak})")
    c.ok(all(e["model"] == "gemini-flash-latest" for e in log), "flash model this time")
    c.eq((log[0]["config"] or {}).get("thinkingConfig"), {"thinkingBudget": 0},
         "Flash takes thinking off on the first try")
    tid = s.active_task()
    t = status(tid)
    c.ok(not os.path.exists(os.path.join("output", "x")) and
         np.array_equal(decode(fetch(f"/api/result/{tid}")), decode(fetch(f"/api/result/{tid}?watermark=0"))),
         "no watermark stamped")
    c.ok(not any(it.get("type") == "credit" for it in t["result"]["items"]), "no credit on the page")
    c.eq(t["font_path"], None, "no stale font")


# ── 5. errors end with a message, never a stuck page ───────────────────
def section_errors(s):
    p = s.page
    print("\n== errors ==")
    set_options(s)
    set_wm(s, "", credit="")

    def fail_case(label, engine, key, model, expect):
        s.select("engine", engine)
        if engine != "local":
            s.set_value("apiKey", key)
            s.select("model", model)
        t0 = time.time()
        res = run(s, [P3], timeout=300000, allow_error=True)
        msg = s.text("#progressMsg")
        c.eq(res, "error", f"{label}: the page ends in an error")
        c.ok(re.search(expect, msg), f"{label}: clear message ({msg!r})")
        c.ok("Traceback" not in msg and "list index" not in msg, f"{label}: no traceback text")
        c.ok(s.visible("#retryPageBtn"), f"{label}: retry offered")
        c.ok(time.time() - t0 < 240, f"{label}: fails fast ({time.time() - t0:.0f}s)")

    s.set_workflow("raw-translate")
    fail_case("offline", "local", "", "", r"Offline engine can't read this page.*manga-ocr.*Gemini or Claude")
    c.ok(not s.visible("#apiKey"), "offline hides the key field")
    # style training needs an AI model: say so, not "add your key"
    p.click("#trainBtn")
    p.fill("#trainName", "QA Offline")
    p.set_input_files("#trainFiles", [P3])
    p.click("#trainLearn")
    c.ok("switch the Translation Engine" in s.text("#trainStatus"),
         f"offline style training explains itself: {s.text('#trainStatus')!r}")
    p.click("#trainClose")
    fail_case("bad key", "gemini", "bad-key", "gemini-flash-latest", r"API key not valid")
    fail_case("quota", "gemini", "quota-key", "gemini-flash-latest", r"429.*quota")
    fail_case("empty reply", "gemini", "empty-key", "gemini-flash-latest", r"no text.*MAX_TOKENS")
    s.select("engine", "gemini")
    s.set_value("apiKey", "stub-key")

    # Model-level answers, through the Type-text endpoint (no page needed).
    def tt(model, key="stub-key"):
        req = urllib.request.Request(
            BASE + "/api/translate-text", method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"text": "ルフィ", "api_key": key, "provider": "gemini",
                             "model": model, "target_lang": "English"}).encode())
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")
    stub_reset(GEMINI)
    code, body = tt("gemini-2.5-pro")
    c.ok(code == 200 and body.get("translation") == "LINE 0",
         f"Gemini 2.5 Pro works through the thinking ladder ({code} {body})")
    thinks = [(e["config"] or {}).get("thinkingConfig") for e in stub_log(GEMINI)]
    c.eq(thinks, [{"thinkingBudget": 0}, {"thinkingLevel": "low"}, {"thinkingBudget": 128}],
         "2.5 Pro: budget 0 and thinkingLevel refused, the 128 cap accepted")
    stub_reset(GEMINI)
    code, body = tt("gemini-retired")
    c.ok(code == 400 and "not found" in body.get("detail", ""),
         f"a model the account can't use says so ({code} {body.get('detail', '')[:90]!r})")
    c.eq(len(stub_log(GEMINI)), 1, "a 404 model is not retried down the ladder")
    stub_reset(GEMINI)
    tt("gemini-flash-latest", key="bad-key")
    c.eq(len(stub_log(GEMINI)), 1, "a bad key is not retried down the ladder")


# ── 6. webtoon: slices -> one long page ────────────────────────────────
def section_webtoon(s):
    p = s.page
    print("\n== webtoon: two slices, one long page ==")
    stub_reset(GEMINI)
    s.set_workflow("scan-translate")
    s.select("engine", "gemini")
    s.set_value("apiKey", "stub-key")
    s.select("model", "gemini-flash-latest")
    set_wm(s, "", credit="TL: Strip")
    set_options(s, webtoonMode=True)
    if s.visible("#newBtn"):
        p.click("#newBtn")
    s.upload([P3, P4])
    s.wait_preview()
    s.record_requests()
    s.go()
    s.wait_all_done(1, timeout=900000, allow_error=True)
    reqs = translate_reqs(s)
    c.eq(len(reqs), 1, "one request for the whole strip")
    c.eq(reqs[0].get("webtoon") if reqs else None, "true", "webtoon flag sent")
    tid = s.active_task()
    t = status(tid)
    out = decode(fetch(f"/api/result/{tid}"))
    c.ok(out.shape[0] > 2500, f"one long page ({out.shape[1]}x{out.shape[0]})")
    c.ok(any(it.get("type") == "credit" and it["translation"] == "TL: Strip"
             for it in t["result"]["items"]), "the credit is on the strip too")
    pr = [e["prompt"] for e in stub_log(GEMINI) if e["kind"] == "regions"]
    c.ok(pr and all("VERTICAL-SCROLL WEBTOON" in x for x in pr), "webtoon reading order in the prompts")
    set_options(s)
    set_wm(s, "", credit="")


# ── HTTP-only runs (no browser needed) ─────────────────────────────────
def http_translate(img, timeout=900, **extra):
    """POST /api/translate with an image (numpy BGR) and form fields; wait
    for it. Returns (task_id, final status)."""
    import cv2
    import uuid
    ok, buf = cv2.imencode(".png", img)
    fields = {"api_key": "stub-key", "provider": "gemini", "model": "gemini-flash-latest",
              "finish": "off", **extra}
    bnd = uuid.uuid4().hex
    body = b""
    for k, v in fields.items():
        body += (f"--{bnd}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").encode()
    body += (f"--{bnd}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"c.png\"\r\n"
             "Content-Type: image/png\r\n\r\n").encode() + buf.tobytes() + f"\r\n--{bnd}--\r\n".encode()
    req = urllib.request.Request(BASE + "/api/translate", data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={bnd}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        tid = json.loads(r.read())["task_id"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = status(tid)
        if st["status"] in ("done", "error"):
            return tid, st
        time.sleep(1)
    raise TimeoutError("run did not finish")


# ── 7. HD Upscale / One-by-one plumbing, over HTTP on a small crop ─────
def section_upscale(s):
    print("\n== upscale: HD Upscale and One-by-one reach the pipeline ==")
    import cv2
    crop = cv2.imread(P3)[:400, :300]
    tid, st = http_translate(crop, upscale="false", one_by_one="true")
    c.eq(st["status"], "done", f"one-by-one run completes ({st.get('message')})")
    base_shape = decode(fetch(f"/api/result/{tid}")).shape
    tid, st = http_translate(crop, upscale="true")
    c.eq(st["status"], "done", f"HD Upscale run completes ({st.get('message')})")
    up = decode(fetch(f"/api/result/{tid}")).shape
    if up[0] > base_shape[0]:
        c.ok(up[0] >= 2 * base_shape[0], f"HD Upscale enlarged the page {base_shape[:2]} -> {up[:2]}")
    else:
        print("  (no upscaler model on this machine; plumbing only)")


# ── 8. SBS pieces and webtoon strips: credit, and failing loudly ───────
def section_pieces(s):
    print("\n== pieces: SBS cut regions and strips ==")
    import cv2
    page = cv2.imread(P3)
    cut = json.dumps([[[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [0.0, 0.5]],
                      [[0.0, 0.5], [1.0, 0.5], [1.0, 1.0], [0.0, 1.0]]])
    tid, st = http_translate(page, cut_regions=cut, credit="TL: Pieces")
    c.eq(st["status"], "done", f"a page cut into two pieces translates ({st.get('message')})")
    items = (st.get("result") or {}).get("items", [])
    c.ok(any(it.get("type") == "credit" and it["translation"] == "TL: Pieces" for it in items),
         "the credit is on a page translated in pieces")
    c.ok(sum(1 for it in items if it.get("type") != "credit") > 0, "the pieces' lines are there")
    # every piece failing is the page failing, with the reason
    tid, st = http_translate(page, cut_regions=cut, api_key="bad-key")
    c.ok(st["status"] == "error" and "API key not valid" in st.get("message", ""),
         f"pieces with a bad key fail with the reason ({st['status']}: {st.get('message', '')[:70]!r})")
    strip = np.vstack([page, page])
    tid, st = http_translate(strip, webtoon="true", api_key="bad-key")
    c.ok(st["status"] == "error" and "API key not valid" in st.get("message", ""),
         f"a strip with a bad key fails with the reason ({st['status']}: {st.get('message', '')[:70]!r})")


with Session() as s:
    s.context.grant_permissions(["clipboard-read", "clipboard-write"], origin=BASE)
    for name in SECTIONS:
        fn = globals().get("section_" + name)
        if not fn:
            print(f"unknown section {name!r}")
            continue
        try:
            fn(s)
        except Exception as e:                      # keep going; report it
            import traceback
            traceback.print_exc()
            c.ok(False, f"section {name} crashed: {e}")
            try:
                s.screenshot(f"translate_{name}_crash.png")
                s.reload()              # drop any modal left open
            except Exception:
                pass
    c.finish(s)
