"""Settings come back from localStorage exactly after a reload, and the
settings sections lay out without clipping or overlap at 1280 and 1600 px."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import Check, SHOTS, Session, skip_unless_server

skip_unless_server()
c = Check("settings persistence + layout")

# (element id, kind, value to set). Kinds: value (input event), select
# (change event), check (toggle).
SETTINGS = [
    ("engine", "select", "gemini"),
    ("sourceLang", "select", "Korean"),
    ("targetLang", "select", "French"),
    ("transStyle", "select", "literal"),
    ("mangaTitle", "value", "Hunter × Hunter"),
    ("stylePrompt", "value", "keep honorifics · UK spelling"),
    ("glossary", "value", "ゴン = Gon\nキルア = Killua"),
    ("fontSelect", "select", None),          # first real font, filled in below
    ("textCase", "select", "keep"),
    ("styleFonts", "check", True),
    ("fontVariety", "select", "expressive"),
    ("pageFinish", "select", "restore"),
    ("watermark", "value", "@persist"),
    ("wmStyle", "select", "pill"),
    ("wmPlace", "select", "tl"),
    ("wmSize", "select", "l"),
    ("wmOpacity", "select", "35"),
    ("wmTileToo", "check", True),
    ("credit", "value", "TL: Persist"),
    ("smartMode", "check", True),
    ("oneByOne", "check", True),
    ("translateSfx", "check", True),
    ("maxQuality", "check", True),
    ("hdUpscale", "check", True),
    ("compressOut", "check", True),
    ("removeWatermark", "check", False),
    ("replaceWatermark", "check", True),
    ("sbsMode", "check", True),
    ("gpuCap", "select", "60"),
    ("endScan", "value", "QA Scans"),
    ("endDiscord", "value", "discord.gg/qa"),
    ("endTheme", "select", "neon"),
    ("endMessage", "value", "thanks"),
    ("endUseColor", "check", True),
    ("chapterName", "value", "QA Ch 9"),
    ("rawStyle", "select", "scan"),
    ("rawStrength", "value", "1.4"),
    ("tileMode", "select", "3"),
    ("protectDark", "check", True),
    ("enhanceProvider", "select", "openai"),
]


def read_all(s):
    out = {}
    for el_id, kind, _ in SETTINGS:
        out[el_id] = s.checked(el_id) if kind == "check" else s.value(el_id)
    out["workflow"] = s.page.evaluate("document.querySelector('.wf-card.active').dataset.wf")
    out["model"] = s.value("model")
    out["apiKey"] = s.value("apiKey")
    return out


def overflow_report(s):
    """Controls that stick out of the settings panel or overlap a sibling."""
    return s.page.evaluate("""() => {
      const bar = document.getElementById('settingsBar');
      const B = bar.getBoundingClientRect();
      const sel = 'input, select, button, textarea, .toggle, img, .font-picker-btn';
      const els = [...bar.querySelectorAll(sel)].filter(e => {
        const r = e.getBoundingClientRect();
        const st = getComputedStyle(e);
        return r.width > 0 && r.height > 0 && st.visibility !== 'hidden' && !e.closest('.toggle input');
      });
      const bad = [];
      const rects = els.map(e => [e, e.getBoundingClientRect()]);
      for (const [e, r] of rects) {
        if (r.right > B.right + 0.5 || r.left < B.left - 0.5)
          bad.push('clipped: ' + (e.id || e.className) + ' ' + Math.round(r.left) + '-' + Math.round(r.right) + ' vs bar ' + Math.round(B.left) + '-' + Math.round(B.right));
      }
      // The eye / "saved" badge sit INSIDE the key input on purpose.
      const overlay = e => e.classList.contains('key-reveal') || e.classList.contains('key-status');
      for (let i = 0; i < rects.length; i++) for (let j = i + 1; j < rects.length; j++) {
        const [a, ra] = rects[i], [b, rb] = rects[j];
        if (a.contains(b) || b.contains(a)) continue;
        if ((overlay(a) || overlay(b)) && a.closest('.key-field') === b.closest('.key-field')) continue;
        const ox = Math.min(ra.right, rb.right) - Math.max(ra.left, rb.left);
        const oy = Math.min(ra.bottom, rb.bottom) - Math.max(ra.top, rb.top);
        if (ox > 2 && oy > 2)
          bad.push('overlap: ' + (a.id || a.className) + ' × ' + (b.id || b.className) + ` (${Math.round(ox)}x${Math.round(oy)})`);
      }
      // text labels must not run past the panel either
      for (const l of bar.querySelectorAll('label, .toggle-hint, .muted')) {
        const r = l.getBoundingClientRect();
        if (r.width && r.right > B.right + 0.5) bad.push('label past edge: ' + l.textContent.trim().slice(0, 30));
      }
      return { bad, hscroll: document.documentElement.scrollWidth > window.innerWidth,
               scrollWidth: document.documentElement.scrollWidth, inner: window.innerWidth };
    }""")


for width in (1280, 1600):
    with Session(width=width, height=1000) as s:
        p = s.page
        p.wait_for_function("() => document.querySelectorAll('#fontSelect option').length > 1", timeout=15000)
        first_font = p.evaluate("document.querySelectorAll('#fontSelect option')[1].value")
        for i, (el_id, kind, val) in enumerate(SETTINGS):
            if el_id == "fontSelect":
                SETTINGS[i] = (el_id, kind, first_font)
        s.set_workflow("raw-scan-translate")           # shows the enhance panel too
        for el_id, kind, val in SETTINGS:
            if kind == "check":
                s.set_toggle(el_id, val)
            elif kind == "select":
                s.select(el_id, val)
            else:
                s.set_value(el_id, val)
        s.select("model", "gemini-pro-latest")
        s.set_value("apiKey", "AIza-test-key")
        before = read_all(s)
        for el_id, kind, val in SETTINGS:
            got = before[el_id]
            c.ok(got == val, f"[{width}] {el_id} accepted {val!r} (got {got!r})")

        # layout at this width, with everything expanded and the preview on
        p.wait_for_function("() => wmPreview.naturalWidth > 0", timeout=60000)   # preview rendered
        rep = overflow_report(s)
        s.screenshot(f"settings_{width}.png")
        c.ok(not rep["hscroll"], f"[{width}] no horizontal page scroll ({rep['scrollWidth']} vs {rep['inner']})")
        c.ok(not rep["bad"], f"[{width}] no clipped or overlapping settings controls: {rep['bad']}")
        # with the enhance panel (it lives in its own card below)
        rep2 = p.evaluate("""() => {
          const panel = document.getElementById('enhancePanel');
          const B = panel.getBoundingClientRect(); const bad = [];
          for (const e of panel.querySelectorAll('input, select, textarea, button')) {
            const r = e.getBoundingClientRect();
            if (r.width && (r.right > B.right + 0.5 || r.left < B.left - 0.5)) bad.push(e.id || e.className);
          }
          return bad; }""")
        c.ok(not rep2, f"[{width}] enhance panel controls fit: {rep2}")

        # reload: every value must come back exactly
        s.reload()
        p.wait_for_function("() => document.querySelectorAll('#fontSelect option').length > 1", timeout=15000)
        p.wait_for_timeout(300)
        after = read_all(s)
        for k, v in before.items():
            c.ok(after.get(k) == v, f"[{width}] {k} restored: {after.get(k)!r} == {v!r}")
        c.ok(p.evaluate("document.getElementById('fontPicker') && document.querySelector('.font-picker-btn').textContent") ==
             first_font.rsplit(".", 1)[0], "font picker button shows the restored font")
        c.ok(s.visible("#wmPreview"), "watermark preview shows again after reload")

        # Reset defaults keeps the API key but clears the rest
        p.click("#resetSettings")
        p.wait_for_selector("#goBtn", state="attached")
        p.wait_for_timeout(500)
        c.eq(s.value("watermark"), "", f"[{width}] reset clears the watermark")
        c.eq(s.value("apiKey"), "AIza-test-key", f"[{width}] reset keeps the API key")
        c.eq(s.value("fontVariety"), "pro", f"[{width}] reset restores the Pro default")

c.finish()
print("screenshots in", SHOTS)
