"""Shared translation prompts and JSON parsing, used by every translation
backend (Claude, Gemini) so they behave identically."""

import json
import re


def _style_block(style: str) -> str:
    """User-editable style instructions, appended to every translation prompt.
    They override the defaults so the user can steer tone, honorifics, etc."""
    style = (style or "").strip()
    if not style:
        return ""
    return f"""

USER STYLE INSTRUCTIONS (follow these — they override the defaults above):
{style}"""


def _source_label(source_lang: str, target_lang: str) -> str:
    """Human-readable name for the source script used in prompts. 'auto' (or
    empty) becomes a generic 'non-{target}' so the model detects any language."""
    sl = (source_lang or "Japanese").strip()
    if sl.lower() in ("", "auto", "auto-detect", "autodetect", "detect", "any"):
        return f"non-{target_lang}"
    return sl


def _manga_context(target_lang: str) -> str:
    """Shared 'treat the whole page as a manga, not isolated lines' guidance.
    Reading the panels in order with cross-panel context makes the translation
    flow as a real conversation and stay consistent."""
    return f"""
Translate this as a MANGA page, not a list of disconnected lines:
- Read the panels in manga order — right-to-left, top-to-bottom — and follow the
  conversation ACROSS panels so each line answers what was just said.
- Use the whole page as context: who is speaking, their mood from the art
  (shouting, whispering, thinking, crying), and what is happening in the scene.
- Stay consistent across the page — a character's voice, tone, pronouns, name
  and honorifics must not change from one bubble to the next.
- Keep running jokes, callbacks and emphasis intact; render natural, idiomatic
  {target_lang} a real reader would enjoy, never a literal word-for-word gloss."""


def _sfx_rule(translate_sfx: bool) -> str:
    """The SFX instruction shared by the detection prompts. By default sound
    effects are left in the artwork; when enabled they are translated too."""
    if translate_sfx:
        return ('- Set type to "sfx" for sound effects / onomatopoeia and DO '
                'translate them into a short punchy equivalent (e.g. BOOM, '
                'CRASH, THUD) so they can be typeset over the artwork.')
    return ('- Set type to "sfx" for sound effects / onomatopoeia (e.g. ドーン, '
            'バァン, わぁぁ). SFX will NOT be replaced — everything else WILL be '
            'translated and placed.')


def region_translate_prompt(target_lang: str, num_regions: int, style: str = "",
                            source_lang: str = "Japanese",
                            translate_sfx: bool = False) -> str:
    src = _source_label(source_lang, target_lang)
    return f"""You are an expert manga/comic translator. You are shown two images:
1. An original manga page
2. The SAME page with detected text regions marked by red numbered boxes

There are {num_regions} numbered regions. For each one that contains {src}
(or other non-{target_lang}) text, translate it into natural {target_lang}.
{_manga_context(target_lang)}

Return ONLY a JSON array — no markdown fences, no commentary:
[
  {{"id": 1, "original": "original text here", "translation": "{target_lang.upper()} TEXT HERE", "type": "dialogue"}},
  {{"id": 2, "original": "original narration", "translation": "NARRATION", "type": "narration"}}
]

Rules:
- Translate EVERY region that contains non-{target_lang} text — including
  short, repeated, faint, or partly-covered lines (e.g. a bubble that just
  says a name twice). Never skip a bubble or leave source text untranslated.
- Return one entry for EVERY numbered region; if a region is genuinely empty
  or already {target_lang}, give it an empty translation rather than omitting it.
- Write natural, idiomatic {target_lang} — translate meaning and emotion,
  never word-for-word. Use contractions; match the scene's tone.
- Use UPPERCASE for dialogue and narration (standard manga typesetting).
- Keep translations concise — they must fit inside small speech bubbles.
- "type" must be one of: "dialogue", "narration", "sfx", "title".
- Skip regions with no readable text or already in {target_lang}.
- Return ONLY the JSON array.{_style_block(style)}"""


def smart_detect_prompt(target_lang: str, style: str = "",
                        source_lang: str = "Japanese",
                        translate_sfx: bool = False) -> str:
    src = _source_label(source_lang, target_lang)
    return f"""You are an expert manga page analyzer and translator.

Carefully examine this manga page. Find EVERY piece of {src} text — speech
bubbles, narration boxes, titles, credits, captions, and annotations.
{_manga_context(target_lang)}

For each text region, return its bounding box as PERCENTAGE coordinates of the
image (so they are resolution independent):
- x_pct, y_pct: top-left corner as a percentage (0-100)
- width_pct, height_pct: size as a percentage of the image dimensions
- rotation_deg: clockwise rotation of the text from horizontal.
  0 = horizontal. 90 = vertical top-to-bottom column. 10-30 for diagonal bars.

Return ONLY a JSON array — no markdown fences, no commentary:
[
  {{
    "x_pct": 50.0,
    "y_pct": 10.0,
    "width_pct": 20.0,
    "height_pct": 8.0,
    "rotation_deg": 0,
    "original": "original text",
    "translation": "{target_lang.upper()} TEXT",
    "type": "dialogue",
    "in_bubble": true
  }}
]

Rules:
- Find ALL text, including chapter titles, author credits, narration, and captions.
- Include TINY magazine margin text — ad columns, editor notes, corner notes,
  and the small credits running along a chapter-title bar — even when the text
  is only a few pixels tall.
- Translate EVERY region — including short, repeated, or faint lines (even a
  bubble that just repeats a name). Never skip a region or leave {src} text in.
- Also catch small hand-written text drawn ON characters or artwork — tiny
  lines on a face, cheek, or body, scribbled asides next to a character, little
  notes over the art. These ARE text: translate them (type "caption"). Only
  ignore actual drawn features (eyes, mouths, blush, screentone, patterns).
- Be precise with bounding box coordinates — they are used for text replacement.
- Use UPPERCASE for dialogue and narration.
- Keep translations concise so they fit inside their regions.
- type must be one of: "dialogue", "narration", "sfx", "title", "credit",
  "caption", "watermark".
- Set type to "watermark" for a scanlation-site stamp or URL added on top of the
  art (e.g. SOMESITE.NET, www.x.com, a site name/logo text). Leave its
  "translation" empty — these are ERASED, not translated.
- "in_bubble": true if the text is enclosed in a speech bubble or drawn box.
  Set to false for titles, credits, captions, watermarks, and any loose text.
{_sfx_rule(translate_sfx)}
- Do NOT include tiny furigana readings above kanji.
- Return ONLY the JSON array.{_style_block(style)}"""


def free_text_detect_prompt(target_lang: str, bubble_ids: list, style: str = "",
                            source_lang: str = "Japanese",
                            translate_sfx: bool = False) -> str:
    src = _source_label(source_lang, target_lang)
    skip = f"Already-handled bubble IDs: {bubble_ids}. " if bubble_ids else ""
    # SFX go in the "find these" list (with type "sfx") only when the user opts
    # in; otherwise they stay in the "Do NOT include" list and are left in art.
    sfx_find = ('- Sound effects / onomatopoeia — translate into a short punchy '
                'equivalent (BOOM, CRASH, …), type "sfx"\n') if translate_sfx else ""
    sfx_skip = "" if translate_sfx else "- Sound effects / onomatopoeia (ドーン, バァン, etc.)\n"
    sfx_types = ', "sfx"' if translate_sfx else ""
    return f"""You are an expert manga page analyzer and translator.

The speech bubbles on this page have already been translated. {skip}Now find any
REMAINING {src} text that is NOT inside a speech bubble and should be translated:

- Chapter titles (e.g. 第1163話 "約束")
- Author names / credits
- Narration text outside bubbles
- Editorial notes, page captions, magazine announcements
- TINY magazine margin/ad text and corner notes (even a few pixels tall), and
  the small credits running along a chapter-title bar
- Location/time labels
- Large dramatic vertical text columns overlaid on the artwork (common in
  action scenes — e.g. a bold vertical column of kanji/kana like
  生きようとしてるからだ!!!, 俺は俺の意志で, etc.)
- Text on diagonal narration bars or banners crossing the page
- Small hand-written text drawn ON a character or the artwork — a tiny line on
  a face/cheek/body, a scribbled aside next to someone, a little note over the
  art. These ARE text — translate them (type "caption").
- A scanlation-site watermark or URL stamped on the art (e.g. SOMESITE.NET,
  www.x.com, a site name/logo text) — report it with type "watermark" and an
  EMPTY translation; it will be ERASED, not translated.
{sfx_find}
Do NOT include:
{sfx_skip}- Text already in {target_lang}
- Tiny furigana readings above kanji
- Drawn FEATURES, not text: eyes, mouths, blush marks, screentone and pattern
  fills. Flag a region only when it is actual written characters — but DO flag
  real written text even when it sits on a face or body.

Find EVERY such region — do not stop at the obvious ones. Include short,
narrow, or vertical columns, faint text, and small boxes tucked next to other
boxes. Each separate framed box (caption / narration panel) is its OWN region.
Scan the ENTIRE page systematically from top-to-bottom, left-to-right — do
not miss any region.

For each text region, return its bounding box as PERCENTAGE coordinates:
- x_pct, y_pct: top-left corner (0-100)
- width_pct, height_pct: size as percentage of image dimensions
- rotation_deg: clockwise rotation of the text from horizontal.
  0 = normal horizontal left-to-right text.
  90 = vertical top-to-bottom column (common for Japanese text on manga pages).
  Use 10-30 for text running along a diagonal narration bar or tilted strip.
  The bounding box must be the AXIS-ALIGNED rectangle that fully contains
  all the text at the reported angle.

Make each box cover the FULL extent of the text — every character from start to
end, including any trailing punctuation (!!!, …, etc.). For vertical columns,
the box must cover the entire column top to bottom. For diagonal bars, include
the full bar extent. Do not clip the edges of characters.
Never let one box overlap, contain, or spill into a neighbouring box; if two
texts sit in separate boxes, return two separate, non-overlapping regions.

Return ONLY a JSON array — no markdown fences:
[
  {{
    "x_pct": 30.0,
    "y_pct": 2.0,
    "width_pct": 40.0,
    "height_pct": 5.0,
    "rotation_deg": 0,
    "original": "第1163話 \\"約束\\"",
    "translation": "CHAPTER 1163: \\"THE PROMISE\\"",
    "type": "title"
  }}
]

type must be one of: "title", "credit", "narration", "caption", "watermark"{sfx_types}.
Translate naturally and idiomatically — meaning and emotion, not word-for-word.
Keep translations concise. Return an empty array [] if no free text is found.
Return ONLY the JSON array.{_style_block(style)}"""


def text_translate_prompt(target_lang: str, style: str = "",
                          source_lang: str = "Japanese",
                          translate_sfx: bool = False,
                          with_image: bool = False) -> str:
    src = _source_label(source_lang, target_lang)
    img_line = (
        "\nYou are ALSO shown the manga page image. READ THE PANEL — the "
        "characters, their expressions and the action — and translate to fit "
        "what is actually happening, not a literal word-for-word of the text.\n"
        if with_image else "")
    return f"""You are an expert manga translator localizing for an official
{target_lang} release. Below is a JSON object mapping each speech-bubble id to
the {src} text that was read from that bubble (by OCR). Translate every
entry into natural, punchy {target_lang}.
{img_line}{_manga_context(target_lang)}
The ids are in reading order; treat them as one flowing conversation.

Return ONLY a JSON array — no markdown fences, no commentary:
[
  {{"id": 1, "original": "<the original text>", "translation": "{target_lang.upper()} TEXT", "type": "dialogue"}}
]

Rules:
- Keep each translation matched to the SAME id — never move text between ids.
- Write like official {target_lang} manga lettering — natural and idiomatic,
  never stiff or word-for-word literal. Translate the MEANING and the emotion.
- Match the scene's tone: shouted lines are short and forceful; inner
  monologue is reflective; casual speech uses contractions (I'M, DON'T, IT'S).
- Preserve emphasis: keep !!, !?, trailing ellipses (...) and dashes (—) for
  unfinished or trailing thoughts.
- Keep character names romanized; keep honorifics (-SAN, -KUN, SENPAI) rather
  than translating them away.
- Use UPPERCASE (standard manga lettering). Keep it concise to fit the bubble.
- "type" is one of: "dialogue", "narration", "sfx".
- Small expression sounds in speech bubbles (にっ = *GRIN*, ハッ = *GASP*,
  フッ = *SMIRK*, etc.) are "dialogue" — translate them into an expressive
  English word wrapped in asterisks (e.g. *GRIN*). Only mark loud dramatic
  sound effects (ドーン, バキ, ゴゴゴ) as "sfx".
- If an entry's text is empty or unreadable, return an empty translation for it.
- Return ONLY the JSON array.{_style_block(style)}"""


def crop_translate_prompt(target_lang: str, source_lang: str = "Japanese",
                          style: str = "") -> str:
    """Read + translate the text in a single cropped region (the editor's Add /
    Lasso-add tool) — vision-based so it works for any source language."""
    src = _source_label(source_lang, target_lang)
    return f"""You are an expert manga translator. This image is a CROPPED region
of a manga page that contains {src} text. Read ALL the text in it and translate
it into natural, idiomatic {target_lang} — read the picture for context.

Return ONLY a JSON array (no markdown), exactly one object:
[{{"original": "<the source text>", "translation": "{target_lang.upper()} TEXT"}}]

Use UPPERCASE for the translation. If there is no readable text, return [].{_style_block(style)}"""


def learn_profile_prompt(target_lang: str = "English", source_lang: str = "Japanese") -> str:
    """Ask the model to study several ALREADY-TRANSLATED pages from one series and
    distill the group's house style into a reusable profile (glossary + rules)."""
    src = _source_label(source_lang, target_lang)
    return f"""You are a senior manga localization editor building a STYLE GUIDE for
a scanlation team. You are shown several finished, already-translated {target_lang}
pages from ONE ongoing series (translated from {src}). Study them as a body of
work and reverse-engineer this team's house style so future chapters can be
translated to match EXACTLY.

Extract:
1. A GLOSSARY of recurring proper nouns and series terms with the team's exact
   {target_lang} rendering — character names, places, organizations, techniques /
   special powers, signature catchphrases. Capture spelling/casing precisely.
2. HONORIFICS policy — are -san/-kun/-sama/-sensei kept, dropped, or adapted?
3. SOUND-EFFECTS policy — are SFX translated/typeset, or left in the art?
4. A short STYLE GUIDE (3-6 sentences): tone, formality, how casual/forceful the
   dialogue reads, punctuation/emphasis habits, and any recurring phrasing.

Return ONLY a JSON object (no markdown fences, no commentary):
{{
  "glossary": [
    {{"term": "<name as it appears / romaji>", "translation": "<exact {target_lang} rendering>", "notes": "<role/context, optional>"}}
  ],
  "honorifics": "<one-line policy>",
  "sfx_policy": "<one-line policy>",
  "style_guide": "<3-6 sentences capturing the house voice>"
}}

Rules:
- Only include glossary entries you actually SEE evidence for on these pages.
- Prefer the team's spelling even if unusual; do not 'correct' it.
- Keep it concise and high-signal; no duplicates.
- Return ONLY the JSON object."""


def extract_json_object(text: str) -> dict:
    """Pull a single JSON object out of a model response (fences/prose tolerant)."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON object from model response: {text[:300]}")


def extract_json_array(text: str) -> list:
    """Robustly pull a JSON array out of a model response that may include
    markdown fences or stray prose."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Some models wrap the array, e.g. {"regions": [...]}
            for v in data.values():
                if isinstance(v, list):
                    return v
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # TRUNCATION SALVAGE: long pages can overrun the model's output budget and
    # the array arrives cut off mid-object. Losing the whole page over the last
    # broken element is far worse than dropping that element — walk the text
    # and recover every COMPLETE top-level {...} object.
    objs = []
    depth = 0
    start = None
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    objs.append(json.loads(text[start:i + 1]))
                except json.JSONDecodeError:
                    pass
                start = None
    if objs:
        print(f"[prompts] response was truncated — salvaged "
              f"{len(objs)} complete objects")
        return objs
    raise ValueError(f"Could not parse JSON from model response: {text[:300]}")
