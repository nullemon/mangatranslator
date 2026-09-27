"""Built-in series presets: what a release team knows about a series that a
general translator doesn't.

When the Manga title setting names a series listed here, its preset is put
in front of the user's own style instructions for every page (the user's
lines still come after it and win). A preset carries the series' official
names and the house conventions of the release it is modelled on — for One
Piece, TCB's, measured on their release of chapter 1194.
"""
import os
import re
from typing import Optional

ONE_PIECE_STYLE = """SERIES PRESET — One Piece, lettered like TCB's official-quality releases.

COVER / TITLE PAGE (the chapter's first page):
- Chapter title: ONE line, "CHAPTER <number>: <TITLE>" — e.g. 第1194話〝万物は変わりゆく〟
  = "CHAPTER 1194: THE IMPERMANENCE OF ALL THINGS". No quote marks around the
  title. Return it as its own region, type "title".
- Cover request caption (扉絵リクエスト「…」P.N ○○): its own region, type
  "caption": "COVER REQUEST BY <PEN NAME>: \\"<the request>\\"" — the pen name
  romanized as written (P.N トシカZOO = TOSHIKA ZOO).
- Author name 尾田栄一郎: a SEPARATE region, type "title", translation
  "ODA EIICHIRO". Never fold it into the cover caption.
- Magazine promo and editorial notices — anime / merchandise announcements
  in the margin (☆アニメ『…』…配信開始!), "next issue on break" notices
  (次号休載…再開は…), magazine issue info: return them with type "promo"
  and an EMPTY translation. They are erased, not translated.
- A one-line teaser at the top or bottom of a page (☆ゾロVSソマーズ!!…,
  ☆至る!!!) IS translated, type "narration".
- Chapter end marker (…第1194話／おわり) = "...CH.1194 /END".

HOUSE STYLE:
- Big sound effects drawn on the artwork are left in Japanese and not
  translated. Sounds lettered INSIDE a speech balloon are translated as
  English onomatopoeia (ばっ! = FWOOSH!, にゅっ = SCHLOOP!, パキパキ = CRUNCH,
  ガチ.. = BITE...).
- Attack names: no quote marks; keep the build-up ellipsis across balloons
  (〝三刀流〟… = "THREE SWORD STYLE..."). Use the furigana reading where one is
  given (魔気(オーメン) = OMEN, 千八十煩悩(ポンド) = 1080 POUND).
- Punctuation: "?!" (never "!?"), "..." for ellipses, "!!" and "!!!" kept,
  a cut-off line ends in "--". Japanese brackets 「」〝〟 and wave dashes 〜
  are dropped.
- Keep "Haki" untranslated; the three kinds are Conqueror's, Armament and
  Observation Haki.
- When a sentence runs across two balloons, split the English so each
  balloon reads naturally, with "..." carrying it over."""

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "series_data")


def _entries(name: str):
    """(section, "jp = en", note) for every entry in series_data/<name>.txt."""
    path = os.path.join(_DATA, name + ".txt")
    try:
        raw = open(path, encoding="utf-8").read()
    except OSError:
        return []
    out, section = [], None
    for line in raw.splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        if t.startswith("[") and t.endswith("]"):
            section = t[1:-1].strip().lower()
            continue
        body, _, note = t.partition("#")
        body, note = body.strip(), note.strip()
        if "=" in body:
            out.append((section, body, note))
    return out


def _load_data(name: str) -> str:
    """The series' word list (series_data/<name>.txt) as prompt lines:
    [glossary] entries as strict "jp = en" lines, [phrasebook] entries as
    the release team's usual renderings. Notes after "#" are for people;
    glossary lines are sent without them so they parse as exact pairs."""
    gloss, phrases = [], []
    for section, body, note in _entries(name):
        if section == "glossary":
            gloss.append(body)
        elif section == "phrasebook":
            phrases.append(body + (f"  ({note})" if note else ""))
    out = []
    if gloss:
        out.append("GLOSSARY (official names, as this release renders them — use EXACTLY):")
        out += gloss
    if phrases:
        out.append("")
        out.append("PHRASEBOOK (how this release renders these sounds and reactions — "
                   "use these renderings; adapt only if the scene clearly differs):")
        out += phrases
    return "\n".join(out)


_DATA_FILES = {"one piece": "one_piece"}

PRESETS = {
    "one piece": ONE_PIECE_STYLE + "\n\n" + _load_data(_DATA_FILES["one piece"]),
}

# Names a user might type in the Manga title box for each preset.
_ALIASES = {
    "one piece": ("one piece", "onepiece", "one-piece", "ワンピース", "op"),
}

_SERIES_RE = re.compile(r'SERIES: this page is from "([^"]+)"')


def detect(style_prompt: str) -> Optional[str]:
    """The preset key named by the Manga title (as the browser puts it into
    the style instructions), or None."""
    m = _SERIES_RE.search(style_prompt or "")
    if not m:
        return None
    name = re.sub(r"\s+", " ", m.group(1)).strip().lower()
    for key, names in _ALIASES.items():
        if name in names:
            return key
    return None


def apply(style_prompt: str) -> str:
    """The style instructions with the matching series preset in front. The
    user's own lines stay after it, so they override the preset."""
    key = detect(style_prompt)
    if not key or PRESETS[key] in (style_prompt or ""):
        return style_prompt or ""
    return PRESETS[key] + "\n\n" + (style_prompt or "")


def phrasebook(style_prompt: str):
    """The matching preset's word list as (japanese, english) pairs, glossary
    first — for the offline engine, which has no prompt to put them in."""
    key = detect(style_prompt)
    for k, name in _DATA_FILES.items():
        if key is None and PRESETS[k] in (style_prompt or ""):
            key = k            # the preset was already applied to the style
    if key not in _DATA_FILES:
        return []
    pairs = []
    for section, body, _note in _entries(_DATA_FILES[key]):
        if section not in ("glossary", "phrasebook"):
            continue
        jp, _, en = body.partition("=")
        if jp.strip() and en.strip():
            pairs.append((jp.strip(), en.strip()))
    return pairs
