"""Which font a line gets, from how it is being said.

A letterer does not set a whole page in one typeface. A scream is heavy and
loud, a thought is soft and often italic, a narration box is a different face
again, and a sound effect is its own thing entirely. Setting everything in the
dialogue font is the single clearest giveaway of a machine-lettered page.

Two sources decide the voice, in this order:

1. What the translator says. The vision model is looking at the panel — the
   character's face, the balloon's shape, how big the original lettering is —
   so it is asked for a `tone` alongside the translation. It is the better
   judge and it costs nothing extra, being one more field in a reply that was
   already being made.

2. What the line itself looks like. When the model does not answer, or an
   older result is being re-rendered, the text is read: a spiky balloon full
   of capitals and exclamation marks is a shout however it was labelled, and
   a line that trails off in an ellipsis is not.

The mapping from voice to font is deliberately data, not code, so a user who
drops their own fonts into fonts/ can point the roles at them without touching
any of this.
"""
import os
import re
from typing import Dict, List, Optional

#: the voices a line can be given, in the order a tie is broken
ROLES = ("sfx", "title", "narration", "shout", "thought", "whisper", "dialogue")

#: What each voice wants from a typeface, best first.
#:
#: These are the faces scanlation actually uses. Matching is loose and by
#: NAME against whatever sits in fonts/, so dropping in Wild Words, Anime Ace,
#: Komika or Manga Temple is enough — nothing here needs editing.
#:
#: There are deliberately no generic display serifs in this table any more. An
#: earlier version reached for Cinzel for titles and Marcellus for narration
#: because they were installed and looked "different", and they are simply not
#: manga faces — a Roman inscriptional serif in a narration box reads as a
#: wedding invitation. A role with nothing suitable installed now falls back to
#: the page's own dialogue font, which is always the safer answer: one honest
#: typeface beats a wrong one.
ROLE_FONTS: Dict[str, List[str]] = {
    # Ordinary speech. Wild Words and Anime Ace are the two standards.
    "dialogue": ["wildwords", "ccwildwords", "animeace", "animeace2_reg",
                 "mangatemple", "komikaaxis", "digitalstrip", "backissues",
                 "comicneue"],
    # The same voice, raised: the BOLD cut of the dialogue face.
    "shout": ["wildwords-bold", "wildwordsbold", "animeace_b", "animeace2_bld",
              "komikaaxis-bold", "badaboom", "bangers", "anton"],
    # A noise, not speech — the loudest, most distorted face available.
    "sfx": ["badaboom", "komikahand", "komikaaxis", "bangers", "anton",
            "reggaeone"],
    # A caption box is a different voice from speech, but still a comic face.
    "narration": ["ccastrocity", "mangatemple", "komikaslim", "digitalstrip",
                  "backissues"],
    # Inner voice: the dialogue face, slanted and a touch smaller.
    "thought": ["wildwords-italic", "animeace2_ital", "komikaslim"],
    # Chapter titles and signage.
    "title": ["badaboom", "komikaaxis", "mangatemple", "bangers", "anton"],
    # Quiet speech: the dialogue face, smaller.
    "whisper": [],
}

#: Free faces worth installing, and where they come from. Printed by
#: setup_models.py --fonts and by the app when a role has nothing to use.
SUGGESTED = [
    ("Anime Ace 2.0", "blambot.com — free for personal use; the classic "
                      "manga-scanlation dialogue face"),
    ("Komika family", "dafont.com/komika-axis.font — free; a good Wild Words "
                      "stand-in with a real bold"),
    ("Manga Temple", "dafont.com/manga-temple.font — free for personal use"),
    ("Badaboom BB", "blambot.com — free for personal use; the standard SFX "
                    "face"),
]

#: Roles that should also be slanted, on top of the font choice.
ITALIC_ROLES = {"thought", "whisper"}

#: Roles set a little smaller than the box would otherwise give.
#:
#: Only ever SMALLER. The renderer has already fitted the line to its balloon,
#: so multiplying that up pushes it straight back out again — a shout came out
#: with its second line hanging off the bottom of the box. A shout reads as
#: loud from the weight of the face; it does not need to be bigger as well.
ROLE_SCALE = {"whisper": 0.92, "thought": 0.96}

_SHOUT_RE = re.compile(r"[!?][!?]|[!]{1}\s*$")
_TRAIL_RE = re.compile(r"(\.\.\.|…)\s*$")


def infer_tone(text: str, kind: str = "", dark: bool = False) -> str:
    """Work out the voice from the line and its region type.

    Used when the translator did not label the line — an older result being
    re-rendered, the offline engine, or a model that ignored the field.
    """
    k = (kind or "").lower()
    if "sfx" in k or "sound" in k:
        return "sfx"
    if k in ("title", "credit"):
        return "title"
    if k in ("narration", "caption"):
        return "narration"

    t = (text or "").strip()
    if not t:
        return "dialogue"
    letters = [c for c in t if c.isalpha()]
    caps = sum(1 for c in letters if c.isupper()) / max(1, len(letters))

    # A shout: exclamation, or written in capitals and short enough to be one.
    if _SHOUT_RE.search(t) or (caps > 0.9 and len(letters) <= 24 and t.endswith("!")):
        return "shout"
    # White lettering on a black balloon is nearly always a shout or a scream.
    if dark and _SHOUT_RE.search(t + "!"):
        return "shout"
    # Trailing off, and nothing forceful about it.
    if _TRAIL_RE.search(t) and "!" not in t:
        return "whisper"
    return "dialogue"


def normalise(tone: str) -> str:
    """Map whatever the model said onto a role we actually have."""
    t = (tone or "").strip().lower()
    if t in ROLES:
        return t
    alias = {
        "shouting": "shout", "yell": "shout", "yelling": "shout",
        "scream": "shout", "screaming": "shout", "angry": "shout",
        "loud": "shout", "excited": "shout",
        "whispering": "whisper", "quiet": "whisper", "soft": "whisper",
        "murmur": "whisper", "mutter": "whisper",
        "thinking": "thought", "thoughts": "thought", "inner": "thought",
        "monologue": "thought", "internal": "thought",
        "caption": "narration", "narrator": "narration", "narrate": "narration",
        "announcement": "title", "sign": "title", "heading": "title",
        "sound": "sfx", "sound_effect": "sfx", "onomatopoeia": "sfx",
        "normal": "dialogue", "speech": "dialogue", "neutral": "dialogue",
    }
    return alias.get(t, "")


def _index(fonts_dir: str = "fonts") -> Dict[str, str]:
    """Everything installed, keyed by a lowercased stem, so the table above can
    name a font without knowing its exact filename or case."""
    out = {}
    try:
        for n in sorted(os.listdir(fonts_dir)):
            if n.lower().endswith((".ttf", ".otf")):
                out[os.path.splitext(n)[0].lower()] = os.path.join(fonts_dir, n)
    except OSError:
        pass
    return out


def build_map(base_font: Optional[str] = None, fonts_dir: str = "fonts",
              overrides: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """role -> font path, using whatever is installed.

    A role with nothing suitable installed falls back to the page's main font,
    which is the right answer: better one honest typeface than a random one
    that happens to be lying around.
    """
    have = _index(fonts_dir)
    out: Dict[str, str] = {}
    for role in ROLES:
        pick = ""
        for want in ROLE_FONTS.get(role, []):
            # CLOSEST match, not the first one found. "wildwords" is a
            # substring of "wildwords-bold" as well, and the bold sorts first
            # on disk, so a plain first-hit search handed ordinary dialogue the
            # shouting cut of its own face. The shortest stem that contains the
            # name is the one that IS that name.
            hits = [(len(stem), path) for stem, path in have.items()
                    if want in stem]
            if hits:
                pick = min(hits)[1]
                break
        out[role] = pick or (base_font or "")
    for role, path in (overrides or {}).items():
        r = normalise(role) or role
        if r in out and path:
            out[r] = path
    return out


def style_for(item: dict, font_map: Dict[str, str], base_font: str = ""):
    """(font_path, italic, size_scale) for one region."""
    role = normalise(item.get("tone", "")) or infer_tone(
        item.get("translation", ""), item.get("type", ""),
        bool(item.get("dark")))
    path = font_map.get(role) or base_font
    return path, role in ITALIC_ROLES, ROLE_SCALE.get(role, 1.0), role
