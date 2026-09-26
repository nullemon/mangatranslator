"""Glossary helpers: pull official names out of pasted wiki text, and keep only
the names a page actually needs in the prompt.

The translator is good at dialogue but cannot know official names from recent
chapters (覇王色 is "Conqueror's Haki", never "Supreme King's"). The user's
network cannot reach the wiki, so they paste whatever they copied from it —
a technique list, a character page, a table, raw wikitext — and parse_pairs()
turns it into `jp = en` glossary lines.

A pasted glossary can run to hundreds or thousands of names. Sending all of
them with every page wastes tokens and dilutes the instruction, so the
pipeline calls focus_style() once the page's bubbles have been read: every
non-glossary instruction is kept, and only the glossary lines whose Japanese
actually appears on the page (or mostly appears — attack names are often
called out one character per bubble) are sent.
"""

import re
from typing import List, Optional, Sequence, Tuple

Pair = Tuple[str, str]

# Glossaries up to this size are sent whole; filtering only kicks in above it.
SMALL = 40


# ── character classes ─────────────────────────────────────────────────────

def _is_jp_char(ch: str) -> bool:
    o = ord(ch)
    return (0x3040 <= o <= 0x30FF          # hiragana + katakana (incl. ・ ー)
            or 0x3400 <= o <= 0x4DBF       # CJK extension A
            or 0x4E00 <= o <= 0x9FFF       # CJK unified ideographs
            or 0xF900 <= o <= 0xFAFF       # CJK compatibility ideographs
            or 0xFF66 <= o <= 0xFF9D       # half-width katakana
            or o == 0x3005)                # 々


def has_japanese(s: str) -> bool:
    return any(_is_jp_char(c) for c in (s or ""))


# ── normalisation ─────────────────────────────────────────────────────────

_PAREN_RE = re.compile(r"[（(][^()（）]*[)）]")
_STRIP_CHARS = set("「」『』“”\"'‘’…・.!?！？、。")


def _strip_furigana(s: str) -> str:
    """Remove parenthesised readings: ゴムゴムの銃 (ピストル) -> ゴムゴムの銃."""
    prev = None
    while prev != s:
        prev, s = s, _PAREN_RE.sub("", s)
    return s


def normalize(s: str) -> str:
    """A matching key: no whitespace, quotes, dots, bangs or furigana."""
    s = _strip_furigana(s or "")
    return "".join(c for c in s if not c.isspace() and c not in _STRIP_CHARS).lower()


# ── wikitext cleanup ──────────────────────────────────────────────────────

_LINK_LABEL_RE = re.compile(r"\[\[([^\[\]|]*)\|([^\[\]]*)\]\]")
_LINK_RE = re.compile(r"\[\[([^\[\]|]*)\]\]")
_BOLD_RE = re.compile(r"'''(.*?)'''")
_ITALIC_RE = re.compile(r"''(.*?)''")
_TEMPLATE_RE = re.compile(r"\{\{\s*(q?nihongo)\s*\|([^{}]*)\}\}", re.IGNORECASE)
_ANY_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}", re.DOTALL)


def _strip_markup(s: str) -> str:
    s = _LINK_LABEL_RE.sub(r"\2", s)
    s = _LINK_RE.sub(r"\1", s)
    s = _BOLD_RE.sub(r"\1", s)
    s = _ITALIC_RE.sub(r"\1", s)
    return s


def _template_pair(body: str) -> Optional[Pair]:
    """{{Nihongo|EN|JP|romaji|...}} -> (JP, EN); EN falls back to the romaji."""
    parts = [p.strip() for p in body.split("|")]
    # Named params (lead=yes, ...) are not positional.
    pos = [p for p in parts if not re.match(r"^\w+\s*=", p)]
    en = pos[0] if pos else ""
    jp = pos[1] if len(pos) > 1 else ""
    if not en and len(pos) > 2:
        en = pos[2]
    jp = _strip_furigana(jp).strip()
    en = _clean_en(en)
    if jp and en and has_japanese(jp) and not has_japanese(en):
        return jp, en
    return None


# ── simple `jp = en` lines ────────────────────────────────────────────────

# NB: ":" is deliberately NOT a separator — English attack names contain it
# ("Santoryu Ogi: Rokudo no Tsuji").
_SEP_RE = re.compile(r"\s*(?:->|→|⇒|=>|＝|=)\s*")
_BULLET_RE = re.compile(r"^\s*(?:[-*•‣▪◦]+\s+|\d+[.)]\s+)")


def _clean_en(s: str) -> str:
    s = (s or "").strip()
    s = re.split(r"\s+#", s)[0].strip()          # "term = En  # note"
    s = s.strip(" \t\"“”「」『』")
    return s


def _simple_pair(seg: str) -> Optional[Pair]:
    seg = _BULLET_RE.sub("", seg).strip()
    m = _SEP_RE.search(seg)
    if not m:
        return None
    left, right = seg[:m.start()].strip(), seg[m.end():].strip()
    if not left or not right:
        return None
    lj, rj = has_japanese(left), has_japanese(right)
    if lj == rj:
        return None
    jp, en = (left, right) if lj else (right, left)
    jp = _strip_furigana(jp).strip(" \t\"“”「」『』")
    en = _clean_en(en)
    if not jp or not en or not has_japanese(jp):
        return None
    return jp, en


def _simple_line(line: str) -> List[Pair]:
    """`覇王色 = Conqueror's Haki`, `- 覇王色 → Conqueror's Haki`, and several
    pairs separated by `·` on one line."""
    segs = [s for s in line.split("·") if s.strip()]
    out = []
    for seg in segs:
        p = _simple_pair(seg)
        if p:
            out.append(p)
    return out


# ── tab-separated table rows ──────────────────────────────────────────────

_LABELS = ("japanese name", "romanized name", "official english name",
           "english name", "japanese", "romanized", "romaji", "english",
           "kanji", "name", "translation", "literal translation", "meaning",
           "first appearance", "type", "debut")


def _is_label(cell: str) -> bool:
    return cell.strip().rstrip(":：").strip().lower() in _LABELS


def _table_row(line: str) -> List[Pair]:
    cells = [c.strip() for c in line.split("\t")]
    ji = next((i for i, c in enumerate(cells) if has_japanese(c)), None)
    if ji is None:
        return []
    jp = _strip_furigana(cells[ji]).strip()

    def usable(c):
        return (c and not has_japanese(c) and not _is_label(c)
                and not re.fullmatch(r"[\d\s.,#-]+", c))
    en = next((c for c in cells[:ji] if usable(c)), "")
    if not en:
        en = next((c for c in cells[ji + 1:] if usable(c)), "")
    en = _clean_en(en)
    if jp and en:
        return [(jp, en)]
    return []


# ── rendered wiki prose: `Name (日本語 Rōmaji?, literally "...")` ──────────

_PROSE_PAREN_RE = re.compile(
    r"[（(]((?:[^()（）]|[（(][^()（）]*[)）])*)[)）]")
_PARTICLES = {"no", "ni", "wo", "o", "to", "de", "ga", "wa", "of", "the",
              "and", "a", "an", "in", "on", "for", "with", "&", "vs.", "vs"}
_OPEN_Q = "\"“「『'‘"
_CLOSE_Q = "\"”」』'’"
_STOP_END = ".,;!?)）]」』\"”"


def _en_before(pre: str) -> str:
    toks = pre.split()
    picked: List[str] = []
    for tok in reversed(toks):
        low = tok.lower()
        if low in _PARTICLES:
            picked.append(tok)
            continue
        if tok[-1] in _STOP_END:
            break
        core = tok.lstrip(_OPEN_Q)
        if not core:
            break
        ok = (core[0].isupper()
              or any(ch in core for ch in ":-'’")
              or any(ch.isdigit() for ch in core))
        if not ok:
            break
        picked.append(core)
        if core != tok:          # reached an opening quote: the name starts here
            break
    picked.reverse()
    while picked and picked[0].lower() in _PARTICLES:
        picked.pop(0)
    return " ".join(picked).strip(_CLOSE_Q + " ")


def _prose_line(line: str) -> List[Pair]:
    out = []
    last = 0
    for m in _PROSE_PAREN_RE.finditer(line):
        inner = m.group(1)
        pre = line[last:m.start()]
        last = m.end()
        if not has_japanese(inner):
            continue
        toks = _strip_furigana(inner).split()
        jp_toks = []
        for t in toks:
            if not has_japanese(t):
                break
            jp_toks.append(t.rstrip("?,;、").strip("「」『』"))
        jp = " ".join(t for t in jp_toks if t).strip()
        if not jp:
            continue
        en = _en_before(pre)
        if en and not has_japanese(en):
            out.append((jp, en))
    return out


# ── infobox text ──────────────────────────────────────────────────────────

_INFO_LABEL_RE = re.compile(
    r"^\s*(japanese name|romanized name|official english name|english name)"
    r"\s*[:：]?\s*(.*)$", re.IGNORECASE)
_INFO_PRIORITY = ("official english name", "english name", "romanized name")


def _info_pair(fields: dict) -> Optional[Pair]:
    jp = _strip_furigana(fields.get("japanese name", "")).strip()
    if not jp or not has_japanese(jp):
        return None
    for k in _INFO_PRIORITY:
        en = _clean_en(fields.get(k, ""))
        if en and not has_japanese(en):
            return jp, en
    return None


# ── public API ────────────────────────────────────────────────────────────

def parse_pairs(text: str) -> List[Pair]:
    """Extract (japanese, english) name pairs from anything pasted from a wiki
    or typed by hand. Deduped by normalised Japanese — the first one wins.
    Lines that are rules rather than pairs ("keep -dono as -dono") are
    ignored."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    # Templates can wrap lines; flatten them so each sits on one line.
    text = _ANY_TEMPLATE_RE.sub(lambda m: " ".join(m.group(0).split()), text)
    text = _strip_markup(text)

    found: List[Pair] = []
    info: dict = {}
    pending_label: Optional[str] = None

    def flush_info():
        nonlocal info
        p = _info_pair(info)
        if p:
            found.append(p)
        info = {}

    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue

        # Infobox: a label followed by its value (same line, or the next).
        if pending_label is not None:
            if not _INFO_LABEL_RE.match(line):
                info[pending_label] = line.split("\t")[0].strip()
                pending_label = None
                continue
            pending_label = None
        m = _INFO_LABEL_RE.match(line)
        if m:
            label = m.group(1).lower()
            rest = m.group(2).strip()
            rest_cells = [c for c in rest.split("\t") if c.strip()]
            # A table header row ("Japanese Name\tRomanized Name\t...") is
            # not an infobox field.
            if any(_is_label(c) for c in rest_cells):
                continue
            if label in info:
                flush_info()
            if rest_cells:
                info[label] = rest_cells[0].strip()
            else:
                pending_label = label
            continue

        # {{Nihongo|EN|JP|...}} / {{Qnihongo|...}}
        def _tpl(mm):
            p = _template_pair(mm.group(2))
            if p:
                found.append(p)
            return " "
        line = _TEMPLATE_RE.sub(_tpl, line).strip()
        if not line or not has_japanese(line):
            continue

        if "\t" in line:
            found.extend(_table_row(line))
            continue
        simple = _simple_line(line)
        if simple:
            found.extend(simple)
            continue
        found.extend(_prose_line(line))
    flush_info()

    out: List[Pair] = []
    seen = set()
    for jp, en in found:
        key = normalize(jp)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append((jp, en))
    return out


def to_lines(pairs: Sequence[Pair]) -> str:
    return "\n".join(f"{jp} = {en}" for jp, en in pairs)


def _rank(jps: Sequence[str], page_text: str, cap: int) -> List[int]:
    """Indices of the glossary entries this page needs, best first."""
    n = len(jps)
    if n <= SMALL:
        return list(range(n))
    page = normalize(page_text or "")
    if not page:
        return list(range(min(n, cap)))
    page_chars = set(page)
    exact, partial = [], []
    for i, jp in enumerate(jps):
        key = normalize(jp)
        if not key:
            continue
        if key in page:
            exact.append(i)
            continue
        if len(key) < 2:
            continue
        # Hiragana are particles / okurigana found on every page; judge the
        # match on the kanji + katakana when the name has any.
        chars = {c for c in key if not 0x3040 <= ord(c) <= 0x309F} or set(key)
        frac = len(chars & page_chars) / len(chars)
        if frac >= 0.6:
            partial.append((-frac, -len(key), i))
    partial.sort()
    return (exact + [i for _, _, i in partial])[:cap]


def relevant(pairs: Sequence[Pair], page_text: str, cap: int = 80) -> List[Pair]:
    """The pairs worth sending for this page. Small glossaries go whole; big
    ones keep names whose Japanese is on the page first, then names most of
    whose characters are (an attack called out one character per bubble)."""
    pairs = list(pairs)
    return [pairs[i] for i in _rank([p[0] for p in pairs], page_text, cap)]


def focus_style(style: str, page_text: str, cap: int = 80) -> str:
    """Trim the glossary lines inside a style prompt down to the ones this page
    needs. Every line that is not a single name pair (instructions, headers,
    rules) is kept verbatim."""
    if not style:
        return style
    lines = style.split("\n")
    pair_idx, jps = [], []
    for i, line in enumerate(lines):
        if not has_japanese(line):
            continue
        ps = parse_pairs(line)
        if len(ps) == 1:
            pair_idx.append(i)
            jps.append(ps[0][0])
    if len(pair_idx) <= SMALL:
        return style
    keep = {pair_idx[k] for k in _rank(jps, page_text, cap)}
    drop = set(pair_idx) - keep
    return "\n".join(l for i, l in enumerate(lines) if i not in drop)
