"""Glossary: pasting names from a wiki, and sending only the ones a page needs.

The user pastes whatever they copied from the One Piece wiki (a technique
list, a character page, a table, raw wikitext) or their own `jp = en` lines;
parse_pairs() must pull the official names out of all of it. Big glossaries
are trimmed per page by focus_style() so the prompt carries only the names
that page actually uses.
Run: python3 tests/test_glossary.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.glossary import normalize, parse_pairs, relevant, focus_style, to_lines


def as_dict(pairs):
    return {jp: en for jp, en in pairs}


def check_simple_lines():
    text = "\n".join([
        "覇王色 = Conqueror's Haki",
        "Armament Haki = 武装色",              # English on the left
        "見聞色 → Observation Haki",
        "悪魔の実 -> Devil Fruit",
        "- 麦わらの一味 → Straw Hat Pirates",   # the profile's own format
        "ガッツ = Guts · キャスカ = Casca · 鷹の団 = Band of the Hawk · keep -dono as -dono",
        "keep -dono as -dono",                  # a rule, not a pair
        "Use UK spelling throughout",
    ])
    d = as_dict(parse_pairs(text))
    assert d.get("覇王色") == "Conqueror's Haki", d
    assert d.get("武装色") == "Armament Haki", d
    assert d.get("見聞色") == "Observation Haki", d
    assert d.get("悪魔の実") == "Devil Fruit", d
    assert d.get("麦わらの一味") == "Straw Hat Pirates", d
    assert d.get("ガッツ") == "Guts" and d.get("キャスカ") == "Casca", d
    assert d.get("鷹の団") == "Band of the Hawk", d
    assert len(d) == 8, d

    # " : " is NOT a separator — English attack names contain colons.
    p = parse_pairs("三刀流奥義 六道の辻 = Santoryu Ogi: Rokudo no Tsuji")
    assert p == [("三刀流奥義 六道の辻", "Santoryu Ogi: Rokudo no Tsuji")], p
    assert parse_pairs("Santoryu Ogi: Rokudo no Tsuji") == []
    assert parse_pairs("見聞色: Observation Haki") == [], "colon must not split"

    # Dedupe by normalised Japanese: the first one wins.
    p = parse_pairs("覇王色 = Conqueror's Haki\n覇王色 = Supreme King's Haki\n「覇王色」 = X")
    assert p == [("覇王色", "Conqueror's Haki")], p
    print("simple lines OK")


def check_table_rows():
    text = "\n".join([
        "Japanese Name\tRomanized Name\tOfficial English Name",
        "English\tKanji\tRomaji",
        "Rokudo no Tsuji\t六道の辻\tRokudō no Tsuji",
        "1\tGomu Gomu no Pistol\tゴムゴムの銃 (ピストル)\tGomu Gomu no Pisutoru",
        "Japanese Name\t覇王色",               # label cell is never the English
    ])
    d = as_dict(parse_pairs(text))
    assert d.get("六道の辻") == "Rokudo no Tsuji", d
    assert d.get("ゴムゴムの銃") == "Gomu Gomu no Pistol", d
    assert "覇王色" not in d, d
    assert len(d) == 2, d
    print("table rows OK")


def check_prose():
    text = "\n".join([
        'Rokudo no Tsuji (六道の辻 Rokudō no Tsuji?, literally "Six Paths Crossroads")',
        "Gomu Gomu no Pistol (ゴムゴムの銃 (ピストル) Gomu Gomu no Pisutoru?)",
        "Gomu Gomu no Gatling （ゴムゴムの銃乱打（ガトリング） Gomu Gomu no Gatoringu?）",
        "Zoro used his technique Santoryu Ogi: Rokudo no Tsuji Kai "
        "(三刀流奥義 六道の辻改 Santōryū Ōgi Rokudō no Tsuji Kai?) to cut him down.",
        "He then fired the Gomu Gomu no Red Hawk (ゴムゴムの火拳銃 (レッドホーク) "
        "Gomu Gomu no Reddo Hōku?) and the Gomu Gomu no Kong Gun "
        "(ゴムゴムの猿王銃 (コングガン) Gomu Gomu no Kongu Gan?).",
        "the ordinary word (ことば) should never become a name",
    ])
    d = as_dict(parse_pairs(text))
    assert d.get("六道の辻") == "Rokudo no Tsuji", d
    assert d.get("ゴムゴムの銃") == "Gomu Gomu no Pistol", d
    assert d.get("ゴムゴムの銃乱打") == "Gomu Gomu no Gatling", d
    assert d.get("三刀流奥義 六道の辻改") == "Santoryu Ogi: Rokudo no Tsuji Kai", d
    assert d.get("ゴムゴムの火拳銃") == "Gomu Gomu no Red Hawk", d
    assert d.get("ゴムゴムの猿王銃") == "Gomu Gomu no Kong Gun", d
    assert "ことば" not in d, d
    # "Zoro used his technique X (...)" gives exactly X.
    p = parse_pairs("Zoro used his technique Rokudo no Tsuji (六道の辻 Rokudō no Tsuji?)")
    assert p == [("六道の辻", "Rokudo no Tsuji")], p
    print("wiki prose OK")


def check_wikitext():
    text = "\n".join([
        "'''{{Nihongo|[[Gomu Gomu no Mi|Gum-Gum Fruit]]|ゴムゴムの実|Gomu Gomu no Mi}}''' is a Devil Fruit.",
        "{{Qnihongo|''Conqueror's Haki''|覇王色の覇気|Haōshoku no Haki}}",
        "{{Nihongo||悪魔の実|Akuma no Mi}}",     # no English: fall back to romaji
        "{{nihongo|[[Monkey D. Luffy]]|モンキー・D・ルフィ|Monkī Dī Rufi}}",
    ])
    d = as_dict(parse_pairs(text))
    assert d.get("ゴムゴムの実") == "Gum-Gum Fruit", d
    assert d.get("覇王色の覇気") == "Conqueror's Haki", d
    assert d.get("悪魔の実") == "Akuma no Mi", d
    assert d.get("モンキー・D・ルフィ") == "Monkey D. Luffy", d
    print("wikitext templates OK")


def check_infobox():
    text = "\n".join([
        "Japanese Name:",
        "覇王色の覇気",
        "Romanized Name:",
        "Haōshoku no Haki",
        "Official English Name:",
        "Conqueror's Haki",
        "Debut:",
        "Chapter 1",
    ])
    assert parse_pairs(text) == [("覇王色の覇気", "Conqueror's Haki")], parse_pairs(text)
    # Same-line / tab layout, and the Romanized Name fallback.
    text = "Japanese Name:\t閻魔\nRomanized Name:\tEnma"
    assert parse_pairs(text) == [("閻魔", "Enma")], parse_pairs(text)
    text = "Japanese Name: 秋水\nEnglish Name: Shusui\nRomanized Name: Shūsui"
    assert parse_pairs(text) == [("秋水", "Shusui")], parse_pairs(text)
    print("infobox OK")


def check_normalize():
    assert normalize("「覇王色！！」") == normalize("覇王色")
    assert normalize("ゴムゴムの銃 (ピストル)") == "ゴムゴムの銃"
    assert normalize("モンキー・D・ルフィ…") == "モンキーdルフィ"
    print("normalize OK")


def big_glossary():
    """60 filler names plus the ones the page tests care about."""
    filler = [(f"人名{chr(0x30A2 + i)}{chr(0x4E00 + 500 + i)}", f"Name {i}") for i in range(60)]
    return filler + [
        ("覇王色", "Conqueror's Haki"),
        ("三刀流 豹琴玉", "Santoryu: Hyo Kin Dama"),
        ("六道の辻", "Rokudo no Tsuji"),
        ("麦わらの一味", "Straw Hat Pirates"),
    ]


def check_relevant():
    pairs = big_glossary()
    # Small glossaries are always sent whole.
    assert relevant(pairs[:30], "何もない") == pairs[:30]
    # Empty page text: cap only.
    assert relevant(pairs, "", cap=10) == pairs[:10]

    # The attack is called out one character per bubble:
    # 「三刀流…」「豹」「琴」 — never the whole name in one place.
    page = "".join(["三刀流…", "豹", "琴", "覇王色だと！？"])
    got = as_dict(relevant(pairs, page))
    assert got.get("覇王色") == "Conqueror's Haki", got   # exact
    assert "三刀流 豹琴玉" in got, got                     # split across bubbles
    assert "六道の辻" not in got, got
    assert "麦わらの一味" not in got, got
    # Exact matches rank first.
    assert relevant(pairs, page)[0] == ("覇王色", "Conqueror's Haki")
    # Cap is honoured.
    assert len(relevant(pairs, page, cap=1)) == 1
    print("relevant OK")


def check_focus_style():
    pairs = big_glossary()
    style = "\n".join([
        'SERIES: this page is from "One Piece".',
        "GLOSSARY — translate these names/terms EXACTLY and consistently on every page:",
        to_lines(pairs),
        "keep -dono as -dono",
        "ガッツ = Guts · キャスカ = Casca",      # two pairs on one line: kept as-is
        "Punchy shounen tone.",
    ])
    out = focus_style(style, "覇王色！！ 三刀流… 豹 琴")
    lines = out.split("\n")
    for keep in ('SERIES: this page is from "One Piece".',
                 "GLOSSARY — translate these names/terms EXACTLY and consistently on every page:",
                 "keep -dono as -dono", "ガッツ = Guts · キャスカ = Casca",
                 "Punchy shounen tone.",
                 "覇王色 = Conqueror's Haki", "三刀流 豹琴玉 = Santoryu: Hyo Kin Dama"):
        assert keep in lines, (keep, out)
    assert "六道の辻 = Rokudo no Tsuji" not in lines, out
    assert "人名ア" + chr(0x4E00 + 500) + " = Name 0" not in lines, out
    # Order of the surviving lines is preserved.
    assert lines.index("覇王色 = Conqueror's Haki") < lines.index("keep -dono as -dono")

    # Profile-format lines are pair lines too.
    prof = "GLOSSARY — use these canonical renderings:\n" + "\n".join(
        f"- {jp} → {en}" for jp, en in pairs)
    out = focus_style(prof, "覇王色")
    assert "- 覇王色 → Conqueror's Haki" in out and "- 六道の辻 → Rokudo no Tsuji" not in out, out

    # Empty page text only caps a huge list.
    out = focus_style(style, "", cap=5)
    assert sum(1 for l in out.split("\n") if " = Name " in l) == 5, out
    assert "Punchy shounen tone." in out

    # Small glossaries are returned unchanged.
    small = "GLOSSARY:\n覇王色 = Conqueror's Haki\n六道の辻 = Rokudo no Tsuji\nkeep -dono"
    assert focus_style(small, "全然関係ない") == small
    assert focus_style("", "x") == ""
    print("focus_style OK")


def main():
    check_simple_lines()
    check_table_rows()
    check_prose()
    check_wikitext()
    check_infobox()
    check_normalize()
    check_relevant()
    check_focus_style()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
