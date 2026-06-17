"""Series translation profiles ("trained" styles).

A profile is what the app learns from a batch of a group's already-translated
chapters: a glossary of canonical names/terms, an honorifics + SFX policy and a
short style guide. Picking a profile at translate time injects all of this into
the vision model's prompt so a new chapter is rendered in the SAME house style —
in-context learning, the practical ML approach for this problem.

Profiles are plain JSON under profiles/ so they're easy to back up / edit / share.
"""

import json
import os
import re
import time
from typing import Dict, List, Optional

PROFILE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "profiles")


def _ensure_dir():
    os.makedirs(PROFILE_DIR, exist_ok=True)


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return s or "series"


def _path(slug: str) -> str:
    return os.path.join(PROFILE_DIR, f"{slug}.json")


def _clean_glossary(items) -> List[dict]:
    out, seen = [], set()
    for it in (items or []):
        if not isinstance(it, dict):
            continue
        term = str(it.get("term", "")).strip()
        tr = str(it.get("translation", "")).strip()
        if not tr or not (term or tr):
            continue
        key = (term.lower(), tr.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append({"term": term, "translation": tr,
                    "notes": str(it.get("notes", "")).strip()})
    return out


def new_profile(name: str) -> dict:
    return {
        "name": (name or "Series").strip() or "Series",
        "slug": slugify(name),
        "glossary": [],
        "honorifics": "",
        "sfx_policy": "",
        "style_guide": "",
        "typeset": {"font": "", "text_case": "upper", "finish": "clean"},
        "sources": 0,
        "updated": int(time.time()),
    }


def normalize(p: dict) -> dict:
    """Coerce an arbitrary dict into a valid profile (used for saves/edits)."""
    base = new_profile(p.get("name", "Series"))
    base["slug"] = slugify(p.get("name") or base["slug"])
    base["glossary"] = _clean_glossary(p.get("glossary"))
    for k in ("honorifics", "sfx_policy", "style_guide"):
        base[k] = str(p.get(k, "") or "").strip()
    ts = p.get("typeset") or {}
    if isinstance(ts, dict):
        base["typeset"] = {
            "font": str(ts.get("font", "")),
            "text_case": str(ts.get("text_case", "upper") or "upper"),
            "finish": str(ts.get("finish", "clean") or "clean"),
        }
    try:
        base["sources"] = int(p.get("sources", 0))
    except (TypeError, ValueError):
        base["sources"] = 0
    base["updated"] = int(time.time())
    return base


def merge_learned(existing: Optional[dict], learned: dict, name: str,
                  added_sources: int = 0) -> dict:
    """Fold a freshly-learned analysis into an existing profile (or a new one),
    unioning the glossary and refreshing the prose fields."""
    prof = existing or new_profile(name)
    prof["name"] = (name or prof.get("name") or "Series").strip()
    prof["slug"] = slugify(prof["name"])

    by_key = {(g["term"].lower(), g["translation"].lower()): g
              for g in prof.get("glossary", [])}
    for g in _clean_glossary(learned.get("glossary")):
        by_key[(g["term"].lower(), g["translation"].lower())] = g
    prof["glossary"] = list(by_key.values())

    for k in ("honorifics", "sfx_policy", "style_guide"):
        v = str(learned.get(k, "") or "").strip()
        if v:
            prof[k] = v
    prof["sources"] = int(prof.get("sources", 0)) + int(added_sources)
    prof["updated"] = int(time.time())
    return prof


def save(profile: dict) -> dict:
    _ensure_dir()
    profile = normalize(profile) if "slug" not in profile else profile
    profile["slug"] = slugify(profile.get("name", profile.get("slug", "series")))
    with open(_path(profile["slug"]), "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    return profile


def load(slug: str) -> Optional[dict]:
    try:
        with open(_path(slugify(slug)), encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def delete(slug: str) -> bool:
    try:
        os.remove(_path(slugify(slug)))
        return True
    except FileNotFoundError:
        return False


def list_profiles() -> List[Dict]:
    _ensure_dir()
    out = []
    for fn in sorted(os.listdir(PROFILE_DIR)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(PROFILE_DIR, fn), encoding="utf-8") as f:
                p = json.load(f)
            out.append({"slug": p.get("slug", fn[:-5]), "name": p.get("name", fn[:-5]),
                        "terms": len(p.get("glossary", [])), "sources": p.get("sources", 0),
                        "updated": p.get("updated", 0)})
        except Exception:
            continue
    return out


def prompt_block(profile: dict) -> str:
    """Render a profile as style instructions to prepend to the translation
    prompt. Returns "" if the profile is empty."""
    if not profile:
        return ""
    lines = [f'SERIES STYLE PROFILE — "{profile.get("name", "Series")}" '
             "(learned from this team's released chapters). Match it EXACTLY:"]
    gloss = profile.get("glossary") or []
    if gloss:
        lines.append("\nGLOSSARY — use these canonical renderings; never "
                     "re-translate or re-spell these terms:")
        for g in gloss[:120]:
            term = g.get("term", "")
            tr = g.get("translation", "")
            note = g.get("notes", "")
            arrow = f"{term} → {tr}" if term else tr
            lines.append(f"- {arrow}" + (f"  ({note})" if note else ""))
    if profile.get("honorifics"):
        lines.append(f"\nHONORIFICS: {profile['honorifics']}")
    if profile.get("sfx_policy"):
        lines.append(f"SOUND EFFECTS: {profile['sfx_policy']}")
    if profile.get("style_guide"):
        lines.append(f"\nHOUSE VOICE / STYLE:\n{profile['style_guide']}")
    out = "\n".join(lines).strip()
    return out
