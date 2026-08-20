"""Offline translation — no API, no key, no network, no per-page cost.

Like a phone's downloaded language pack: a small translation model lives on
disk and runs on your own GPU, so a page is translated in a fraction of a
second instead of waiting on a vision API call.

What it is good for
-------------------
- Speed. Whole pages translate in well under a second, and a 20-page batch
  never waits on a network round trip.
- Bulk / draft passes, SFX, and any chapter you are going to letter by hand
  anyway.
- Working with no internet at all (once the model is downloaded).

What it is NOT
--------------
It is a sentence-level model. It sees ONE line at a time with no picture and
no memory of the previous bubble, so it cannot do what the big vision models
do: read the panel, work out who is speaking, and keep a character's voice
consistent down the page. Expect flatter, more literal English, and the
occasional miss on fragments and slang. Use it for volume; use Gemini/Claude
when the wording matters.

Models are the Helsinki-NLP OPUS-MT set — a few hundred MB each, one per
source language. Override the choice with MANGA_MT_MODEL if you prefer
another (a manga/VN-tuned checkpoint drops straight in, same interface).
"""
import os
import re
from typing import Dict, List, Optional

# One model per source language. All are small seq2seq (Marian) checkpoints,
# roughly 300MB each — you only need the languages you actually read.
DEFAULT_MODELS = {
    "japanese": "Helsinki-NLP/opus-mt-ja-en",
    "korean": "Helsinki-NLP/opus-mt-ko-en",
    "chinese": "Helsinki-NLP/opus-mt-zh-en",
    "arabic": "Helsinki-NLP/opus-mt-ar-en",
    "spanish": "Helsinki-NLP/opus-mt-es-en",
    "french": "Helsinki-NLP/opus-mt-fr-en",
    "german": "Helsinki-NLP/opus-mt-de-en",
    "portuguese": "Helsinki-NLP/opus-mt-roa-en",
    "russian": "Helsinki-NLP/opus-mt-ru-en",
    "italian": "Helsinki-NLP/opus-mt-it-en",
    "indonesian": "Helsinki-NLP/opus-mt-id-en",
    "thai": "Helsinki-NLP/opus-mt-th-en",
    "vietnamese": "Helsinki-NLP/opus-mt-vi-en",
}

# What a fresh install actually needs. Everything else is opt-in, because
# each pack is a few hundred MB and most people read one or two languages.
COMMON_LANGS = ("japanese", "korean")


def installed_langs():
    """Which packs are already on disk (no download, no model load)."""
    out = []
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return out
    for lang, mid in DEFAULT_MODELS.items():
        try:
            hit = try_to_load_from_cache(mid, "config.json")
            if isinstance(hit, str) and os.path.exists(hit):
                out.append(lang)
        except Exception:
            continue
    return out
FALLBACK_MODEL = "Helsinki-NLP/opus-mt-mul-en"   # many-to-English

_CACHE: Dict[str, "LocalMT"] = {}


def model_id_for(source_lang: str) -> str:
    """Which checkpoint translates FROM this language."""
    override = (os.environ.get("MANGA_MT_MODEL") or "").strip()
    if override:
        return override
    key = (source_lang or "Japanese").strip().lower()
    for name, mid in DEFAULT_MODELS.items():
        if key.startswith(name[:2]) or key == name:
            return mid
    return DEFAULT_MODELS.get(key, FALLBACK_MODEL)


class LocalMT:
    """A loaded offline translation model. Process-wide cached per model id."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.ok = False
        self._tok = None
        self._model = None
        self._device = "cpu"
        self._load()

    #: set when loading failed, so callers can tell the user WHY
    last_error = ""

    def _load(self):
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        except Exception as e:
            LocalMT.last_error = (
                "The `transformers` library is missing. Install the offline "
                "packs with: python setup_models.py --offline-translate")
            print(f"[local-mt] transformers unavailable ({e})")
            return
        # These checkpoints are Marian models, whose tokenizer is
        # sentencepiece-based. Without sentencepiece, transformers cannot build
        # it and reports the deeply unhelpful "Unrecognized configuration class
        # MarianConfig to build an AutoTokenizer" — with MarianConfig listed
        # among the supported ones. Check up front and say what is actually
        # wrong.
        import importlib.util
        if importlib.util.find_spec("sentencepiece") is None:
            LocalMT.last_error = (
                "`sentencepiece` is missing — the offline translation models "
                "need it to read text. Install it with:\n"
                "    python setup_models.py --offline-translate\n"
                "or directly:  pip install sentencepiece sacremoses")
            print(f"[local-mt] {LocalMT.last_error}", flush=True)
            return
        try:
            import torch
            from .device import torch_device
            print(f"[local-mt] loading {self.model_id} "
                  f"(first run downloads a few hundred MB)...", flush=True)
            self._tok = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
            self._device = torch_device()
            try:
                self._model = self._model.to(self._device)
            except Exception:
                self._device = "cpu"          # e.g. an unsupported Metal op
                self._model = self._model.to("cpu")
            self._model.eval()
            self.ok = True
            print(f"[local-mt] ready on {self._device}", flush=True)
        except Exception as e:
            msg = str(e)
            if "AutoTokenizer" in msg and "Unrecognized configuration" in msg:
                # same root cause, reached a different way
                msg = ("the tokenizer could not be built — this almost always "
                       "means `sentencepiece` is missing. "
                       "pip install sentencepiece sacremoses")
            LocalMT.last_error = f"{self.model_id}: {msg}"
            print(f"[local-mt] could not load {self.model_id}: {msg}")
            self.ok = False

    # ── text prep ───────────────────────────────────────────────────────
    @staticmethod
    def _clean_source(text: str) -> str:
        """Manga OCR returns one line per balloon row. The model wants a
        sentence, so join the rows — but keep real sentence breaks, which is
        what stops two separate lines being fused into nonsense."""
        t = (text or "").replace("\r", "")
        t = re.sub(r"[ \t]+", " ", t)
        # Japanese/Chinese have no spaces: joining rows with a space would
        # insert one mid-word. Join with nothing, and only keep newlines that
        # follow sentence-ending punctuation.
        parts = [p.strip() for p in t.split("\n") if p.strip()]
        if not parts:
            return ""
        out = parts[0]
        for p in parts[1:]:
            out += ("\n" if out[-1] in "。．.!！?？…」』】" else "") + p
        return out

    @staticmethod
    def _collapse_repeats(text: str, keep: int = 2, max_phrase: int = 5) -> str:
        """Small MT models sometimes loop ("No! No! No! No! No!"). Collapse any
        phrase repeated 3+ times in a row down to `keep` copies.

        Done on WORDS rather than with a regex: a regex needs word boundaries
        that punctuation breaks, which is exactly what manga lines are full of.
        """
        words = text.split()
        n = len(words)
        out, i = [], 0
        while i < n:
            hit = False
            for plen in range(1, max_phrase + 1):
                if i + 2 * plen > n:
                    continue
                phrase = words[i:i + plen]
                reps, j = 1, i + plen
                while j + plen <= n and words[j:j + plen] == phrase:
                    reps += 1
                    j += plen
                if reps >= 3:
                    for _ in range(keep):
                        out.extend(phrase)
                    i = j
                    hit = True
                    break
            if not hit:
                out.append(words[i])
                i += 1
        return " ".join(out)

    @staticmethod
    def _polish(text: str) -> str:
        """Tidy the model's output for lettering: collapse whitespace, cut the
        repetition loops small MT models sometimes emit, and drop the stray
        quotes they like to wrap short lines in."""
        t = re.sub(r"\s+", " ", (text or "")).strip()
        t = LocalMT._collapse_repeats(t)
        if len(t) >= 2 and t[0] in "\"'\u201c\u201d" and t[-1] in "\"'\u201c\u201d":
            t = t[1:-1].strip()
        return t

    # ── translation ─────────────────────────────────────────────────────
    def translate_many(self, texts: List[str], max_batch: int = 16) -> List[str]:
        """Translate a list of lines. Batched, so a whole page is one or two
        forward passes rather than a call per bubble."""
        if not self.ok or not texts:
            return ["" for _ in texts]
        import torch

        prepared = [self._clean_source(t) for t in texts]
        results: List[str] = []
        for i in range(0, len(prepared), max_batch):
            chunk = prepared[i:i + max_batch]
            keep = [n for n, c in enumerate(chunk) if c]
            if not keep:
                results.extend("" for _ in chunk)
                continue
            batch = [chunk[n] for n in keep]
            try:
                enc = self._tok(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=512)
                enc = {k: v.to(self._device) for k, v in enc.items()}
                with torch.inference_mode():
                    gen = self._model.generate(
                        **enc,
                        max_new_tokens=256,
                        num_beams=4,          # markedly better than greedy
                        no_repeat_ngram_size=4,
                        length_penalty=1.0,
                    )
                dec = self._tok.batch_decode(gen, skip_special_tokens=True)
            except Exception as e:
                print(f"[local-mt] batch failed: {e}")
                dec = ["" for _ in batch]
            out = ["" for _ in chunk]
            for n, d in zip(keep, dec):
                out[n] = self._polish(d)
            results.extend(out)
        return results

    def translate_one(self, text: str) -> str:
        return self.translate_many([text])[0]


def get(source_lang: str = "Japanese") -> Optional[LocalMT]:
    """Load (once) and return the offline model for this source language."""
    mid = model_id_for(source_lang)
    hit = _CACHE.get(mid)
    if hit is None:
        hit = LocalMT(mid)
        _CACHE[mid] = hit
    return hit if hit.ok else None


def unload() -> bool:
    """Drop the loaded models (used by the idle memory sweeper)."""
    global _CACHE
    had = bool(_CACHE)
    _CACHE = {}
    return had
