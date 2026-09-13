"""Text normalisation and language detection, shared by the offline build and the
online query path.

Lives outside `offline/` and `online/` so both can use the exact same functions
without `online/` importing `offline/` (the layering rule) and without the logic
being duplicated. A query cleaned — or a language detected — differently from the
indexed corpus would silently skew everything downstream.

Everything here REWRITES text. Nothing here drops a record: offline may discard a
useless row, but online cannot — a live customer is waiting — so row-level
filtering lives in `offline/filter.py` and its online counterpart is escalation.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# --- cleaning ---------------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@\w+")
_ELONGATION_RE = re.compile(r"(.)\1{2,}")
_WHITESPACE_RE = re.compile(r"\s+")

# Emoji are demojized (😡 -> :enraged_face:) rather than deleted: in a complaint
# corpus they often carry the whole affect, and an unknown-token emoji is worth
# less to the encoder than a real word. Deleting them was throwing that away.
_EMOJI_DELIMITERS = (" :", ": ")

# No quote-stripping rule. Removing text between quotation marks deleted exactly
# the part worth keeping -- customers quote the error they are reporting
# ('the app says "delivery failed"') -- and it only matched ASCII/curly double
# quotes, so it cleaned English differently from French or Japanese.


def clean_text(text: str) -> str:
    """Normalise one message. Safe to call on already-clean text (idempotent)."""
    if not isinstance(text, str):
        return ""

    # NFKC first: fullwidth forms, styled maths letters and compatibility
    # characters all collapse to their plain equivalents before anything is matched.
    text = unicodedata.normalize("NFKC", text)

    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _demojize(text)
    text = _ELONGATION_RE.sub(r"\1\1", text)  # "soooo" -> "soo", "!!!!!" -> "!!"
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _demojize(text: str) -> str:
    import emoji

    return emoji.demojize(text, delimiters=_EMOJI_DELIMITERS)


# --- language detection -----------------------------------------------------

UNKNOWN_LANGUAGE = "unknown"

# Calibrated on support-style phrasing. py3langid picks the right language even on
# short text, but its confidence falls off sharply, so length does most of the work
# and the probability floor is only a backstop:
#   "mi paquete no ha llegado"  -> es 0.59      "thanks"      -> xh 0.11
#   "donde esta mi pedido"      -> es 0.37      "ok" / "hi"   -> sr 0.01
#   "where is my order"         -> en 0.93      "DM sent"     -> ro 0.02
# Anything that fails either gate is UNKNOWN_LANGUAGE, never a guess: downstream
# this is meant to route to a human, and a wrong label is worse than no label.
MIN_CHARS = 12
MIN_CONFIDENCE = 0.30

# Demojized names are English words, so they drag detection towards English and
# swamp short messages: "donde esta mi pedido" plus four emoji is classified pt
# (0.23) instead of es. Strip them before classifying, never before embedding.
_EMOJI_NAME_RE = re.compile(r":[a-z0-9_+\-]+:")


@lru_cache(maxsize=1)
def _identifier():
    """Loaded once per process — building it per call would dominate the runtime."""
    from py3langid.langid import MODEL_FILE, LanguageIdentifier

    return LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)


def detect_language(
    text: str,
    *,
    min_chars: int = MIN_CHARS,
    min_confidence: float = MIN_CONFIDENCE,
) -> str:
    """Best-effort ISO 639-1 code for `text`, or UNKNOWN_LANGUAGE.

    Pass already-cleaned text: @mentions and URLs are a large fraction of a short
    tweet and drag the detector around.
    """
    if not isinstance(text, str):
        return UNKNOWN_LANGUAGE

    stripped = _EMOJI_NAME_RE.sub(" ", text).strip()
    if len(stripped) < min_chars:
        return UNKNOWN_LANGUAGE

    language, confidence = _identifier().classify(stripped)
    if float(confidence) < min_confidence:
        return UNKNOWN_LANGUAGE
    return str(language)
