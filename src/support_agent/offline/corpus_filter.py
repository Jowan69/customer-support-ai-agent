"""Row-level filtering of the retrieval corpus. Offline only.

Drops pairs that should never reach the index:
  * nothing left after cleaning ("@AmazonHelp 👍" -> "")
  * too short to be a usable retrieval example ("thanks", "any update")
  * exact duplicates of a pair already kept

Why this is not in `text.py`: those functions are shared with `online/`, and online
cannot drop anything — a live customer is waiting. The runtime counterpart of
"discard this row" is "escalate to a human", which belongs in `online/decide.py`.

Only pairs.parquet is filtered. messages.parquet is left whole because chains.py
walks it by tweet_id and removing rows would break the thread reconstruction.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd

# Emoji names survive cleaning but are not content: ":thumbs_up:" alone is an empty
# message for retrieval purposes, so it must not satisfy the length gates.
_EMOJI_NAME_RE = re.compile(r":[a-z0-9_+\-]+:")
_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)

# Chinese, Japanese, Thai, Lao, Khmer and Myanmar do not put spaces between words,
# so `\w+` sees a whole sentence as a single token. Applying a minimum word count to
# them would discard every message in those languages - the exact bias the
# multilingual encoder was adopted to avoid. For these scripts the character gate
# alone decides, and it is generous enough: they carry far more meaning per
# character than Latin text does.
_UNSPACED_SCRIPT_RE = re.compile(
    "["
    "\u3040-\u30ff"  # hiragana, katakana
    "\u3400-\u4dbf"  # CJK extension A
    "\u4e00-\u9fff"  # CJK unified ideographs
    "\u0e00-\u0e7f"  # Thai
    "\u0e80-\u0eff"  # Lao
    "\u1000-\u109f"  # Myanmar
    "\u1780-\u17ff"  # Khmer
    "]"
)


def _word_count(text: str) -> int:
    """Number of words, or `inf` for scripts that are not space-delimited."""
    if _UNSPACED_SCRIPT_RE.search(text):
        return math.inf
    return len(_WORD_RE.findall(text))


def _content_only(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return _EMOJI_NAME_RE.sub(" ", text).strip()


def _dedup_key(text: str) -> str:
    """Casefolded, whitespace-collapsed cleaned text.

    Exact matching on a normalised key — no similarity threshold to tune, so the
    result is deterministic and reproducible.
    """
    return " ".join(_content_only(text).casefold().split())


def filter_pairs(
    pairs: pd.DataFrame,
    *,
    min_chars: int,
    min_words: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return (kept pairs, counts of what each rule removed)."""
    if "customer_msg_clean" not in pairs.columns:
        raise RuntimeError("pairs.parquet has no customer_msg_clean column - run clean.py first")

    total = len(pairs)
    content = pairs["customer_msg_clean"].map(_content_only)

    empty = content.str.len() == 0
    kept = pairs[~empty]
    content = content[~empty]

    too_short = (content.str.len() < min_chars) | (content.map(_word_count) < min_words)
    kept = kept[~too_short]

    before_dedup = len(kept)
    keys = kept["customer_msg_clean"].map(_dedup_key)
    kept = kept[~keys.duplicated(keep="first")]

    counts = {
        "total": total,
        "removed_empty": int(empty.sum()),
        "removed_too_short": int(too_short.sum()),
        "removed_duplicate": before_dedup - len(kept),
        "kept": len(kept),
    }
    return kept.reset_index(drop=True), counts


def run(processed_dir: Path, *, min_chars: int, min_words: int) -> dict[str, int]:
    path = processed_dir / "pairs.parquet"
    pairs = pd.read_parquet(path)
    kept, counts = filter_pairs(pairs, min_chars=min_chars, min_words=min_words)
    kept.to_parquet(path, index=False)
    return counts


if __name__ == "__main__":
    from config.settings import settings

    print(
        run(
            settings.processed_dir,
            min_chars=settings.min_message_chars,
            min_words=settings.min_message_words,
        )
    )
