"""Adds cleaned-text and detected-language columns to both processed tables.

Runs after `pairs.py`, so it touches messages and pairs together:
  messages: text_clean, language
  pairs:    customer_msg_clean, customer_language

Language is detected from the cleaned text (mentions and URLs are a large share of
a short tweet), and the detector is the same one `online/` uses at request time —
offline labels and runtime labels have to agree or every language-based decision
built on them is unreliable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from support_agent.text import clean_text, detect_language

__all__ = ["clean_text", "detect_language", "run"]


def run(processed_dir: Path) -> pd.DataFrame:
    messages = pd.read_parquet(processed_dir / "messages.parquet")
    messages["text_clean"] = messages["text"].astype(str).map(clean_text)
    messages["language"] = messages["text_clean"].map(detect_language)
    messages.to_parquet(processed_dir / "messages.parquet", index=False)

    pairs_path = processed_dir / "pairs.parquet"
    pairs = pd.read_parquet(pairs_path)
    pairs["customer_msg_clean"] = pairs["customer_msg"].astype(str).map(clean_text)
    pairs["customer_language"] = pairs["customer_msg_clean"].map(detect_language)
    pairs.to_parquet(pairs_path, index=False)

    return messages


if __name__ == "__main__":
    from config.settings import settings

    run(settings.processed_dir)
