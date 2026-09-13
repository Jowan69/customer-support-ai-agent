"""Central settings: paths, model names, thresholds. Reads .env for secrets."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # --- versioning (stamped into every decision row) ---
    artifact_version: str = "v2"
    intents_version: str = "v2"

    # --- paths ---
    root_dir: Path = ROOT
    data_dir: Path = ROOT / "data"
    raw_dir: Path = ROOT / "data" / "raw"
    processed_dir: Path = ROOT / "data" / "processed"
    labels_dir: Path = ROOT / "data" / "labels"
    artifacts_dir: Path = ROOT / "artifacts"
    # Small, tracked inputs the eval needs and the artifacts directory cannot supply:
    # artifacts/ is ~500MB per version and stays out of the repo, so the two files the
    # frozen eval reads live here instead. See frozen/README.md.
    frozen_dir: Path = ROOT / "frozen"
    config_dir: Path = ROOT / "config"
    cache_dir: Path = ROOT / ".cache"
    db_path: Path = ROOT / ".cache" / "decisions.db"

    @property
    def artifact_dir(self) -> Path:
        return self.artifacts_dir / self.artifact_version

    @property
    def intents_path(self) -> Path:
        """config/intents.<intents_version>.json.

        Versioned by path, not just by the `version` string inside the file. The
        taxonomy is what `data/labels/golden.csv` was labelled against: every intent
        in that CSV is a name from this file, and the labels mean nothing without it.
        When this was a single unversioned `config/intents.json`, rebuilding the
        offline artifacts and re-naming the new clusters overwrote it in place - and
        the golden set silently became a set of labels for a taxonomy that no longer
        existed. A rebuild now writes `intents.v3.json` and leaves `intents.v2.json`,
        and the golden set, alone.

        The unversioned file is still read if the versioned one is absent, so an older
        working copy keeps running; nothing writes back to it.
        """
        versioned = self.config_dir / f"intents.{self.intents_version}.json"
        legacy = self.config_dir / "intents.json"
        if not versioned.exists() and legacy.exists():
            return legacy
        return versioned

    # --- corpus filtering (offline only; see offline/filter.py) ---
    min_message_chars: int = 10
    min_message_words: int = 3

    # --- retrieval ---
    top_k: int = 5

    # --- models ---
    # multilingual: the corpus contains non-English customer messages, and an
    # English-only encoder clusters them by language instead of by meaning
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    grok_model: str = "openai/gpt-oss-20b"
    grok_api_key: str = Field(default="", alias="GROK_API_KEY")

    # --- thresholds ---
    classify_confidence_threshold: float = 0.6
    retrieval_score_threshold: float = 0.5
    mean_retrieval_score_threshold: float = 0.4

    # --- decision weighting (see online/decide.py; the combination rule is provisional) ---
    escalation_weight_threshold: int = 2

    # --- language ---
    # Auto-send only where a reply can actually be reviewed. Everything else,
    # "unknown" included, goes to a human: you cannot audit a language nobody reads.
    auto_send_languages: tuple[str, ...] = ("en",)

    # --- intents that always go to a human, whatever else the signals say ---
    escalate_always_intents: tuple[str, ...] = ("fraud_or_unauthorized",)

    # --- intents that raise a sensitive-action flag ---
    refund_intents:  tuple[str, ...] = ("pricing_or_promotion",)
    # fraud_or_unauthorized is deliberately absent: escalate_always_intents short-
    # circuits decide() before the weighted concerns are read, so flagging it here
    # would be configuration that can never be consulted.
    payment_intents: tuple[str, ...] = ("prime_membership",)

    # --- intents where a correct reply needs account-specific facts the agent
    # cannot see (this order's real status, whether a refund actually went out)
    # or a backend action it cannot take (issue the refund, change the account,
    # reroute the package) - see LABELLING.md section 4. Weighted at
    # escalation_weight_threshold so any one forces escalation on its own,
    # independent of classifier confidence or retrieval strength: those measure
    # whether the model is sure what the message *is*, not whether it can act on
    # it. Added 2026-09-13 after eval/decisions.py's reason breakdown showed the
    # old weighting auto-sent these intents on confidence/retrieval alone and
    # was wrong most of the time (e.g. refund_status via refund_intent-only:
    # 12.5% correct on n=8; order_status/delivery_not_received/account_manage
    # had no intent-based concern at all and fell into confident_and_grounded,
    # the worst-performing auto bucket at 16.7% correct on n=60).
    #
    # Extended the same day: after the first four were pulled out,
    # confident_and_grounded was re-broken-down by intent and the same pattern
    # showed up in nearly every intent left in it - these six were each >=85%
    # "should have escalated" in that breakdown (account_access, item_damaged
    # and item_wrong_or_missing were 100%; delivery_late 96%; device_or_content_
    # issue 89%; delivery_courier_issue 88%). service_feedback (75%) and
    # contact_request (50%, genuinely mixed) were left out on purpose - below
    # the same bar the first four were held to.
    account_specific_intents: tuple[str, ...] = (
        "refund_status", "account_manage", "order_status", "delivery_not_received",
        "delivery_late", "delivery_courier_issue", "device_or_content_issue",
        "account_access", "item_damaged", "item_wrong_or_missing",
    )

    # --- llm client ---
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 3


settings = Settings()
