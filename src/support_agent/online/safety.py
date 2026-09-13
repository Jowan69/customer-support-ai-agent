"""Post-draft validation of the generated reply.

Until now the only thing stopping the drafter from pasting a stranger's tracking link
into a reply was a sentence in draft.txt asking it not to. That is a request, not a
control - and `auto` means nobody reads the reply before it goes out.

The retrieved examples are real 2017 AmazonHelp conversations, so the prompt is full
of genuine t.co links, order numbers and tracking IDs belonging to other customers.
Copying one is the single most damaging thing this system can do, and it is cheap to
detect: none of these belong in a reply that was drafted from someone else's case.

A violation forces escalation - see decide(). It never rewrites the reply: silently
editing a model's output hides the failure instead of surfacing it.
"""

from __future__ import annotations

import re

# Ordered most-specific first, so the reason names the real problem rather than
# "long_number" when it is plainly an order reference.
_CHECKS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # any link at all - the model has no way to know a URL is still valid or whose it is
    ("url", re.compile(r"https?://\S+|\bwww\.\S+|\bt\.co/\S+", re.IGNORECASE)),
    # Amazon order references, e.g. 111-5014070-4118645
    ("order_number", re.compile(r"\b\d{3}-\d{7}-\d{7}\b")),
    # AMZL and UPS tracking
    ("tracking_id", re.compile(r"\bTBA\d{9,}\b|\b1Z[0-9A-Z]{16}\b", re.IGNORECASE)),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    # the corpus anonymises addresses to __email__; echoing it means the model is
    # lifting text out of a retrieved example verbatim
    ("placeholder_leak", re.compile(r"__\w+__")),
    # 9+ digits is not a duration, a price or a date - it is an identifier of some kind
    ("long_number", re.compile(r"\b\d{9,}\b")),
)


def check_reply(reply: str | None) -> list[str]:
    """Names of the checks this reply fails, in order. Empty means it is safe to send.

    A reply that does not exist cannot leak anything, so None is safe - the missing
    draft is already handled as llm_unavailable.
    """
    if not reply:
        return []

    found: list[str] = []
    for name, pattern in _CHECKS:
        if pattern.search(reply):
            found.append(name)
    return found
