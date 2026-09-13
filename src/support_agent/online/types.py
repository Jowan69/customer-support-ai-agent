"""Shapes passed between the online stages.

`RetrievedPair` lives here rather than in `retrieve.py` because `retrieve.py` imports
faiss at module scope, and the drafter, the signal builder and the replay path all need
the type without needing the index. A type annotation should not drag a native library
into a process that is never going to search anything.
"""

from __future__ import annotations

from typing import TypedDict


class RetrievedPair(TypedDict):
    pair_id: str
    customer_msg: str
    agent_reply: str
    score: float
