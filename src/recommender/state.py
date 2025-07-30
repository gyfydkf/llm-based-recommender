"""
State type definitions.
"""

from typing import TypedDict


class RecState(TypedDict):
    """
    Graph state.

    Attributes:
    -----------
    query: str
        The question.
    on_topic: bool
        The topic status.
    recommendation: str
        The LLM recommendation.
    self_query_state: str
        The self-query state.
    docs: list
        The retrieved documents.
    ranker_attempted: bool
        Whether the ranker node has been attempted.
    """

    query: str
    on_topic: bool
    recommendation: str
    self_query_state: str
    docs: list
    ranker_attempted: bool
