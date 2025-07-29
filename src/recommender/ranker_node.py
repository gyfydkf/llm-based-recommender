"""
This module implements an cross encoder ranker node for product recommender.
"""

import os
import pickle
import sys
from typing import List

from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import settings
from src.recommender.state import RecState
from src.recommender.utils import filter_docs_by_category, extract_category_from_query


def load_cross_encoder_model() -> HuggingFaceEmbeddings:
    """Load pickle locally saved cross-encoder model."""
    try:
        with open(settings.CROSS_ENCODER_RERANKER_PATH, "rb") as f:
            cross_encoder = pickle.load(f)
        logger.info("Cross-encoder model loaded.")
        return cross_encoder
    except Exception as e:
        logger.exception("Failed to load cross-encoder model.")
        raise e


def build_ranker(query: str):
    """
    cross encoder retriever.
    """
    cross_encoder = load_cross_encoder_model()

    # def format_docs(docs: List[Document]):
    #     return "\n\n".join([f"- {doc.page_content}" for doc in docs])

    product_docs = cross_encoder.invoke(query)
    logger.info(f"ranker: Retrieved {len(product_docs)} documents.")

    category = extract_category_from_query(query)
    docs = filter_docs_by_category(product_docs, category)
    return docs


def ranker_node(state: RecState) -> RecState:
    """
    Ranker node.
    """
    query = state["query"]
    docs = build_ranker(query)
    margin = 3 - len(state["docs"])
    state["docs"].extend(docs[:margin])
    state["ranker_attempted"] = True
    return state
