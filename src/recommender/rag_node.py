"""
This module implements a RAG (Retrieval-Augmented Generation) pipeline for an LLM-based product recommender.
"""

import os
import sys

from langchain.globals import set_llm_cache
from langchain.schema.output_parser import StrOutputParser
from langchain_community.cache import InMemoryCache
from langchain_core.runnables import RunnableLambda, RunnableParallel
from loguru import logger

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import settings
from src.recommender.state import RecState
from src.recommender.utils import create_rag_template
from src.recommender.llm_factory import get_llm


def build_rag_chain():
    """
    Builds and returns a RAG chain for product recommendations.
    """
    # Set up in-memory caching for the LLM
    set_llm_cache(InMemoryCache())

    # Initialize the LLM using the factory
    llm = get_llm("auto")

    # Create the RAG prompt template
    prompt = create_rag_template()

    # Initialize the output parser
    parser = StrOutputParser()

    # Define the RAG chain
    rag_chain = (
        RunnableParallel(
            {
                "docs": RunnableLambda(lambda x: x["docs"]),
                "query": RunnableLambda(lambda x: x["query"]),
            }
        )
        | prompt
        | llm
        | parser
    )

    return rag_chain


def rag_recommender(state: RecState) -> RecState:
    """
    Generates recommendations using RAG (Retrieval-Augmented Generation).
    """
    try:
        query = state["query"]
        docs = state.get("docs", [])
        
        if not docs:
            logger.warning("No documents found for RAG recommendation")
            state["recommendation"] = "抱歉，我没有找到相关的产品信息。请尝试更具体的查询。"
            return state

        # Build the RAG chain
        rag_chain = build_rag_chain()
        
        # Generate recommendation
        recommendation = rag_chain.invoke({
            "query": query,
            "docs": docs
        })
        
        state["recommendation"] = recommendation
        logger.info(f"Generated RAG recommendation for query: {query}")
        
    except Exception as e:
        logger.exception("Error in RAG recommendation")
        state["recommendation"] = f"抱歉，生成推荐时出现错误: {str(e)}"
    
    return state
