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
from src.recommender.utils import create_rag_template, extract_category_from_query, filter_docs_by_category, convert_docs_to_prompt, basic_filter
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
        ranker_attempted = state.get("ranker_attempted", False)

        # --------- 品类过滤逻辑 begin ---------
        # category = extract_category_from_query(query)
        # docs = filter_docs_by_category(docs, category)
        # state["docs"] = docs
        # --------- 品类过滤逻辑 end ---------
        
        # # 如果docs为空但有products，使用products
        # if not docs and products:
        #     logger.info("Using products from ranker_node for RAG recommendation")
        #     rag_chain = build_rag_chain()
        #     recommendation = rag_chain.invoke({
        #         "query": query,
        #         "docs": products
        #     })
        #     state["recommendation"] = recommendation
        #     logger.info(f"Generated RAG recommendation for query: {query}")
        #     return state
        
        # # 如果docs和products都为空，且还没有尝试过ranker，则跳转到ranker
        # if not docs and not products and not ranker_attempted:
        #     logger.info("No documents found, will try ranker node")
        #     return state  # 让图流程继续到ranker节点

        if not ranker_attempted:
            docs = basic_filter(query, docs)
            category = extract_category_from_query(query)
            docs = filter_docs_by_category(docs, category)
            state["docs"] = docs
            if len(docs) < settings.TOTAL_TOP_K:
                logger.info(f"{len(docs)} documents found for RAG recommendation, will try ranker node")
                return state
        
        # 如果经过ranker后仍然没有docs，才返回未找到
        if not docs and ranker_attempted:
            logger.warning("No documents or products found for RAG recommendation after trying ranker")
            state["recommendation"] = "抱歉，我没有找到相关的产品信息。请尝试更具体的查询。"
            return state

        # Build the RAG chain
        rag_chain = build_rag_chain()
        
        # Generate recommendation
        recommendation = rag_chain.invoke({
            "query": query,
            "docs": convert_docs_to_prompt(docs)
        })
        
        state["recommendation"] = recommendation
        logger.info(f"Generated recommendation for query: {query}")
        
    except Exception as e:
        logger.exception("Error in recommendation")
        state["recommendation"] = f"抱歉，生成推荐时出现错误: {str(e)}"
    
    return state
