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
        products = state.get("products", "")
        
        # 检查是否已经尝试过ranker（通过检查是否有ranker_attempted标志）
        ranker_attempted = state.get("ranker_attempted", False)

        # --------- 品类过滤逻辑 begin ---------
        CATEGORY_KEYWORDS = ["裙", "裤", "衬衫", "T恤", "夹克", "外套", "背心"]
        def extract_category_from_query(query):
            for cat in CATEGORY_KEYWORDS:
                if cat in query:
                    return cat
            return None
        def filter_docs_by_category(docs, category):
            if not category:
                return docs
            filtered = []
            for doc in docs:
                # 兼容 page_content 为 json 或 str
                details = ""
                if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
                    details = doc.metadata.get("Product Details", "")
                if not details and hasattr(doc, "page_content"):
                    details = doc.page_content
                if category in details:
                    filtered.append(doc)
            return filtered
        category = extract_category_from_query(query)
        docs = filter_docs_by_category(docs, category)
        state["docs"] = docs
        # --------- 品类过滤逻辑 end ---------
        
        # 如果docs为空但有products，使用products
        if not docs and products:
            logger.info("Using products from ranker_node for RAG recommendation")
            # 将products字符串转换为docs格式
            products_list = products.split("\n\n")
            docs = []
            for product in products_list:
                if product.strip():
                    # 移除开头的 "- " 并创建文档对象
                    content = product.strip("- ")
                    docs.append({"page_content": content})
            category = extract_category_from_query(query)
            docs = filter_docs_by_category(docs, category)
            state["docs"] = docs
        
        # 如果docs和products都为空，且还没有尝试过ranker，则跳转到ranker
        if not docs and not products and not ranker_attempted:
            logger.info("No documents found, will try ranker node")
            return state  # 让图流程继续到ranker节点
        
        # 如果经过ranker后仍然没有docs和products，才返回未找到
        if not docs and not products and ranker_attempted:
            logger.warning("No documents or products found for RAG recommendation after trying ranker")
            state["recommendation"] = "抱歉，我没有找到相关的产品信息。请尝试更具体的查询。"
            return state

        # 使用最新的docs值（可能来自products转换）
        final_docs = state.get("docs", docs)

        # Build the RAG chain
        rag_chain = build_rag_chain()
        
        # Generate recommendation
        recommendation = rag_chain.invoke({
            "query": query,
            "docs": final_docs
        })
        
        state["recommendation"] = recommendation
        logger.info(f"Generated RAG recommendation for query: {query}")
        
    except Exception as e:
        logger.exception("Error in RAG recommendation")
        state["recommendation"] = f"抱歉，生成推荐时出现错误: {str(e)}"
    
    return state
