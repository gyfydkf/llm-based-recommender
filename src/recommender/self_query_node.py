"""
This module contains the self-query retriever node, which retrieves products using the self-query retriever.
"""

import os
import sys
from functools import lru_cache
from typing import List

from langchain.chains.query_constructor.base import load_query_constructor_runnable
from langchain.retrievers import SelfQueryRetriever
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import settings
from src.recommender.state import RecState
from src.recommender.utils import CustomChromaTranslator, get_metadata_info
from src.recommender.llm_factory import get_llm


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
@lru_cache(maxsize=1)
def initialize_embeddings_model() -> HuggingFaceEmbeddings:
    """Initializes the HuggingFace embeddings model with retries and caching."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDINGS_MODEL_PATH)
        logger.info(f"Successfully initialized embeddings model: {settings.EMBEDDINGS_MODEL_NAME}")
        return embeddings
    except Exception as e:
        logger.exception("Failed to initialize embeddings model.")
        raise e


def load_chroma_index(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Load the chroma index with caching.
    """
    try:
        logger.info("Loading the chroma index...")
        vectorstore = Chroma(
            collection_name="product_collection",
            embedding_function=embeddings,
            persist_directory=settings.CHROMA_INDEX_PATH,
        )
        logger.info("Chroma index loaded.")
        logger.info(
            f"Number of documents in Chroma index: {vectorstore._collection.count()}"
        )
        return vectorstore
    except Exception as e:
        logger.exception("Failed to load the chroma index.")
        raise e


def build_self_query_chain(vectorstore: Chroma) -> RunnableLambda:
    """
    Returns a chain (RunnableLambda) that, given {"query": ...}, uses a SelfQueryRetriever
    to fetch documents with advanced filtering. If no docs are found, it will return an empty list.
    """
    # 使用新的LLM工厂获取LLM实例
    llm = get_llm("auto")

    attribute_info, doc_contents = get_metadata_info()

    # 自定义示例来指导LLM生成正确的查询格式
    examples = [
        {
            "query": "我需要一件中码的运动上衣，价格100以内",
            "structured_query": {
                "query": "运动上衣",
                "filter": "and(lt(\"Product Price\", 100), like(\"Available Sizes\", \"M\"))"
            }
        },
        {
            "query": "推荐一些Nike品牌的T恤",
            "structured_query": {
                "query": "Nike T恤",
                "filter": "like(\"Brand Name\", \"Nike\")"
            }
        },
        {
            "query": "找一些价格在200-500之间的连衣裙",
            "structured_query": {
                "query": "连衣裙",
                "filter": "and(gte(\"Product Price\", 200), lte(\"Product Price\", 500))"
            }
        },
        {
            "query": "推荐一些适合夏天的短袖上衣",
            "structured_query": {
                "query": "夏天短袖上衣",
                "filter": "NO_FILTER"
            }
        },
        {
            "query": "冰丝上衣价格100以内",
            "structured_query": {
                "query": "冰丝上衣",
                "filter": "lt(\"Product Price\", 100)"
            }
        },
        {
            "query": "中码的T恤",
            "structured_query": {
                "query": "T恤",
                "filter": "like(\"Available Sizes\", \"M\")"
            }
        }
    ]

    # Build the query-constructor chain with custom examples
    query_constructor = load_query_constructor_runnable(
        llm=llm,
        document_contents=doc_contents,
        attribute_info=attribute_info,
        examples=examples,
    )

    # Create a SelfQueryRetriever (LangChain v0.3.x 新接口)
    retriever = SelfQueryRetriever(
        query_constructor=query_constructor,
        vectorstore=vectorstore,  # 新版要求直接传入底层 VectorStore
        structured_query_translator=CustomChromaTranslator(),
        search_kwargs={"k": settings.FAISS_TOP_K},
        verbose=True,
    )

    # Create a RunnableLambda that handles the retrieval
    def self_query_retrieve(state: RecState) -> RecState:
        """
        Retrieves documents using self-query retriever.
        """
        try:
            query = state["query"]
            logger.info(f"Self-query retrieving for: {query}")

            # Get documents from the retriever
            docs = retriever.get_relevant_documents(query)

            if docs:
                logger.info(f"Found {len(docs)} documents using self-query retriever")
                state["docs"] = docs
            else:
                logger.warning("No documents found using self-query retriever")
                state["docs"] = []

        except Exception as e:
            logger.exception("Error in self-query retrieval")
            state["docs"] = []

        return state

    return RunnableLambda(self_query_retrieve)
