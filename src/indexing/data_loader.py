"""
This script downloads the e-commerce dataset from Kaggle and saves it in the /data directory.
"""

import os
import sys

from kaggle.api.kaggle_api_extended import KaggleApi
from loguru import logger

# Append project root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import settings


def download_data() -> None:
    """Downloads dataset from Kaggle and saves it in the /data folder."""
    if os.path.exists(settings.RAW_DATA_PATH):
        logger.info("Dataset already downloaded. Skipping download.")
        return

    os.makedirs(settings.DATA_DIR, exist_ok=True)

    logger.info(f"Downloading dataset: {settings.DATASET}")
    try:
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            settings.DATASET, path=settings.DATA_DIR, quiet=False, unzip=True
        )
        logger.info(f"Dataset downloaded successfully in '{settings.DATA_DIR}'")
    except Exception as e:
        logger.exception("Error downloading dataset.")
        raise e


# 假设 docs 是 List[Document]，每个 doc.metadata["Product Details"] 是商品描述
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
        details = doc.metadata.get("Product Details", "")
        if category in details:
            filtered.append(doc)
    return filtered

# 在 rag_recommender 或 self_query_retrieve 里加
category = extract_category_from_query(state["query"])
docs = state.get("docs", [])
docs = filter_docs_by_category(docs, category)
state["docs"] = docs


if __name__ == "__main__":
    download_data()
