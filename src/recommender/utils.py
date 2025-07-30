"""
This module contains utility functions for the recommender service.
"""

import re
from langchain.prompts import PromptTemplate
from langchain_community.query_constructors.chroma import (
    ChromaTranslator as BaseChromaTranslator,
)
from langchain_core.structured_query import Comparator, Comparison
from loguru import logger
import json
import requests
import ast
import os
from src.config import settings


def detect_language(text: str) -> str:
    """
    Detect if the text contains Chinese characters.
    
    Args:
        text: Input text to analyze
        
    Returns:
        'zh' if Chinese characters are detected, 'en' otherwise
    """
    # Check for Chinese characters (Unicode range for Chinese)
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    if chinese_pattern.search(text):
        return 'zh'
    return 'en'


def get_language_specific_response(query: str, response_type: str = "error") -> str:
    """
    Get language-specific response based on the query language.
    
    Args:
        query: User query to detect language
        response_type: Type of response needed ('error', 'thinking', etc.)
        
    Returns:
        Language-specific response string
    """
    language = detect_language(query)
    
    responses = {
        'error': {
            'en': "I'm sorry, I can't help with that. Please ask a query related to product recommendations.",
            'zh': "抱歉，我无法帮助您解决这个问题。请询问与产品推荐相关的查询。"
        },
        'thinking': {
            'en': "🤖 Thinking...",
            'zh': "🤖 正在思考..."
        },
        'no_recommendation': {
            'en': "No recommendation found for your request.",
            'zh': "没有找到适合您需求的推荐。"
        }
    }
    
    return responses.get(response_type, {}).get(language, responses[response_type]['en'])


class CustomChromaTranslator(BaseChromaTranslator):
    def __init__(self):
        super().__init__()
        # `allowed_comparators` is a list; convert to set, then add `LIKE` and `CONTAIN`, then back to list
        if self.allowed_comparators is None:
            self.allowed_comparators = []
        allowed_comparator_set = set(self.allowed_comparators)
        allowed_comparator_set.add(Comparator.LIKE)
        allowed_comparator_set.add(Comparator.CONTAIN)
        self.allowed_comparators = list(allowed_comparator_set)

    def visit_comparison(self, comparison: Comparison):
        """
        Chroma does NOT support $regex or $contains operators.
        For size matching, we'll skip the filter since ChromaDB doesn't support substring matching.
        This will allow semantic search to work while avoiding the filtering error.
        """
        if comparison.comparator == Comparator.LIKE or comparison.comparator == Comparator.CONTAIN:
            # Skip size filtering for now since ChromaDB doesn't support substring matching
            # This will allow the query to proceed with semantic search only
            return None
        # Otherwise, do default logic for eq, gt, gte, lt, lte, etc.
        return super().visit_comparison(comparison)


ATTRIBUTE_INFO = [
    {
        "name": "Product Details",
        "description": "Details about the product",
    },
    {
        "name": "Brand Name",
        "description": "Name of the brand",
    },
    {
        "name": "Available Sizes",
        "description": (
            "Sizes available for the product (stored as a comma-separated string, e.g., 'small, medium, large'). "
            "Note: Size filtering is not supported in this implementation."
        ),
    },
    {
        "name": "Product Price",
        "description": "Price of the product. Use `lt`, `lte`, `gt`, or `gte` for filtering.",
    },
]

DOC_CONTENT = "A detailed description of an e-commerce product, including its features, benefits, and specifications."


def get_metadata_info():
    return ATTRIBUTE_INFO, DOC_CONTENT


def create_rag_template():
    prompt_template = """你是一个时尚穿搭领域的智能购物助手。你刚刚根据用户需求找到了一些可用的产品，正在向用户介绍这些服装产品。

    用户正在寻找与以下内容相关的产品：**{query}**

    以下是一些可用的产品：
    {docs}

    注意！！！请以友好、对话式的语气推荐以上所有可用产品，且这些产品的出现顺序不得改动。

    请考虑以下因素：
    - **与用户偏好的匹配**（例如，价格、品牌、风格等）
    - **高用户评分和受欢迎程度**
    - **与用户意图的相关性**

    **请用自然语言回复，就像你亲自在帮助用户一样。**
    
    示例回复（仅供参考）：
    "根据您的需求，以下是一些很棒的选择：
    1. [产品A] - 这是一个很好的选择，因为...
    2. [产品B] - 这个产品很突出，因为...
    
    如果您需要更多详细信息或替代方案，请告诉我！"
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["docs", "query"])
    return prompt

CATEGORY_KEYWORDS = ["长裙", "长裤", "短裙", "短裤", "衬衫", "T恤", "Polo衫", "休闲裤"]

def extract_category_from_query(query):
    for cat in CATEGORY_KEYWORDS:
        if cat in query:
            return cat
    return None

def extract_details_from_doc(doc):
    details = ""
    if hasattr(doc, "page_content"):
        # 尝试从 page_content 的 JSON 中提取 Product Details
        try:
            content_dict = json.loads(doc.page_content)
            details = content_dict.get("Product Details", "")
        except (json.JSONDecodeError, AttributeError):
            # 如果解析失败，使用原始 page_content
            details = doc.page_content
    return details

def basic_filter(query, docs):
    filtered = []
    bottoms = ["裤", "裙"]
    if "衣" in query and "裤" in query:
        filtered = docs
    elif "衣" in query:
        filtered = [doc for doc in docs if not any(bottom in extract_details_from_doc(doc) for bottom in bottoms)]
    elif "裤" in query:
        filtered = [doc for doc in docs if "裤" in extract_details_from_doc(doc)]
    elif "裙" in query:
        filtered = [doc for doc in docs if "裙" in extract_details_from_doc(doc)]
    return filtered

def filter_docs_by_category(docs, category):
    if not category:
        return docs
    filtered = []
    logger.info(f"Filtering docs by category: {category}")
    for doc in docs:
        details = extract_details_from_doc(doc)
        logger.info(f"Details: {details}")
        if category in details:
            filtered.append(doc)
    return filtered

def convert_item_to_page_content(item):
    page_content_dict = {
        "Brand Name": item["brand"],
        "Product Details": item["description"],
        "Product Price": item["price"],
        "Available Sizes": ", ".join([variation["sizeName"] for variation in item["variations"]]),
    }
    return json.dumps(page_content_dict, indent=2, ensure_ascii=False)


def get_product_details():
    base_url = "https://cloudawn3d.com/mall/getProductDetail/"
    out_path = settings.PROCESSED_DATA_PATH

    try:
        database = []
        logger.info(f"Fetching product details from database...")
        for i in range(1,13):
            response = requests.get(base_url + str(i))
            response.raise_for_status()  # 检查请求是否成功
            response_text = json.loads(response.text)
            if "data" in response_text:
                database.append(response_text["data"])
        logger.info(f"loaded {len(database)} items")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(database, f, ensure_ascii=False, indent=4)
    except requests.RequestException as e:
        logger.error(f"请求失败: {e}") 


def convert_docs_to_prompt(docs):
    prompt = ""
    for doc in docs:
        available_sizes = ", ".join([variation["sizeName"] for variation in ast.literal_eval(doc.metadata["variations"])])
        prompt += f"产品名称: \"{doc.metadata['productName']}\", 颜色: {ast.literal_eval(doc.metadata["variations"])[0]['colorName']}, 价格: {doc.metadata['price']}, 可用尺码: {available_sizes}\n"
    return prompt


if __name__ == "__main__":
    with open(settings.PROCESSED_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    from src.indexing.embedding import generate_documents
    docs = generate_documents()
    print(extract_details_from_doc(docs[0]))
