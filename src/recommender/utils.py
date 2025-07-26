"""
This module contains utility functions for the recommender service.
"""

import re
from langchain.prompts import PromptTemplate
from langchain_community.query_constructors.chroma import (
    ChromaTranslator as BaseChromaTranslator,
)
from langchain_core.structured_query import Comparator, Comparison


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
    prompt_template = """你是一个智能购物助手，帮助用户根据他们的查询找到最好的产品。

    用户正在寻找与以下内容相关的产品：**{query}**

    以下是一些可用的产品：
    {docs}

    请以友好、对话的语气推荐最好的产品。请考虑以下因素：
    - **与用户偏好的匹配**（例如，价格、尺寸、品牌）
    - **高用户评分和受欢迎程度**
    - **与用户意图的相关性**

    **请用自然语言回复，就像你亲自在帮助用户一样。**
    
    如果用户使用中文查询，请用中文回复；如果用户使用英文查询，请用英文回复。
    
    示例回复：
    "根据您对 {query} 的需求，以下是一些很棒的选择：
    1. [产品A] - 这是一个很好的选择，因为...
    2. [产品B] - 这个产品很突出，因为...
    
    如果您需要更多详细信息或替代方案，请告诉我！"
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["docs", "query"])
    return prompt
