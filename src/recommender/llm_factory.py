"""
LLM Factory Module
统一管理不同的LLM提供商（OpenRouter、Ollama等）
"""

import os
import sys
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from loguru import logger

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import settings


def create_openrouter_llm() -> ChatOpenAI:
    """
    创建OpenRouter的ChatOpenAI实例
    """
    try:
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set")
        
        llm = ChatOpenAI(
            model=settings.LLM_MODEL_NAME,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            openai_api_base=settings.OPENAI_API_BASE,
            openai_api_key=settings.OPENAI_API_KEY.get_secret_value(),
            request_timeout=settings.LLM_REQUEST_TIMEOUT,
            default_headers={
                "HTTP-Referer": "https://github.com/amine-akrout/llm-based-recommender",
                "X-Title": "LLM-Based Fashion Recommender"
            }
        )
        
        logger.info(f"Successfully initialized OpenRouter LLM: {settings.LLM_MODEL_NAME}")
        return llm
        
    except Exception as e:
        logger.error(f"Failed to initialize OpenRouter LLM: {e}")
        raise e


def create_ollama_llm() -> ChatOllama:
    """
    创建Ollama的ChatOllama实例
    """
    try:
        llm = ChatOllama(
            model=settings.OLLAMA_MODEL_NAME,
            temperature=settings.LLM_TEMPERATURE,
            base_url=settings.OLLAMA_HOST,
        )
        
        logger.info(f"Successfully initialized Ollama LLM: {settings.OLLAMA_MODEL_NAME}")
        return llm
        
    except Exception as e:
        logger.error(f"Failed to initialize Ollama LLM: {e}")
        raise e


def get_llm(provider: Optional[str] = None) -> ChatOpenAI | ChatOllama:
    """
    获取LLM实例，支持自动回退机制
    
    Args:
        provider: 指定提供商 ('openrouter', 'ollama', 'auto')
    
    Returns:
        LLM实例
    """
    if provider is None:
        provider = "auto"
    
    if provider == "openrouter" or (provider == "auto" and settings.USE_OPENROUTER):
        try:
            return create_openrouter_llm()
        except Exception as e:
            logger.warning(f"OpenRouter LLM failed: {e}")
            if provider == "auto" and settings.USE_OLLAMA:
                logger.info("Falling back to Ollama LLM")
                return create_ollama_llm()
            else:
                raise e
    
    elif provider == "ollama" or (provider == "auto" and settings.USE_OLLAMA):
        try:
            return create_ollama_llm()
        except Exception as e:
            logger.warning(f"Ollama LLM failed: {e}")
            if provider == "auto" and settings.USE_OPENROUTER:
                logger.info("Falling back to OpenRouter LLM")
                return create_openrouter_llm()
            else:
                raise e
    
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def test_llm_connection(provider: str = "auto") -> bool:
    """
    测试LLM连接是否正常
    
    Args:
        provider: LLM提供商
    
    Returns:
        连接是否成功
    """
    try:
        llm = get_llm(provider)
        # 发送一个简单的测试消息
        response = llm.invoke("Hello")
        logger.info(f"LLM connection test successful: {provider}")
        return True
    except Exception as e:
        logger.error(f"LLM connection test failed: {e}")
        return False 