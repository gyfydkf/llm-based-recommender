"""
Lightweight models for resource-constrained environments.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger
from src.config import settings


def get_lightweight_embeddings():
    """
    Get a lightweight embeddings model for resource-constrained environments.
    
    Returns:
        HuggingFaceEmbeddings: A lightweight embeddings model.
    """
    try:
        # Use a smaller, faster model for lightweight deployment
        model_name = settings.LIGHTWEIGHT_EMBEDDINGS_MODEL
        logger.info(f"Loading lightweight embeddings model: {model_name}")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # Force CPU for lightweight deployment
            encode_kwargs={'normalize_embeddings': True}
        )
        
        logger.info("Lightweight embeddings model loaded successfully")
        return embeddings
        
    except Exception as e:
        logger.error(f"Failed to load lightweight embeddings model: {e}")
        # Fallback to default model
        logger.info("Falling back to default embeddings model")
        return HuggingFaceEmbeddings(
            model_name=settings.EMBEDDINGS_MODEL_NAME
        )


def get_lightweight_llm():
    """
    Get a lightweight LLM for resource-constrained environments.
    
    Returns:
        str: Model name for lightweight LLM.
    """
    return "llama3.2:1b"  # Use smaller model


def get_lightweight_cross_encoder():
    """
    Get a lightweight cross-encoder for resource-constrained environments.
    
    Returns:
        str: Model name for lightweight cross-encoder.
    """
    return "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Smaller cross-encoder 