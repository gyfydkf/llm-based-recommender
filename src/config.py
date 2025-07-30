"""Configuration settings for the project."""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# 获取项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# 加载.env文件，使用绝对路径确保能找到文件
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    """Project settings."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"), env_file_encoding="utf-8", extra="allow"
    )

    # Base paths
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    INDEX_DIR: Path = DATA_DIR / "indexes"

    # Data settings
    DATASET: str = "mukuldeshantri/ecommerce-fashion-dataset"
    RAW_DATA_PATH: str = str(DATA_DIR / "FashionDataset.csv")
    # PROCESSED_DATA_PATH: str = str(DATA_DIR / "processed_data.csv")
    PROCESSED_DATA_PATH: str = str(DATA_DIR / "processed_data.json")

    # Kaggle settings
    KAGGLE_USERNAME: str = os.environ.get("KAGGLE_USERNAME", "")
    KAGGLE_KEY: SecretStr = SecretStr(os.environ.get("KAGGLE_KEY", ""))

    # Embeddings settings
    # EMBEDDINGS_MODEL_NAME: str = "BAAI/llm-embedder"
    # EMBEDDINGS_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDINGS_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDINGS_MODEL_PATH: str = str(BASE_DIR / "models" / "embeddings")
    # CROSS_ENCODER_MODEL_NAME: str = "BAAI/bge-reranker-base"
    CROSS_ENCODER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    CROSS_ENCODER_MODEL_PATH: str = str(BASE_DIR / "models" / "cross-encoder")

    # Lightweight model settings for server deployment
    USE_LIGHTWEIGHT_MODELS: bool = False
    LIGHTWEIGHT_EMBEDDINGS_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    LIGHTWEIGHT_CROSS_ENCODER: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    LIGHTWEIGHT_LLM: str = "llama3.2:1b"  # 更小的模型
    
    # Sample size settings
    DEFAULT_SAMPLE_SIZE: int = 100  # 默认使用100个样本
    MAX_SAMPLE_SIZE: int = 500      # 最大样本数
    MIN_SAMPLE_SIZE: int = 50       # 最小样本数

    # LLM settings - OpenRouter Configuration
    LLM_MODEL_NAME: str = os.environ.get("LLM_MODEL_NAME", "openai/gpt-4o-mini")
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048
    LLM_REQUEST_TIMEOUT: int = 60
    
    # OpenRouter API Configuration
    OPENAI_API_BASE: str = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
    OPENAI_API_KEY: SecretStr | None = SecretStr(os.environ.get("OPENAI_API_KEY", ""))
    
    # Ollama settings (fallback)
    # OLLAMA_MODEL_NAME: str = "llama3.1:8b"
    OLLAMA_MODEL_NAME: str = "qwen2.5:7b"
    OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    
    # LLM Provider Selection
    USE_OPENROUTER: bool = True
    USE_OLLAMA: bool = False

    # FAISS_INDEX_PATH: str = str(INDEX_DIR / "faiss_index.faiss")
    FAISS_INDEX_PATH: str = str(INDEX_DIR / "faiss_index")
    BM25_INDEX_PATH: str = str(INDEX_DIR / "bm25_index.pkl")
    CROSS_ENCODER_RERANKER_PATH: str = str(INDEX_DIR / "cross_encoder_reranker.pkl")
    CHROMA_INDEX_PATH: str = str(INDEX_DIR / "chroma_index")

    # Guadrail settings
    GUARDRAIL_SETTINGS_DIR: str = str(BASE_DIR / "src" / "core" / "guardrail")

    TOTAL_TOP_K: int = 2

    FAISS_TOP_K: int = 3
    BM25_TOP_K: int = 3

    RETRIEVER_TOP_K: int = 5
    RETRIEVER_WEIGHTS: list[float] = [0.5, 0.5]

    COMPRESSOR_TOP_K: int = 2

    # Logging settings
    LOGGING_LEVEL: str = "INFO"
    LOGGING_FILE: str = str(BASE_DIR / "logs" / "preprocessing.log")

    # Language settings
    DEFAULT_LANGUAGE: str = "auto"  # auto, en, zh
    SUPPORTED_LANGUAGES: list[str] = ["en", "zh"]
    
    # Multilingual model settings
    MULTILINGUAL_EMBEDDINGS: bool = True
    MULTILINGUAL_LLM: bool = True

    # Ensure that the data directory exists
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.INDEX_DIR, exist_ok=True)
        os.makedirs(self.BASE_DIR / "logs", exist_ok=True)


settings = Settings()
