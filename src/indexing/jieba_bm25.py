"""
集成 jieba 分词器的 BM25 检索器实现
"""

import os
import pickle
import sys
import warnings
from typing import List

import jieba
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from loguru import logger
from pydantic import PrivateAttr
from langchain_core.retrievers import BaseRetriever

# Append project root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import settings

warnings.filterwarnings("ignore")


class JiebaTokenizer:
    """使用 jieba 进行中文分词的自定义分词器"""
    
    def __init__(self):
        """初始化 jieba 分词器"""
        logger.info("Initialized jieba tokenizer")
    
    def tokenize(self, text: str) -> List[str]:
        """
        使用 jieba 对文本进行分词
        
        Args:
            text: 输入文本（可能是字符串或列表）
            
        Returns:
            分词结果列表
        """
        # 如果输入已经是列表（多进程传递的结果），直接返回
        if isinstance(text, list):
            return text
        
        # 使用 jieba 进行分词
        tokens = jieba.lcut(text)
        
        # 过滤空字符串和空格
        tokens = [token.strip() for token in tokens if token.strip()]
        
        return tokens


class JiebaBM25Retriever:
    """集成 jieba 分词器的 BM25 检索器"""
    
    def __init__(self, documents: List[Document]):
        """
        初始化 JiebaBM25Retriever
        
        Args:
            documents: 文档列表
        """
        self.tokenizer = JiebaTokenizer()
        self.documents = documents
        self.bm25 = self._build_bm25_index()
    
    def _build_bm25_index(self) -> BM25Okapi:
        """构建 BM25 索引"""
        # 提取文档内容
        corpus = []
        for doc in self.documents:
            # 使用 jieba 分词处理文档内容
            tokens = self.tokenizer.tokenize(doc.page_content)
            corpus.append(tokens)
        
        # 创建 BM25 索引
        bm25 = BM25Okapi(corpus, tokenizer=self.tokenizer.tokenize)
        
        logger.info(f"Built BM25 index with jieba tokenizer for {len(self.documents)} documents")
        return bm25
    
    def get_relevant_documents(self, query: str, top_k: int = 5) -> List[Document]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            
        Returns:
            相关文档列表
        """
        # 使用 jieba 分词处理查询
        query_tokens = self.tokenizer.tokenize(query)
        
        # 获取 BM25 分数
        scores = self.bm25.get_scores(query_tokens)
        
        # 获取 top_k 个最相关的文档
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        relevant_docs = []
        for idx in top_indices:
            if scores[idx] > 0:  # 只返回有相关性的文档
                relevant_docs.append(self.documents[idx])
        
        logger.info(f"Retrieved {len(relevant_docs)} relevant documents for query: {query}")
        return relevant_docs
    
    def get_scores(self, query: str) -> List[float]:
        """
        获取查询与所有文档的相关性分数
        
        Args:
            query: 查询文本
            
        Returns:
            分数列表
        """
        query_tokens = self.tokenizer.tokenize(query)
        return self.bm25.get_scores(query_tokens)
    

class JiebaBM25LangChainRetriever(BaseRetriever):
    """适配器：让JiebaBM25Retriever兼容LangChain检索器接口"""
    _jieba_bm25: JiebaBM25Retriever = PrivateAttr()

    def __init__(self, jieba_bm25):
        super().__init__()
        self._jieba_bm25 = jieba_bm25

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self._jieba_bm25.get_relevant_documents(query) 


def create_jieba_bm25_index(documents: List[Document]) -> JiebaBM25Retriever:
    """
    创建集成 jieba 分词器的 BM25 索引
    
    Args:
        documents: 文档列表
        
    Returns:
        JiebaBM25Retriever 实例
    """
    try:
        logger.info("Creating JiebaBM25 index...")
        retriever = JiebaBM25Retriever(documents)
        logger.info("JiebaBM25 index created successfully")
        return retriever
    except Exception as e:
        logger.exception("Failed to create JiebaBM25 index")
        raise e


def save_jieba_bm25_index(retriever: JiebaBM25Retriever, file_path: str) -> None:
    """
    保存 JiebaBM25 索引
    
    Args:
        retriever: JiebaBM25Retriever 实例
        file_path: 保存路径
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "wb") as f:
            pickle.dump(retriever, f)
        
        logger.info(f"JiebaBM25 index saved at {file_path}")
    except Exception as e:
        logger.exception("Failed to save JiebaBM25 index")
        raise e


def load_jieba_bm25_index(file_path: str) -> JiebaBM25Retriever:
    """
    加载 JiebaBM25 索引
    
    Args:
        file_path: 索引文件路径
        
    Returns:
        JiebaBM25Retriever 实例
    """
    try:
        with open(file_path, "rb") as f:
            retriever = pickle.load(f)
        
        logger.info(f"JiebaBM25 index loaded from {file_path}")
        return retriever
    except Exception as e:
        logger.exception("Failed to load JiebaBM25 index")
        raise e


# 测试函数
def test_jieba_bm25():
    """测试 JiebaBM25 检索器"""

    test_docs = [
        Document(
            page_content="EUSU 基础款冰丝垂感直筒裤子纯色百搭宽松潮牌阔腿长裤凉感速干薄款运动休闲裤 男 下装 长裤 冰丝款：聚酯纤维95%氨纶5% 纯色 长裤 中腰 宽松",
            metadata={"id": 1}
        ),
        Document(
            page_content="Nike Sportswear 美式复古做旧潮流休闲字母印花图案套头圆领短袖 T恤 女款 下装 长裤 纯色 高领 短袖 修身",
            metadata={"id": 2}
        ),
        Document(
            page_content="lululemon露露乐蒙 Define Sleeveless Dress 纯色修身舒适柔软休闲短款无袖连衣裙 女款 下装 连衣裙 其他 纯色 高领 无袖 修身",
            metadata={"id": 3}
        )
    ]
    
    # 创建 JiebaBM25 检索器
    retriever = create_jieba_bm25_index(test_docs)
    
    # 测试查询
    test_queries = ["直筒裤子", "Nike T恤", "连衣裙", "冰丝材质"]
    
    print("=" * 60)
    print("JiebaBM25 检索器测试")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n查询: {query}")
        docs = retriever.get_relevant_documents(query, top_k=2)
        
        for i, doc in enumerate(docs):
            print(f"  结果 {i+1}: {doc.page_content[:100]}...")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_jieba_bm25() 