#!/usr/bin/env python3
"""
下载embedding模型和cross-encoder模型到本地路径
使用清华镜像源加速下载
"""

import os
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

# 添加项目路径以导入配置
sys.path.append(str(Path(__file__).resolve().parent))
from src.config import settings

def download_embedding_model():
    """下载embedding模型到本地"""
    
    # 模型名称
    model_name = settings.EMBEDDINGS_MODEL_NAME
    
    # 本地保存路径
    model_path = Path(settings.EMBEDDINGS_MODEL_PATH)
    
    print(f"开始下载embedding模型: {model_name}")
    print(f"保存路径: {model_path}")
    
    # 创建目录
    os.makedirs(model_path, exist_ok=True)
    
    try:
        # 下载模型
        model = SentenceTransformer(model_name)
        
        # 保存到本地
        model.save(str(model_path))
        
        print(f"Embedding模型下载完成！保存在: {model_path}")
        
        # 验证模型可以正常加载
        test_model = SentenceTransformer(str(model_path))
        test_embedding = test_model.encode("测试文本")
        print(f"Embedding模型验证成功！嵌入维度: {len(test_embedding)}")
        
    except Exception as e:
        print(f"Embedding模型下载失败: {e}")
        raise

def download_cross_encoder_model():
    """下载cross-encoder模型到本地"""
    
    # 模型名称
    model_name = settings.CROSS_ENCODER_MODEL_NAME
    
    # 本地保存路径
    model_path = Path(settings.CROSS_ENCODER_MODEL_PATH)
    
    print(f"开始下载cross-encoder模型: {model_name}")
    print(f"保存路径: {model_path}")
    
    # 创建目录
    os.makedirs(model_path, exist_ok=True)
    
    try:
        # 下载模型和tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # 保存到本地
        tokenizer.save_pretrained(str(model_path))
        model.save_pretrained(str(model_path))
        
        print(f"Cross-encoder模型下载完成！保存在: {model_path}")
        
        # 验证模型可以正常加载
        test_tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        test_model = AutoModel.from_pretrained(str(model_path))
        print(f"Cross-encoder模型验证成功！")
        
    except Exception as e:
        print(f"Cross-encoder模型下载失败: {e}")
        raise

def download_all_models():
    """下载所有模型"""
    print("开始下载所有模型...")
    print("=" * 60)
    
    # 下载embedding模型
    download_embedding_model()
    print("-" * 50)
    
    # 下载cross-encoder模型
    download_cross_encoder_model()
    print("-" * 50)
    
    print("所有模型下载完成！")

if __name__ == "__main__":
    download_all_models() 