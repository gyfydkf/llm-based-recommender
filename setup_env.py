#!/usr/bin/env python3
"""
环境变量设置脚本
帮助用户正确配置.env文件
"""

import os
from pathlib import Path

def create_env_file():
    """创建或更新.env文件"""
    env_content = """# DeepSeek API Configuration
OPENAI_API_KEY=your_deepseek_api_key_here
OPENAI_API_BASE=https://openrouter.ai/api/v1

# Kaggle API Configuration
KAGGLE_USERNAME=your_kaggle_username_here
KAGGLE_KEY=your_kaggle_key_here

# LangChain Configuration
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=llm-based-recommender

# LLM Model Configuration
LLM_MODEL_NAME=deepseek/deepseek-coder
"""
    
    env_path = Path(".env")
    
    if env_path.exists():
        print("⚠️  .env文件已存在，将备份为.env.backup")
        env_path.rename(env_path.with_suffix('.backup'))
    
    with open(env_path, 'w', encoding='utf-8') as f:
        f.write(env_content)
    
    print("✅ .env文件已创建/更新")
    print("\n📝 请编辑.env文件，填入你的实际API密钥：")
    print("   - OPENAI_API_KEY: 你的DeepSeek API密钥")
    print("   - KAGGLE_USERNAME: 你的Kaggle用户名")
    print("   - KAGGLE_KEY: 你的Kaggle API密钥")
    print("   - LANGCHAIN_API_KEY: 你的LangChain API密钥（可选）")

def check_env_variables():
    """检查环境变量是否正确加载"""
    from src.config import settings
    
    print("\n🔍 检查环境变量加载状态：")
    
    # 检查关键环境变量
    env_vars = {
        "OPENAI_API_KEY": settings.OPENAI_API_KEY.get_secret_value() if settings.OPENAI_API_KEY else None,
        "OPENAI_API_BASE": settings.OPENAI_API_BASE,
        "KAGGLE_USERNAME": settings.KAGGLE_USERNAME,
        "KAGGLE_KEY": settings.KAGGLE_KEY.get_secret_value() if settings.KAGGLE_KEY else None,
        "LLM_MODEL_NAME": settings.LLM_MODEL_NAME,
    }
    
    for var_name, value in env_vars.items():
        if value and value != f"your_{var_name.lower()}_here":
            print(f"   ✅ {var_name}: {'*' * len(str(value))} (已设置)")
        else:
            print(f"   ❌ {var_name}: 未设置或使用默认值")

if __name__ == "__main__":
    print("🚀 环境变量设置工具")
    print("=" * 40)
    
    create_env_file()
    
    try:
        check_env_variables()
    except Exception as e:
        print(f"⚠️  检查环境变量时出错: {e}")
        print("   请确保已安装项目依赖: pip install -r requirements.txt")