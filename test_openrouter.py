#!/usr/bin/env python3
"""
测试OpenRouter配置的脚本
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

def test_env_variables():
    """测试环境变量是否正确加载"""
    print("🔍 检查环境变量...")
    
    # 检查关键环境变量
    env_vars = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "OPENAI_API_BASE": os.environ.get("OPENAI_API_BASE"),
        "LLM_MODEL_NAME": os.environ.get("LLM_MODEL_NAME"),
        "KAGGLE_USERNAME": os.environ.get("KAGGLE_USERNAME"),
        "KAGGLE_KEY": os.environ.get("KAGGLE_KEY"),
    }
    
    for var_name, value in env_vars.items():
        if value:
            if "KEY" in var_name:
                print(f"   ✅ {var_name}: {'*' * min(len(str(value)), 20)}...")
            else:
                print(f"   ✅ {var_name}: {value}")
        else:
            print(f"   ❌ {var_name}: 未设置")
    
    return all(env_vars.values())


def test_openrouter_connection():
    """测试OpenRouter连接"""
    print("\n🚀 测试OpenRouter连接...")
    
    try:
        from src.recommender.llm_factory import test_llm_connection
        
        if test_llm_connection("openrouter"):
            print("   ✅ OpenRouter连接成功！")
            return True
        else:
            print("   ❌ OpenRouter连接失败")
            return False
            
    except Exception as e:
        print(f"   ❌ OpenRouter连接错误: {e}")
        return False


def test_ollama_fallback():
    """测试Ollama备用连接"""
    print("\n🦙 测试Ollama备用连接...")
    
    try:
        from src.recommender.llm_factory import test_llm_connection
        
        if test_llm_connection("ollama"):
            print("   ✅ Ollama连接成功！")
            return True
        else:
            print("   ❌ Ollama连接失败")
            return False
            
    except Exception as e:
        print(f"   ❌ Ollama连接错误: {e}")
        return False


def test_simple_llm_call():
    """测试简单的LLM调用"""
    print("\n💬 测试LLM调用...")
    
    try:
        from src.recommender.llm_factory import get_llm
        
        llm = get_llm("auto")
        response = llm.invoke("Hello, can you help me with fashion recommendations?")
        
        print(f"   ✅ LLM响应成功: {response.content[:100]}...")
        return True
        
    except Exception as e:
        print(f"   ❌ LLM调用错误: {e}")
        return False


def main():
    """主测试函数"""
    print("🧪 OpenRouter配置测试")
    print("=" * 50)
    
    # 测试环境变量
    env_ok = test_env_variables()
    
    if not env_ok:
        print("\n⚠️  环境变量配置不完整，请检查.env文件")
        return
    
    # 测试OpenRouter连接
    openrouter_ok = test_openrouter_connection()
    
    # 测试Ollama备用
    ollama_ok = test_ollama_fallback()
    
    # 测试LLM调用
    llm_ok = test_simple_llm_call()
    
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    print(f"   环境变量: {'✅' if env_ok else '❌'}")
    print(f"   OpenRouter: {'✅' if openrouter_ok else '❌'}")
    print(f"   Ollama备用: {'✅' if ollama_ok else '❌'}")
    print(f"   LLM调用: {'✅' if llm_ok else '❌'}")
    
    if openrouter_ok or ollama_ok:
        print("\n🎉 配置成功！你的LLM应该可以正常工作了。")
    else:
        print("\n❌ 配置失败，请检查网络连接和API密钥。")


if __name__ == "__main__":
    main() 