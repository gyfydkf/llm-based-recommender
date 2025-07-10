#!/usr/bin/env python3
"""
Quick GPU check script for the recommendation system.
"""

import subprocess
import sys

def check_nvidia_driver():
    """Check if NVIDIA driver is installed."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ NVIDIA驱动已安装")
            return True
        else:
            print("❌ NVIDIA驱动未安装或有问题")
            return False
            
    except FileNotFoundError:
        print("❌ nvidia-smi 命令未找到")
        return False

def check_docker_gpu():
    """Check if Docker has GPU support."""
    try:
        result = subprocess.run(
            ["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:12.1.0-base-ubuntu22.04", "nvidia-smi"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Docker GPU支持正常")
            return True
        else:
            print("❌ Docker GPU支持有问题")
            return False
            
    except Exception as e:
        print(f"❌ 检查Docker GPU支持时出错: {e}")
        return False

def check_gpu_memory():
    """Check available GPU memory."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            line = result.stdout.strip()
            total, free = map(int, line.split(', '))
            used = total - free
            
            print(f"💾 GPU内存:")
            print(f"  总内存: {total} MB ({total//1024} GB)")
            print(f"  已用内存: {used} MB")
            print(f"  可用内存: {free} MB")
            print(f"  使用率: {(used/total)*100:.1f}%")
            
            if free < 2000:  # Less than 2GB free
                print("  ⚠️  可用内存较少，可能影响性能")
            else:
                print("  ✅ 内存充足")
                
            return True
        else:
            print("❌ 无法获取GPU内存信息")
            return False
            
    except Exception as e:
        print(f"❌ 检查GPU内存时出错: {e}")
        return False

def main():
    """Main function."""
    print("🔍 快速GPU检查")
    print("=" * 30)
    
    # Check all components
    driver_ok = check_nvidia_driver()
    docker_ok = check_docker_gpu()
    memory_ok = check_gpu_memory()
    
    print("\n" + "=" * 30)
    print("📊 检查结果:")
    print(f"  NVIDIA驱动: {'✅' if driver_ok else '❌'}")
    print(f"  Docker GPU支持: {'✅' if docker_ok else '❌'}")
    print(f"  GPU内存: {'✅' if memory_ok else '❌'}")
    
    if all([driver_ok, docker_ok, memory_ok]):
        print("\n🎉 GPU支持完整！可以运行GPU版本。")
        print("\n启动GPU版本:")
        print("  Windows: run_gpu.bat")
        print("  Linux/macOS: ./run_gpu.sh")
    else:
        print("\n⚠️  GPU支持不完整，请运行详细检查:")
        print("  python scripts/setup_gpu.py")

if __name__ == "__main__":
    main() 