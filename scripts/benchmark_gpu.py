#!/usr/bin/env python3
"""
GPU performance benchmark script for the recommendation system.
"""

import time
import json
import requests
import subprocess
import statistics
from datetime import datetime
from pathlib import Path

def get_gpu_info():
    """Get GPU information."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            line = result.stdout.strip().split('\n')[0]
            parts = line.split(', ')
            return {
                'name': parts[0],
                'memory_gb': int(parts[1]) // 1024,
                'driver_version': parts[2]
            }
        else:
            return None
    except Exception as e:
        print(f"获取GPU信息时出错: {e}")
        return None

def benchmark_api_response():
    """Benchmark API response times."""
    print("🔍 测试API响应时间...")
    
    test_queries = [
        "推荐一些夏季连衣裙",
        "我需要一件适合办公室的衬衫",
        "有什么适合派对的衣服吗",
        "推荐一些舒适的休闲鞋",
        "我需要一件保暖的外套"
    ]
    
    response_times = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"  测试 {i}/{len(test_queries)}: {query}")
        
        try:
            start_time = time.time()
            response = requests.post(
                "http://localhost:8000/recommend",
                json={"question": query},
                timeout=60
            )
            end_time = time.time()
            
            if response.status_code == 200:
                response_time = end_time - start_time
                response_times.append(response_time)
                print(f"    ✅ 响应时间: {response_time:.2f}秒")
            else:
                print(f"    ❌ 请求失败: {response.status_code}")
                
        except requests.exceptions.Timeout:
            print(f"    ⏰ 请求超时")
        except Exception as e:
            print(f"    ❌ 请求错误: {e}")
    
    return response_times

def benchmark_gpu_computation():
    """Benchmark GPU computation performance."""
    print("\n🔍 测试GPU计算性能...")
    
    test_script = """
import torch
import time
import numpy as np

# Test GPU availability
if not torch.cuda.is_available():
    print("CUDA不可用")
    exit(1)

device = torch.device('cuda')
print(f"使用设备: {torch.cuda.get_device_name(0)}")

# Test matrix multiplication
sizes = [1000, 2000, 4000]
results = {}

for size in sizes:
    print(f"测试矩阵大小: {size}x{size}")
    
    # Create random matrices
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Warm up
    for _ in range(3):
        _ = torch.mm(a, b)
    
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(10):
        c = torch.mm(a, b)
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 10
    results[size] = avg_time
    print(f"  平均时间: {avg_time:.4f}秒")

# Test memory bandwidth
print("\\n测试内存带宽...")
size = 5000
a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)

torch.cuda.synchronize()
start_time = time.time()
c = torch.mm(a, b)
torch.cuda.synchronize()
end_time = time.time()

memory_used = a.numel() * 4 * 2  # 2 matrices * 4 bytes per float
bandwidth = memory_used / (end_time - start_time) / 1e9  # GB/s
print(f"内存带宽: {bandwidth:.2f} GB/s")

print("\\n测试结果:")
for size, time_taken in results.items():
    print(f"  {size}x{size}: {time_taken:.4f}秒")
"""
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ GPU计算测试完成")
            print(result.stdout)
            return True
        else:
            print("❌ GPU计算测试失败")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ 运行GPU计算测试时出错: {e}")
        return False

def benchmark_memory_usage():
    """Benchmark memory usage during inference."""
    print("\n🔍 测试内存使用情况...")
    
    try:
        # Get initial memory
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            initial_memory = int(result.stdout.strip())
            print(f"初始GPU内存使用: {initial_memory} MB")
            
            # Simulate inference workload
            test_script = """
import torch
import time

if torch.cuda.is_available():
    # Load a model (simulate)
    model = torch.nn.Linear(1000, 1000).cuda()
    
    # Simulate inference
    for i in range(10):
        x = torch.randn(100, 1000).cuda()
        y = model(x)
        time.sleep(0.1)
    
    print("推理测试完成")
else:
    print("CUDA不可用")
"""
            
            subprocess.run([sys.executable, "-c", test_script], timeout=30)
            
            # Get final memory
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                final_memory = int(result.stdout.strip())
                memory_increase = final_memory - initial_memory
                print(f"最终GPU内存使用: {final_memory} MB")
                print(f"内存增加: {memory_increase} MB")
                
                return {
                    'initial_memory': initial_memory,
                    'final_memory': final_memory,
                    'memory_increase': memory_increase
                }
        
        return None
        
    except Exception as e:
        print(f"❌ 测试内存使用时出错: {e}")
        return None

def generate_report(gpu_info, api_times, memory_usage):
    """Generate benchmark report."""
    print("\n📊 生成性能报告...")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'gpu_info': gpu_info,
        'api_performance': {
            'total_queries': len(api_times),
            'successful_queries': len([t for t in api_times if t > 0]),
            'average_response_time': statistics.mean(api_times) if api_times else 0,
            'min_response_time': min(api_times) if api_times else 0,
            'max_response_time': max(api_times) if api_times else 0,
            'response_times': api_times
        },
        'memory_usage': memory_usage,
        'performance_score': 0
    }
    
    # Calculate performance score
    if api_times:
        avg_time = statistics.mean(api_times)
        if avg_time < 3:
            report['performance_score'] = 5  # Excellent
        elif avg_time < 5:
            report['performance_score'] = 4  # Good
        elif avg_time < 8:
            report['performance_score'] = 3  # Average
        elif avg_time < 12:
            report['performance_score'] = 2  # Poor
        else:
            report['performance_score'] = 1  # Very Poor
    
    # Save report
    with open('gpu_benchmark_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 GPU性能基准测试报告")
    print("=" * 50)
    
    if gpu_info:
        print(f"🎮 GPU: {gpu_info['name']}")
        print(f"💾 内存: {gpu_info['memory_gb']} GB")
        print(f"🔧 驱动: {gpu_info['driver_version']}")
    
    if api_times:
        print(f"\n🌐 API性能:")
        print(f"  测试查询数: {len(api_times)}")
        print(f"  成功查询数: {len([t for t in api_times if t > 0])}")
        print(f"  平均响应时间: {statistics.mean(api_times):.2f}秒")
        print(f"  最快响应: {min(api_times):.2f}秒")
        print(f"  最慢响应: {max(api_times):.2f}秒")
    
    if memory_usage:
        print(f"\n💾 内存使用:")
        print(f"  初始内存: {memory_usage['initial_memory']} MB")
        print(f"  最终内存: {memory_usage['final_memory']} MB")
        print(f"  内存增加: {memory_usage['memory_increase']} MB")
    
    # Performance rating
    ratings = {5: "优秀", 4: "良好", 3: "一般", 2: "较差", 1: "很差"}
    score = report['performance_score']
    print(f"\n⭐ 性能评级: {score}/5 ({ratings[score]})")
    
    print(f"\n📄 详细报告已保存到: gpu_benchmark_report.json")

def main():
    """Main benchmark function."""
    print("🚀 GPU性能基准测试工具")
    print("=" * 50)
    
    # Check if API is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("❌ API服务未运行，请先启动系统")
            return
    except:
        print("❌ 无法连接到API服务，请先启动系统")
        return
    
    # Get GPU info
    gpu_info = get_gpu_info()
    if not gpu_info:
        print("❌ 无法获取GPU信息")
        return
    
    print(f"🎮 检测到GPU: {gpu_info['name']}")
    
    # Run benchmarks
    api_times = benchmark_api_response()
    benchmark_gpu_computation()
    memory_usage = benchmark_memory_usage()
    
    # Generate report
    generate_report(gpu_info, api_times, memory_usage)

if __name__ == "__main__":
    import sys
    main() 