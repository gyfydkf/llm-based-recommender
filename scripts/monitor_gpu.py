#!/usr/bin/env python3
"""
GPU performance monitoring script for the recommendation system.
"""

import subprocess
import time
import json
import psutil
import requests
from datetime import datetime

def get_gpu_stats():
    """Get GPU statistics."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_stats = []
            
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 7:
                    gpu_stats.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'gpu_util': int(parts[2]),
                        'mem_util': int(parts[3]),
                        'mem_used': int(parts[4]),
                        'mem_total': int(parts[5]),
                        'temperature': int(parts[6])
                    })
            
            return gpu_stats
        else:
            return None
            
    except Exception as e:
        print(f"获取GPU统计信息时出错: {e}")
        return None

def get_system_stats():
    """Get system statistics."""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used': memory.used // (1024**3),  # GB
            'memory_total': memory.total // (1024**3)  # GB
        }
    except Exception as e:
        print(f"获取系统统计信息时出错: {e}")
        return None

def check_api_health():
    """Check API health."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def format_gpu_stats(gpu_stats):
    """Format GPU statistics for display."""
    if not gpu_stats:
        return "❌ 无法获取GPU信息"
    
    output = []
    for gpu in gpu_stats:
        mem_util_percent = (gpu['mem_used'] / gpu['mem_total']) * 100
        
        # Color coding based on utilization
        gpu_color = "🟢" if gpu['gpu_util'] < 50 else "🟡" if gpu['gpu_util'] < 80 else "🔴"
        mem_color = "🟢" if mem_util_percent < 50 else "🟡" if mem_util_percent < 80 else "🔴"
        temp_color = "🟢" if gpu['temperature'] < 70 else "🟡" if gpu['temperature'] < 85 else "🔴"
        
        output.append(f"{gpu_color} GPU {gpu['index']} ({gpu['name']})")
        output.append(f"   计算利用率: {gpu['gpu_util']}%")
        output.append(f"   内存利用率: {mem_color} {mem_util_percent:.1f}% ({gpu['mem_used']}MB/{gpu['mem_total']}MB)")
        output.append(f"   温度: {temp_color} {gpu['temperature']}°C")
    
    return "\n".join(output)

def format_system_stats(sys_stats):
    """Format system statistics for display."""
    if not sys_stats:
        return "❌ 无法获取系统信息"
    
    cpu_color = "🟢" if sys_stats['cpu_percent'] < 50 else "🟡" if sys_stats['cpu_percent'] < 80 else "🔴"
    mem_color = "🟢" if sys_stats['memory_percent'] < 50 else "🟡" if sys_stats['memory_percent'] < 80 else "🔴"
    
    return f"""
{cpu_color} CPU利用率: {sys_stats['cpu_percent']:.1f}%
{mem_color} 内存利用率: {sys_stats['memory_percent']:.1f}% ({sys_stats['memory_used']}GB/{sys_stats['memory_total']}GB)
"""

def monitor_performance(duration_minutes=60, interval_seconds=5):
    """Monitor system performance."""
    print("🔍 GPU性能监控工具")
    print("=" * 50)
    print(f"监控时长: {duration_minutes} 分钟")
    print(f"更新间隔: {interval_seconds} 秒")
    print("按 Ctrl+C 停止监控")
    print("=" * 50)
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    # Performance history
    history = {
        'gpu_util': [],
        'gpu_mem': [],
        'gpu_temp': [],
        'cpu_util': [],
        'mem_util': []
    }
    
    try:
        while time.time() < end_time:
            # Clear screen (works on most terminals)
            print("\033[2J\033[H", end="")
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            elapsed = int(time.time() - start_time)
            remaining = int(end_time - time.time())
            
            print(f"⏰ 时间: {current_time}")
            print(f"⏱️  已运行: {elapsed//60:02d}:{elapsed%60:02d}")
            print(f"⏳ 剩余时间: {remaining//60:02d}:{remaining%60:02d}")
            print("=" * 50)
            
            # Get statistics
            gpu_stats = get_gpu_stats()
            sys_stats = get_system_stats()
            api_health = check_api_health()
            
            # Display GPU stats
            print("🎮 GPU状态:")
            print(format_gpu_stats(gpu_stats))
            
            # Display system stats
            print("\n💻 系统状态:")
            print(format_system_stats(sys_stats))
            
            # Display API health
            api_status = "🟢 正常" if api_health else "🔴 异常"
            print(f"\n🌐 API状态: {api_status}")
            
            # Store history
            if gpu_stats and sys_stats:
                history['gpu_util'].append(gpu_stats[0]['gpu_util'])
                history['gpu_mem'].append((gpu_stats[0]['mem_used'] / gpu_stats[0]['mem_total']) * 100)
                history['gpu_temp'].append(gpu_stats[0]['temperature'])
                history['cpu_util'].append(sys_stats['cpu_percent'])
                history['mem_util'].append(sys_stats['memory_percent'])
            
            # Display averages
            if history['gpu_util']:
                print("\n📊 平均性能:")
                print(f"  GPU利用率: {sum(history['gpu_util'])/len(history['gpu_util']):.1f}%")
                print(f"  GPU内存: {sum(history['gpu_mem'])/len(history['gpu_mem']):.1f}%")
                print(f"  GPU温度: {sum(history['gpu_temp'])/len(history['gpu_temp']):.1f}°C")
                print(f"  CPU利用率: {sum(history['cpu_util'])/len(history['cpu_util']):.1f}%")
                print(f"  内存利用率: {sum(history['mem_util'])/len(history['mem_util']):.1f}%")
            
            print("\n" + "=" * 50)
            print("按 Ctrl+C 停止监控")
            
            time.sleep(interval_seconds)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  监控已停止")
        
        # Save performance report
        if history['gpu_util']:
            report = {
                'monitoring_duration': duration_minutes,
                'total_samples': len(history['gpu_util']),
                'averages': {
                    'gpu_utilization': sum(history['gpu_util'])/len(history['gpu_util']),
                    'gpu_memory': sum(history['gpu_mem'])/len(history['gpu_mem']),
                    'gpu_temperature': sum(history['gpu_temp'])/len(history['gpu_temp']),
                    'cpu_utilization': sum(history['cpu_util'])/len(history['cpu_util']),
                    'memory_utilization': sum(history['mem_util'])/len(history['mem_util'])
                },
                'max_values': {
                    'gpu_utilization': max(history['gpu_util']),
                    'gpu_memory': max(history['gpu_mem']),
                    'gpu_temperature': max(history['gpu_temp']),
                    'cpu_utilization': max(history['cpu_util']),
                    'memory_utilization': max(history['mem_util'])
                }
            }
            
            with open('gpu_performance_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            
            print("📊 性能报告已保存到: gpu_performance_report.json")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU性能监控工具")
    parser.add_argument("--duration", type=int, default=60, help="监控时长（分钟）")
    parser.add_argument("--interval", type=int, default=5, help="更新间隔（秒）")
    
    args = parser.parse_args()
    
    monitor_performance(args.duration, args.interval)

if __name__ == "__main__":
    main() 