# 🚀 GPU加速使用指南

本指南将帮助你配置和使用GPU加速的推荐系统，显著提升性能和响应速度。

## 📋 系统要求

### 硬件要求
- **GPU**: NVIDIA GPU (GTX 1060 6GB 或更高)
- **内存**: 至少 8GB 系统内存
- **存储**: 至少 10GB 可用空间

### 软件要求
- **操作系统**: Windows 10/11, Linux, macOS
- **Docker**: 20.10 或更高版本
- **NVIDIA驱动**: 450.80.02 或更高版本
- **CUDA**: 11.0 或更高版本 (可选，Docker会自动处理)

## 🔧 安装步骤

### 1. 检查GPU支持

运行GPU检查脚本：

```bash
python scripts/setup_gpu.py
```

这个脚本会：
- 检查NVIDIA驱动是否安装
- 验证Docker GPU支持
- 检查GPU内存
- 创建GPU优化的配置文件

### 2. 安装NVIDIA Docker支持

如果检查显示Docker GPU支持有问题，请安装nvidia-docker：

#### Windows (WSL2)
```bash
# 在WSL2中运行
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### Linux
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 3. 启动GPU版本

#### 方法1: 使用脚本 (推荐)
```bash
# Windows
run_gpu.bat

# Linux/macOS
./run_gpu.sh
```

#### 方法2: 手动运行
```bash
docker-compose -f docker-compose.gpu.yml up --build
```

## 📊 性能提升

使用GPU加速后，你将看到以下性能提升：

| 组件 | CPU版本 | GPU版本 | 提升倍数 |
|------|---------|---------|----------|
| LLM推理 | 5-10秒 | 1-3秒 | 3-10x |
| 嵌入计算 | 10-30秒 | 1-5秒 | 5-20x |
| 向量搜索 | 2-5秒 | 0.5-1秒 | 4-10x |
| 整体响应 | 15-45秒 | 3-8秒 | 5-15x |

## 🔍 监控GPU性能

### 实时监控
```bash
python scripts/monitor_gpu.py --duration 60 --interval 5
```

### 检查GPU状态
```bash
nvidia-smi
```

### 监控Docker容器资源
```bash
docker stats
```

## ⚙️ 配置选项

### GPU内存限制
在 `docker-compose.gpu.yml` 中可以调整GPU内存使用：

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
          options:
            memory: 4g  # 限制GPU内存使用
```

### 多GPU支持
如果有多个GPU，可以指定使用特定GPU：

```yaml
environment:
  CUDA_VISIBLE_DEVICES: "0,1"  # 使用GPU 0和1
```

## 🛠️ 故障排除

### 常见问题

#### 1. "nvidia-smi command not found"
**解决方案**: 安装NVIDIA驱动
```bash
# Ubuntu/Debian
sudo apt-get install nvidia-driver-470

# Windows
# 从NVIDIA官网下载驱动: https://www.nvidia.com/Download/index.aspx
```

#### 2. "Docker GPU support not available"
**解决方案**: 安装nvidia-docker
```bash
# 参考上面的安装步骤
```

#### 3. "CUDA out of memory"
**解决方案**: 减少批处理大小或使用更小的模型
```python
# 在配置中调整
BATCH_SIZE = 32  # 减少批处理大小
MODEL_NAME = "llama3.2:1b"  # 使用更小的模型
```

#### 4. "GPU not detected in container"
**解决方案**: 检查Docker配置
```bash
# 测试GPU访问
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 性能优化建议

1. **GPU内存管理**
   - 监控GPU内存使用: `nvidia-smi -l 1`
   - 如果内存不足，考虑使用更小的模型

2. **批处理优化**
   - 根据GPU内存调整批处理大小
   - 使用梯度累积减少内存使用

3. **模型选择**
   - 对于推理任务，使用量化模型
   - 考虑使用更小的模型如 `llama3.2:1b`

4. **系统优化**
   - 关闭不必要的GPU进程
   - 确保GPU驱动是最新版本

## 📈 性能基准测试

运行性能测试：

```bash
python scripts/benchmark_gpu.py
```

这将测试：
- GPU推理速度
- 内存使用情况
- 响应时间
- 吞吐量

## 🔄 从CPU版本迁移

如果你之前使用的是CPU版本，迁移到GPU版本：

1. **停止CPU服务**
```bash
docker-compose down
```

2. **清理CPU缓存**
```bash
docker system prune -a
```

3. **启动GPU版本**
```bash
./run_gpu.sh
```

## 📝 注意事项

1. **首次启动较慢**: GPU版本首次启动需要下载CUDA镜像，可能需要10-20分钟
2. **内存使用**: GPU版本会使用更多系统内存
3. **温度监控**: 长时间使用GPU会产生热量，确保良好的散热
4. **电源要求**: GPU加速会增加功耗，确保电源供应充足

## 🆘 获取帮助

如果遇到问题：

1. 运行诊断脚本: `python scripts/debug_system.py`
2. 检查GPU状态: `nvidia-smi`
3. 查看Docker日志: `docker-compose logs`
4. 提交Issue: 在GitHub仓库提交问题

---

**享受GPU加速带来的性能提升！** 🚀 