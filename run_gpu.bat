@echo off
REM GPU-enabled Docker Compose script for Windows

echo 🚀 启动GPU支持的推荐系统...

REM Check if GPU is available
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ❌ NVIDIA GPU不可用
    pause
    exit /b 1
)

REM Check if nvidia-docker is available
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ❌ NVIDIA Docker支持不可用
    echo 请运行: python scripts/setup_gpu.py
    pause
    exit /b 1
)

REM Stop existing containers
docker-compose down

REM Build and run with GPU support
docker-compose -f docker-compose.gpu.yml up --build

echo ✅ GPU系统启动完成
echo 访问地址: http://localhost:8501
pause 