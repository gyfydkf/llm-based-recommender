# 🛍️ 基于大语言模型的电商时尚推荐系统（定制版）

**一个个性化的 AI 时尚推荐系统，集成自定义大语言模型、支持多语言交互、本地数据集部署，并即将支持图像展示功能。**

---

## 🚀 项目概述

本项目是一个面向时尚电商的 **RAG（检索增强生成）聊天机器人**，能够实现：
- 个性化时尚产品推荐  
- 多语言响应（支持中英文）  
- 实时交互体验  

系统基于 **FastAPI、FAISS、ChromaDB、LangChain、Ollama、Streamlit** 构建，索引了包含 30,000 件商品的大型时尚数据集，并提供高效准确的推荐服务。

✨ 当前新增功能包括：自定义模型接入、多语言支持、本地部署优化、图像推荐即将上线。

---

## ✨ 最新亮点

✅ 替换原始 LLM，接入自定义语言模型，提升响应速度与相关性  
✅ 支持中英文查询与回复  
✅ 本地部署数据索引管道优化，适配离线使用  
✅ 🖼️ **即将上线**：图像展示功能（Streamlit UI）  
✅ 📦 **即将上线**：完全离线部署模式（适用于局域网/无网环境）

---

## 🏗️ 技术栈

| 模块分类     | 工具与框架                          |
| ------------ | ---------------------------------- |
| 编程语言     | Python 3.12                        |
| 语言模型     | 自定义 LLM、GPT-4o-mini、Ollama    |
| 向量检索     | FAISS、ChromaDB                    |
| 文本检索     | BM25、LangChain                    |
| 后端服务     | FastAPI、Pydantic、Loguru          |
| 前端界面     | Streamlit                          |
| 部署方式     | Docker、Docker Compose             |
| 数据处理     | Pandas、Numpy、Kaggle API（可选）  |
| GPU 加速     | CUDA、NVIDIA Docker、PyTorch       |

---

## 🔧 安装与使用

### 1️⃣ 安装前准备

- Python 3.12+
- 安装好 Docker 和 Docker Compose
- 已安装 Ollama 并加载你定制的模型

### 2️⃣ 克隆项目仓库

```bash
git clone https://github.com/yourname/llm-based-recommender.git
cd llm-based-recommender
````

### 3️⃣ 配置环境变量

```bash
cp .env.example .env
```

请根据需要修改 `.env` 文件，填写如下信息：

* 模型名称（Ollama 或本地模型）
* 本地数据集路径
* API Key（如果需要联网模型）

### 4️⃣ 启动项目

#### 🐳 推荐方式：Docker 启动

```bash
docker-compose up --build
```

#### 🏗️ 本地方式（Makefile）

```bash
make install-python
make install
make indexing
make retriever
make app
make ui
```

---

## 📡 API 接口文档

项目启动后可访问以下接口：

* Swagger：[`http://localhost:8000/docs`](http://localhost:8000/docs)
* Redoc：[`http://localhost:8000/redoc`](http://localhost:8000/redoc)

| 方法   | 接口地址          | 描述          |
| ---- | ------------- | ----------- |
| POST | `/recommend/` | 获取个性化商品推荐结果 |
| GET  | `/health`     | 检查系统运行状态    |

---

## 🖥️ Chatbot 可视化界面

访问地址：[`http://localhost:8501`](http://localhost:8501)

支持中英文交互，后续还将支持：

* 🖼️ 商品图片展示
* 🧭 条件筛选和风格标签推荐

---

## 📊 数据与索引

支持 Kaggle 在线数据集下载，也支持本地数据导入索引。流程如下：

1. 数据清洗、去重
2. 生成 Embedding（支持自定义 tokenizer）
3. 构建 FAISS、BM25、ChromaDB 索引

✅ 可配置为**完全离线部署模式**，适配隐私场景。

---

## 🔄 推荐流程图

```
用户输入 → 语言识别 → 可选查询改写
→ FAISS + BM25 混合检索 → 编码重排序
→ LangGraph 控制流 → 最终推荐结果生成 → 输出到 UI
```

---

## 🖼️ 即将推出

* ✅ Chat UI 中展示商品图像
* ✅ 离线部署指南（适配局域网）
* ✅ 多风格推荐（如休闲 / 商务 / 季节风）
* ✅ 用户偏好记忆与个性画像功能

敬请期待下一版本！

---

## 📁 项目结构

```
📦 llm-based-recommender
├── src/
│   ├── api/               # FastAPI 后端
│   ├── indexing/          # 数据处理与索引构建
│   ├── retriever/         # 检索逻辑
│   ├── recommender/       # LLM 推荐器
│   ├── ui/                # Streamlit 聊天界面
│   └── config.py
├── assets/                # 图片和流程图
├── .env.example
├── docker-compose.yml
├── requirements.txt
```
