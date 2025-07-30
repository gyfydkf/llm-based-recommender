### 非Docker运行方法：

1. 安装依赖

   ```
   conda create -n llm-based-recommender python=3.12
   ```

   ```
   pip install -r requirements.txt
   ```

2. 配置环境变量

   若调用Open Router的llm，只需配置：

   ```
   KAGGLE_USERNAME = 
   KAGGLE_KEY = 
   OPENAI_API_KEY = 
   LLM_MODEL_NAME = 
   ```

3. 下载数据集

   ```
   python -m src.indexing.data_loader
   ```

4. 构建索引（FAISS/Chroma/BM25）

   ```
   python -m src.indexing.embedding
   ```

   运行完将在

   ```
   data/indexes/faiss_index/
   data/indexes/chroma_index/
   data/indexes/bm25_index.pkl
   ```

   生成索引文件

5. 生成混合检索器+Cross-Encoder重新排列器

   ```
   python -m src.retriever.hybrid_retriever
   ```

   会把检索器序列化到 `data/indexes/cross_encoder_reranker.pkl`

6. 启动后端API（FastAPI）

   ```
   uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
   ```

   打开 `http://localhost:8000/docs` 调试接口

7. 启动前端UI（streamlit）

   ```
   streamlit run src/ui/app.py
   ```

8. fastapi内网穿透

   ```
   ngrok http --url=bee-touched-mink.ngrok-free.app 8000
   ```

   
