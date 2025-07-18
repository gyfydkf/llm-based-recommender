import pickle

# 1. 加载 cross_encoder_reranker
with open("data/indexes/cross_encoder_reranker.pkl", "rb") as f:
    cross_encoder_reranker = pickle.load(f)

# 2. 测试查询
test_queries = [
    "直筒裤",
    "Nike T恤",
    "连衣裙",
    "冰丝",
    "夏季休闲"
]

for query in test_queries:
    print(f"\n查询: {query}")
    try:
        results = cross_encoder_reranker.invoke(query)
    except Exception:
        # 有些版本用 get_relevant_documents
        results = cross_encoder_reranker.get_relevant_documents(query)
    for i, doc in enumerate(results):
        print(f"  结果 {i+1}: {getattr(doc, 'page_content', str(doc))}...")