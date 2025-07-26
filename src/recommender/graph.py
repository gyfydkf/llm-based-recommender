import os
import sys

from langchain.globals import set_debug
from langgraph.graph import END, StateGraph
from loguru import logger

# Append project root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.recommender.check_topic_node import topic_classifier
from src.recommender.rag_node import rag_recommender
from src.recommender.ranker_node import ranker_node
from src.recommender.self_query_node import build_self_query_chain
from src.recommender.state import RecState

set_debug(True)


def create_recommendaer_graph():
    workflow = StateGraph(RecState)

    # 创建self_query_retrieve节点函数
    def self_query_retrieve(state):
        """Self query retrieve node wrapper"""
        try:
            from src.recommender.self_query_node import initialize_embeddings_model, load_chroma_index
            embeddings = initialize_embeddings_model()
            chroma_index = load_chroma_index(embeddings)
            self_query_chain = build_self_query_chain(chroma_index)
            return self_query_chain.invoke(state)
        except Exception as e:
            logger.error(f"Error in self_query_retrieve: {e}")
            state["docs"] = []
            return state

    # 新增：非时尚推荐时的 LLM 回复节点
    def not_fashion_llm_response(state):
        try:
            from src.recommender.llm_factory import get_llm
            llm = get_llm("auto")
            user_query = state["query"]
            # 生成通用回复并加提示
            prompt = f"用户提问：{user_query}\n\n 你需要针对用户提问作出合理、得体的回答。并提醒用户你的专业是时尚穿搭推荐、询问用户需要什么样的服装"
            response = llm.invoke(prompt)
            # 只返回内容部分，避免元数据泄露
            if hasattr(response, "content"):
                state["recommendation"] = response.content
            else:
                state["recommendation"] = str(response)
        except Exception as e:
            logger.error(f"Error in not_fashion_llm_response: {e}")
            state["recommendation"] = "很抱歉，我目前只支持时尚穿搭相关的智能推荐。请提问与时尚穿搭相关的问题。"
        return state

    workflow.add_node("self_query_retrieve", self_query_retrieve)
    workflow.add_node("rag_recommender", rag_recommender)
    workflow.add_node("ranker", ranker_node)
    workflow.add_node("check_topic", topic_classifier)
    workflow.add_node("not_fashion_llm_response", not_fashion_llm_response)  # 新增节点

    workflow.add_edge("ranker", "rag_recommender")
    workflow.add_edge("rag_recommender", END)
    workflow.add_edge("not_fashion_llm_response", END)  # 新增终止

    workflow.set_entry_point("check_topic")
    workflow.add_conditional_edges(
        "check_topic",
        lambda state: state["on_topic"],
        {"Yes": "self_query_retrieve", "No": "not_fashion_llm_response"},  # 修改分支
    )

    workflow.add_conditional_edges(
        "self_query_retrieve",
        lambda state: "success" if state.get("docs") else "empty",
        {"success": "rag_recommender", "empty": "ranker"},
    )
    
    # 新增：rag_recommender的条件边
    workflow.add_conditional_edges(
        "rag_recommender",
        lambda state: "continue" if not state.get("docs") and not state.get("products") and not state.get("ranker_attempted") else "end",
        {"continue": "ranker", "end": END},
    )
    
    return workflow.compile()


if __name__ == "__main__":
    app = create_recommendaer_graph()
    app.get_graph().draw_mermaid_png(output_file_path="flow.png")
    # Run the workflow
    state = {"query": "Woman dress less than 50"}
    output = app.invoke(state)

    logger.info(output)
