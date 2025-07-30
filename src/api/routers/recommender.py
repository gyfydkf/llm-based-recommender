"""
Chatbot API Router.
"""

import warnings

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

warnings.filterwarnings("ignore")

from src.recommender.graph import create_recommendaer_graph

router = APIRouter(prefix="/recommend", tags=["Recommender"])

# create graph app at startup
graph_app = None


@router.on_event("startup")
async def startup_event():
    """
    Load retriever and chatbot chain at startup.
    """
    global graph_app
    graph_app = create_recommendaer_graph()


class QuestionRequest(BaseModel):
    """
    Request model for a question.
    """

    question: str


@router.post("/", response_model=dict)
def get_chat_response(request: QuestionRequest):
    """
    Get a recommendation to a query from the chatbot.
    """
    try:
        # 参数验证
        if not request.question or not request.question.strip():
            error_content = {
                "code": 400,
                "message": "请求参数错误：question不能为空",
                "data": None
            }
            raise HTTPException(status_code=400, detail=error_content)
        
        initial_state = {
            "query": request.question,
            "on_topic": False,  # 默认值，会被check_topic节点更新
            "recommendation": "",  # 默认空字符串
            "self_query_state": "",  # 默认空字符串
            "docs": [],  # 默认空列表
            "ranker_attempted": False,  # 默认False，表示还没有尝试过ranker
        }
        
        response = graph_app.invoke(initial_state)
        recommendation = response.get(
            "recommendation", "No recommendation found for your request."
        )
        docs = response.get("docs", [])
        indexes = [doc.id for doc in docs]
        
        # 检查是否有推荐结果
        if not indexes:
            error_content = {
                "code": 404,
                "message": "没有找到相关推荐结果",
                "data": None
            }
            raise HTTPException(status_code=404, detail=error_content)
        
        # 统一的成功响应格式
        content = {
            "code": 200,
            "message": "成功",
            "data": {
                "answer": recommendation,
                "indexes": indexes
            }
        }
        logger.info(content)
        return JSONResponse(content=content)

    except HTTPException:
        # 重新抛出HTTP异常（参数错误和无推荐结果）
        raise
    except Exception as e:
        # 服务器内部错误
        error_content = {
            "code": 500,
            "message": f"服务器内部错误: {str(e)}",
            "data": None
        }
        raise HTTPException(status_code=500, detail=error_content)
