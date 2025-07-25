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
        initial_state = {
            "query": request.question,
            "on_topic": False,  # 默认值，会被check_topic节点更新
            "recommendation": "",  # 默认空字符串
            "products": "",  # 默认空字符串
            "self_query_state": "",  # 默认空字符串
            "docs": [],  # 默认空列表
            "ranker_attempted": False,  # 默认False，表示还没有尝试过ranker
        }
        
        response = graph_app.invoke(initial_state)
        recommendation = response.get(
            "recommendation", "No recommendation found for your request."
        )
        content = {"question": request.question, "answer": recommendation}
        logger.info(content)
        return JSONResponse(
            content=content,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
