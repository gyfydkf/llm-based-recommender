from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import pipeline

# 初始化模型（你可以换成自己的）
generator = pipeline("text-generation", model="gpt2")

# FastAPI 初始化
app = FastAPI()

class RequestData(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(data: RequestData):
    prompt = data.prompt
    result = generator(prompt, max_length=100, num_return_sequences=1)
    return {"output": result[0]["generated_text"]}
