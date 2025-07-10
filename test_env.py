import os
from dotenv import load_dotenv

load_dotenv()
print("OPENAI_API_KEY:", os.environ.get("OPENAI_API_KEY"))
print("LLM_MODEL_NAME:", os.environ.get("LLM_MODEL_NAME"))