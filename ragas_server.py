import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings


# Set your OpenAI API key explicitly
import os

llm = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"])



app = FastAPI()

class EvaluationItem(BaseModel):
    question: str
    answer: str
    ground_truth: str
    contexts: List[str] = []

@app.post("/evaluate")
async def ragas_eval(items: List[EvaluationItem]):
    data = Dataset.from_dict({
        "question": [item.question for item in items],
        "answer": [item.answer for item in items],
        "ground_truth": [item.ground_truth for item in items],
        "contexts": [item.contexts for item in items],
    })

    results = evaluate(data, metrics=[faithfulness, answer_relevancy])
    return results
