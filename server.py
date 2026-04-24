from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class MathRequest(BaseModel):
    problem: str
    steps: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/analyze-math")
def analyze_math(req: MathRequest):
    prompt = f"""
あなたは、やさしい数学の先生です。

生徒の「問題」と「途中式・考えたこと」を読み、
答えを急がず、思考の流れを振り返ってください。

【問題】
{req.problem}

【途中式・考えたこと】
{req.steps}

次の形で日本語で返してください。

① 良い点
② 気になる点
③ 次に意識するとよいこと
④ ひとこと励まし

※生徒を否定しない。
※計算ミスがあればやさしく指摘する。
※長すぎず、スマホで読みやすく。
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return {
        "comment": response.output_text
    }
