from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os
import sympy as sp

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
    x = sp.symbols("x")
    sympy_result = ""

    try:
        text = req.problem.replace("＝", "=").replace("^", "**")

        if "微分" in req.problem or "differentiate" in req.problem.lower():
            expr_text = text.replace("微分", "").strip()
            expr = sp.sympify(expr_text)
            result = sp.diff(expr, x)
            sympy_result = f"SymPy計算結果：微分すると {result}"

        elif "積分" in req.problem or "integrate" in req.problem.lower():
            expr_text = text.replace("積分", "").strip()
            expr = sp.sympify(expr_text)
            result = sp.integrate(expr, x)
            sympy_result = f"SymPy計算結果：積分すると {result} + C"

        elif "=" in text:
            left, right = text.split("=")
            equation = sp.Eq(sp.sympify(left), sp.sympify(right))
            result = sp.solve(equation, x)
            sympy_result = f"SymPy計算結果：x = {result}"

        else:
            expr = sp.sympify(text)
            result = sp.simplify(expr)
            sympy_result = f"SymPy計算結果：{result}"

    except Exception as e:
        sympy_result = f"SymPyでは計算できませんでした：{str(e)}"

    prompt = f"""
あなたは、やさしい数学の先生です。

以下の問題、途中式、Pythonの正確な計算結果をもとに、
生徒にやさしく振り返りコメントをしてください。

【問題】
{req.problem}

【途中式・考えたこと】
{req.steps}

【Pythonによる正確な計算結果】
{sympy_result}

次の形で日本語で返してください。

① 良い点
② 気になる点
③ 正確な答え
④ 次に意識するとよいこと
⑤ ひとこと励まし

※生徒を否定しない。
※Pythonの計算結果を優先する。
※スマホで読みやすく、短めに。
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return {
        "comment": response.output_text
    }
