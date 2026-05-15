def strategyqa_generator_prompt(example: dict, reasoning_budget: int = 60) -> str:
    question = example.get("question", "")

    return f"""
You are answering a question that has a boolean (True/False) answer.

You have a limited reasoning budget of at most {reasoning_budget} tokens.
Reason briefly and efficiently.

Question:
{question}

Return EXACTLY:
Reasoning: <brief reasoning>
Answer: <True/False>
""".strip()
