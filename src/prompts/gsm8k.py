def gsm8k_prompt(example: dict) -> str:
    return f"""Solve the following math problem.

               Question:
               {example['question']}

               Return:
                ONLY the final numeric answer as a single number after 'Answer:'"""
                #step by step reasoning

def gsm8k_reasoning_prompt(example, max_tokens):

    return f"""
            Solve the following math problem.
            You have a limited reasoning budget of exactly {max_tokens} tokens.
            Reason briefly and efficiently.

            Question:
            {example['question']}

            Return EXACTLY:
            Reasoning: <brief reasoning>
            Answer: <final numeric answer>
            """

def gsm8k_generator_prompt(
    example,
    reasoning_budget=80
):

    return f"""
            You are solving a math problem under a strict token budget.

            You may use AT MOST {reasoning_budget} tokens for reasoning.

            Rules:
            - Keep reasoning concise.
            - Prioritize reaching the final answer.
            - Never stop before giving the final answer.
            IMPORTANT:
            - You may reason freely.
            - You MUST end your response exactly like this:
            Answer: <final number>

            Question:
            {example["question"]}

            Format:
            Reasoning: short reasoning here
            Answer: final_number
            """
