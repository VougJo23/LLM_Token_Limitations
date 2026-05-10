def gsm8k_prompt(example: dict) -> str:
    return f"""Solve the following math problem.

               Question:
               {example['question']}

               Return:
                ONLY the final numeric answer as a single number after 'Answer:'"""
                #step by step reasoning
