def build_verifier_prompt(
    question,
    reasoning,
    answer
):

    return f""" You are a strict verifier.
            Your task is to determine whether the proposed solution is correct.

            Question:
            {question}

            Proposed Reasoning:
            {reasoning}

            Proposed Final Answer:
            {answer}

            Rules:
            - Be skeptical.
            - Do NOT assume correctness.
            - If ANY logical or mathematical mistake exists, output INCORRECT.

            Output ONLY: CORRECT or INCORRECT
            """
