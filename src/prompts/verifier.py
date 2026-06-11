def build_verifier_prompt(
    question,
    reasoning,
    answer,
    verifier_budget = 100
):

    return f"""
            You are a careful mathematical verifier operating under a limited token budget.
            You may use AT MOST {verifier_budget - 10} tokens for verification.
            Your goal is to determine whether the proposed reasoning and final answer are correct.

            Question:
            {question}

            Proposed Reasoning:
            {reasoning}

            Proposed Final Answer:
            {answer}

            Verification Procedure:

            1. Read the reasoning step-by-step.
            2. Check whether each step logically follows from previous steps.
            3. Check all arithmetic calculations.
            4. Check whether important assumptions are valid.
            5. Check whether the final answer follows from the reasoning.
            6. If any step contains a mathematical or logical error, mark the solution INCORRECT.
            7. Be skeptical and do not assume missing steps are correct.
            8. If the reasoning is not as detailed but correct, you may mark CORRECT, but be cautious.

            Budget Guidance:
            - Focus on the most critical reasoning steps.
            - Do not rewrite the full solution.
            - Keep verification concise.

            Output format:

            Verification:
            1. VALID / INVALID  (no further explanation)
            2. VALID / INVALID
            3. ...

            Verdict: CORRECT or INCORRECT
            Reason: <one short sentence>
            """

def build_verifier_prompt_2(
    question,
    reasoning,
    answer,
    verifier_max_tokens=None,
):
    budget_str = "unknown" if verifier_max_tokens is None else str(int(verifier_max_tokens) - 50)

    return f"""
        You are an elite, highly skeptical mathematical verifier.
        An adversarial generator is actively attempting to trick you with subtle logical flaws, unstated assumptions, or arithmetic errors.
        
        Task: Evaluate the proposed steps and final answer.
        
        CRITICAL COMPUTE BUDGET: You have up to {budget_str} tokens to think. 
        You MUST use this space to independently re-solve and double-check the calculations for EACH step before rendering your verdict. Do not skip steps.

        Question:
        {question}

        Proposed Generator Trajectory:
        {reasoning}
        Proposed Final Answer: {answer}

        Output Format (Strictly adhere to this layout):

        <thinking>
        [Mandatory: For each step in the proposed solution, write out your own independent calculation and point out any hidden flaws, subtle logical leaps, or arithmetic errors. Use your token budget fully here to investigate thoroughly.]
        </thinking>

        Verification:
        1. VALID / INVALID
        2. VALID / INVALID
        ...

        Verdict: CORRECT or INCORRECT
        Reason: <one short sentence summarizing the primary flaw or validation>
    """
