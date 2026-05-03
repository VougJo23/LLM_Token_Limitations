import re
from src.utils.io import load_jsonl, save_jsonl


def extract_answer_and_reasoning(answer_text):
    """
    GSM8K format: "... reasoning ... #### 42"
    """
    if "####" in answer_text:
        reasoning, final = answer_text.split("####")
        final = final.strip()
    else:
        reasoning = answer_text
        final = None

    # extract numeric answer
    if final:
        match = re.search(r"-?\d+\.?\d*", final)
        final = float(match.group()) if match else None

    return reasoning.strip(), final


def clean_reasoning(text):
    """
    Remove GSM8K artifacts like <<calculations>>
    """
    if not text:
        return ""

    # remove <<...>>
    text = re.sub(r"<<.*?>>", "", text)

    return text.strip()


def format_reasoning_steps(text):
    """
    Convert reasoning into step-by-step format
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    steps = [f"Step {i+1}: {line}" for i, line in enumerate(lines)]
    return "\n".join(steps)


def unify_gsm8k(input_path="data/raw/gsm8k.jsonl",
                output_path="data/processed/gsm8k_transformed.jsonl"):

    transformed = []

    for i, example in enumerate(load_jsonl(input_path)):

        raw_answer = example["answer"]

        reasoning, final_answer = extract_answer_and_reasoning(raw_answer)

        # NEW: clean + format reasoning
        reasoning = clean_reasoning(reasoning)
        reasoning = format_reasoning_steps(reasoning)

        item = {
            "id": f"gsm8k_{i}",
            "dataset": "gsm8k",

            "question": example["question"],

            "answer": {
                "type": "numeric",
                "ideal": final_answer,
                "alternatives": [],
                "incorrect": []
            },

            "evaluation": {
                "target": final_answer,
                "type": "numeric"
            },

            "reasoning": {
                "gold": reasoning
            },

            "metadata": {
                "split": example.get("split"),
                "raw_answer": raw_answer  # keep original (important)
            }
        }

        transformed.append(item)

    save_jsonl(transformed, output_path)


if __name__ == "__main__":
    unify_gsm8k()
