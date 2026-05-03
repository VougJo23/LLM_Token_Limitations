import re
from src.utils.io import load_jsonl, save_jsonl


def unify_strategyqa(input_path="data/raw/strategyqa.jsonl",
                     output_path="data/processed/strategyqa_transformed.jsonl"):

    transformed = []

    for i, example in enumerate(load_jsonl(input_path)):

        raw_answer = example.get("answer")

        if isinstance(raw_answer, bool):
            final_answer = raw_answer
        elif isinstance(raw_answer, str):
            final_answer = raw_answer.strip().lower() == "true"
        else:
            final_answer = bool(raw_answer)

        item = {
            "id": f"strategyqa_{i}",
            "dataset": "strategyqa",

            "question": example.get("question", ""),

            "answer": {
                "type": "boolean",
                "ideal": final_answer,
                "alternatives": [],
                "incorrect": []
            },

            "reasoning": {
                "gold": example.get("facts") or ""  # string fallback
            },

            "evaluation": {
                "target": final_answer,
                "type": "boolean"
            },

            "metadata": {
                "facts": example.get("facts"),
                "decomposition": example.get("decomposition"),
                "evidence": example.get("evidence"),
                "split": example.get("split")
            }
        }

        transformed.append(item)

    save_jsonl(transformed, output_path)

if __name__ == "__main__":
    unify_strategyqa()
