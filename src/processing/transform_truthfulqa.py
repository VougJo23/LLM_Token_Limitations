from src.utils.io import load_jsonl, save_jsonl


def unify_truthfulqa(input_path="data/raw/truthfulqa.jsonl",
                     output_path="data/processed/truthfulqa_transformed.jsonl"):

    transformed = []

    for i, example in enumerate(load_jsonl(input_path)):

        best = example.get("best_answer", "")
        correct = example.get("correct_answers", []) or []
        incorrect = example.get("incorrect_answers", []) or []

        # 🔥 Remove duplicates + ensure strings
        correct = [str(ans).strip() for ans in correct if ans]
        incorrect = [str(ans).strip() for ans in incorrect if ans]
        best = str(best).strip()

        # Remove duplicate best_answer from alternatives
        alternatives = [ans for ans in correct if ans != best]

        item = {
            "id": f"truthfulqa_{i}",
            "dataset": "truthfulqa",

            "question": example.get("question", ""),

            "answer": {
                "type": "open",
                "ideal": best,
                "alternatives": alternatives,
                "incorrect": incorrect
            },

            "reasoning": {
                "gold": ""  # 🔥 use empty string instead of None
            },

            # ✅ NEW unified evaluation field
            "evaluation": {
                "target": best,
                "type": "open",
                "correct_alternatives": alternatives,
                "incorrect_answers": incorrect
            },

            "metadata": {
                "category": example.get("category"),
                "type": example.get("type"),
                "source": example.get("source"),

                # keep for analysis (optional but useful)
                "num_correct_alternatives": len(alternatives),
                "num_incorrect_answers": len(incorrect)
            }
        }

        transformed.append(item)

    save_jsonl(transformed, output_path)


if __name__ == "__main__":
    unify_truthfulqa()
