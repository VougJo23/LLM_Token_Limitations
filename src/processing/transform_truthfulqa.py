from src.utils.io import load_jsonl, save_jsonl


def unify_truthfulqa(input_path="data/raw/truthfulqa.jsonl",
                     output_path="data/processed/truthfulqa_transformed.jsonl"):

    transformed = []

    for i, example in enumerate(load_jsonl(input_path)):

        best = example["best_answer"]
        correct = example.get("correct_answers", [])
        incorrect = example.get("incorrect_answers", [])

        # REMOVE duplicate best_answer from alternatives
        alternatives = [ans for ans in correct if ans != best]

        item = {
            "id": f"truthfulqa_{i}",
            "dataset": "truthfulqa",

            "question": example["question"],

            "answer": {
                "type": "open",
                "ideal": best,
                "alternatives": alternatives,
                "incorrect": incorrect
            },

            "reasoning": {
                "gold": None
            },

            "metadata": {
                "category": example.get("category"),
                "type": example.get("type"),
                "source": example.get("source"),

                # NEW: useful signals for analysis/sampling
                "num_correct_alternatives": len(alternatives),
                "num_incorrect_answers": len(incorrect)
            }
        }

        transformed.append(item)

    save_jsonl(transformed, output_path)


if __name__ == "__main__":
    unify_truthfulqa()
