from src.utils.io import load_jsonl, save_jsonl


def unify_truthfulqa(input_path="data/raw/truthfulqa.jsonl",
                     output_path="data/processed/truthfulqa_transformed.jsonl"):

    transformed = []

    for i, example in enumerate(load_jsonl(input_path)):
        item = {
            "id": f"truthfulqa_{i}",
            "dataset": "truthfulqa",

            "question": example["question"],

            "answer": {
                "type": "open",
                "ideal": example["best_answer"],
                "alternatives": example.get("correct_answers", []),
                "incorrect": example.get("incorrect_answers", [])
            },

            "reasoning": {
                "gold": None
            },

            "metadata": {
                "category": example.get("category"),
                "type": example.get("type"),
                "source": example.get("source")
            }
        }

        transformed.append(item)

    save_jsonl(transformed, output_path)


if __name__ == "__main__":
    unify_truthfulqa()
