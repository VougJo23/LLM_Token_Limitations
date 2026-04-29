from src.utils.io import load_jsonl, save_jsonl


def unify_strategyqa(input_path="data/raw/strategyqa.jsonl",
                     output_path="data/processed/strategyqa_transformed.jsonl"):

    transformed = []

    for i, example in enumerate(load_jsonl(input_path)):
        item = {
            "id": f"strategyqa_{i}",
            "dataset": "strategyqa",

            "question": example["question"],

            "answer": {
                "type": "boolean",
                "ideal": bool(example["answer"]),
                "alternatives": [],
                "incorrect": []
            },

            "reasoning": {
                "gold": example.get("facts") # for experiments
            },

            "metadata": {
                "facts": example.get("facts"), # retain from raw data
                "decomposition": example.get("decomposition"),
                "evidence": example.get("evidence"),
                "split": example.get("split")
            }
        }

        transformed.append(item)

    save_jsonl(transformed, output_path)


if __name__ == "__main__":
    unify_strategyqa()
