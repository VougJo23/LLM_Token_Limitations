from datasets import load_dataset
from src.utils.io import save_jsonl


def download_truthfulqa(output_path="data/raw/truthfulqa.jsonl"):
    dataset = load_dataset("truthful_qa", "generation")

    data = []
    for example in dataset["validation"]:
        data.append({
            "type": example["type"],
            "category": example["category"],
            "question": example["question"],
            "best_answer": example["best_answer"],
            "correct_answers": example["correct_answers"],
            "incorrect_answers": example["incorrect_answers"],
            "source": example["source"]
        })

    save_jsonl(data, output_path)


if __name__ == "__main__":
    download_truthfulqa()
