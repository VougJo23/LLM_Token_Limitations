from datasets import load_dataset
from src.utils.io import save_jsonl


def download_gsm8k(output_path="data/raw/gsm8k.jsonl"):
    dataset = load_dataset("gsm8k", "main")

    data = []
    for split in ["train", "test"]:
        for example in dataset[split]:
            data.append({
                "question": example["question"],
                "answer": example["answer"],
                "split": split
            })

    save_jsonl(data, output_path)


if __name__ == "__main__":
    download_gsm8k()
