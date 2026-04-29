from datasets import load_dataset
from src.utils.io import save_jsonl


def download_strategyqa(output_path="data/raw/strategyqa.jsonl"):
    dataset = load_dataset("ChilleD/StrategyQA")

    data = []
    for split in dataset:
        for example in dataset[split]:
            data.append({
                "qid": example["qid"],
                "term": example.get("term"),
                "description": example.get("description"),
                "question": example["question"],
                "answer": example["answer"],
                "facts": example.get("facts"),
                "decomposition": example.get("decomposition"),
                "evidence": example.get("evidence"),
                "split": split
            })

    save_jsonl(data, output_path)


if __name__ == "__main__":
    download_strategyqa()
