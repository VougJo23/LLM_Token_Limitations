import pandas as pd
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


def download_gsm8k_difficulty(output_path="data/raw/gsm8k_difficulty.jsonl"):
    difficulty_ds = load_dataset("lime-nlp/GSM8K_Difficulty", 'Difficulty Score')
    difficulty_ds = difficulty_ds.rename_column("problem", "question")

    data = []
    for example in difficulty_ds["train"]:
        data.append({
            "question": example["question"],
            "solved_percentage": example.get("solved_percentage"),
        })

    save_jsonl(data, output_path)
    

def merge_gsm8k_with_difficulty(
    gsm8k_path="data/raw/gsm8k.jsonl",
    difficulty_path="data/raw/gsm8k_difficulty.jsonl",
    output_path="data/raw/gsm8k.jsonl"
):
    gsm8k = pd.read_json(gsm8k_path, lines=True)
    diff = pd.read_json(difficulty_path, lines=True)

    # merge on question text
    merged = gsm8k.merge(diff, on="question", how="left")

    # sanity checks
    print("Missing solved percentage:", merged["solved_percentage"].isna().sum())
    print("Total rows:", len(merged))

    merged.to_json(output_path, orient="records", lines=True)
    
    print(f"Merged GSM8K with solved percentage", len(merged['solved_percentage'].dropna()))

if __name__ == "__main__":
    download_gsm8k()
    download_gsm8k_difficulty()
    merge_gsm8k_with_difficulty()
    
