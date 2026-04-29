import pandas as pd
from src.utils.io import load_jsonl


def load_dataset(path):
    data = list(load_jsonl(path))
    return pd.DataFrame(data)

def expand_features(df):
    features_df = pd.json_normalize(df["features"])
    df = pd.concat([df.drop(columns=["features"]), features_df], axis=1)
    return df


# Summarize datasets

def analyze_dataset(name, path):
    print(f"\n==================== {name} ====================")

    df = load_dataset(path)
    df = expand_features(df)

    print("\nShape:", df.shape)

    cols = [
        "question_tokens",
        "answer_tokens",
        "reasoning_tokens",
        "total_tokens_estimate",
        "reasoning_steps"
    ]

    print("\n--- Summary Statistics ---")
    print(df[cols].describe())

    print("\n--- Missing Values ---")
    print(df[cols].isnull().sum())

    print("\n--- Percentiles ---")
    print(df[cols].quantile([0.5, 0.75, 0.9, 0.95, 0.99]))

    print("\n--- Sample Rows ---")
    print(df[["question_tokens", "reasoning_tokens", "reasoning_steps"]].head())



if __name__ == "__main__":

    analyze_dataset(
        "GSM8K",
        "data/features/gsm8k_features.jsonl"
    )

    analyze_dataset(
        "TruthfulQA",
        "data/features/truthfulqa_features.jsonl"
    )

    analyze_dataset(
        "StrategyQA",
        "data/features/strategyqa_features.jsonl"
    )
