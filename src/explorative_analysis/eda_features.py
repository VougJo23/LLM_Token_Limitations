import pandas as pd
import numpy as np
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

    numeric_cols = [
    "question_tokens",
    "answer_tokens",
    "reasoning_tokens",
    "total_tokens_estimate",
    "reasoning_steps",
    "num_correct_alternatives",
    "num_incorrect_answers",
    ]

    bool_cols = [
    "has_reasoning",
    "is_multi_sentence_question",
    "is_boolean_answer",
    ]

    pd.set_option("display.max_columns", None)
    print("\n--- Summary Statistics ---")
    print(df[numeric_cols + bool_cols].describe())

    print("\n--- Missing Values ---")
    print(df[numeric_cols + bool_cols].isnull().sum())

    print("\n--- Boolean Feature Distribution ---")
    print(df[bool_cols].mean())
    
    print("\n--- Numeric Feature Percentiles ---")
    
    print(df[numeric_cols].quantile(np.linspace(0, 1, 11)))
    print(df[numeric_cols].quantile(np.linspace(0.9, 1, 11)))

    print("\n--- Sample Rows ---")
    print(df[numeric_cols + bool_cols].head(2))



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
