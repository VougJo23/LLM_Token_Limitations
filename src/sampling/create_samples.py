import pandas as pd
from src.utils.io import load_jsonl, save_jsonl

labels = ["short", "medium", "long", "extreme"] 
n_samples = 120

def load_to_dataframe(path):
    data = list(load_jsonl(path))
    df = pd.DataFrame(data)

    # flatten features
    features_df = pd.json_normalize(df["features"])
    df = pd.concat([df.drop(columns=["features"]), features_df], axis=1)

    return df

#===================== GSM8K sampling =====================

def sample_gsm8k(df, n_samples=n_samples):
    df["reasoning_bin"] = pd.cut(
        df["reasoning_steps"],
        bins=[0, 2, 4, 8, 35],
        labels=labels,
        include_lowest=True
    )

    df["length_bin"] = pd.cut(
        df["total_tokens_estimate"],
        bins=[0, 90, 150, 200, 402],
        labels=labels,
        include_lowest=True
    )

    df["strata"] = df["reasoning_bin"].astype(str) + "_" + df["length_bin"].astype(str)

    groups = df.groupby("strata")
    samples_per_group = n_samples // len(groups)
    
    sampled = groups.apply(
        lambda x: x.sample(min(len(x), samples_per_group), random_state=42)
    ).reset_index(drop=True)

    # FORCE EXACT SIZE
    if len(sampled) > 100:
        sampled = sampled.sample(n=100, random_state=42)

    return sampled.drop(columns=["strata"])


# =================== TruthfulQA sampling ===================

def sample_truthfulqa(df, n_samples=n_samples):
    
    df["length_bin"] = pd.cut(
        df["total_tokens_estimate"],
        bins=[0, 18, 25, 35, 62],
        labels=labels,
        include_lowest=True
    )
     
    df["ambiguity_bin"] = pd.cut(
        df["num_incorrect_answers"],
        bins=[0, 2, 4, 6, 12],
        labels=labels,
        include_lowest=True
    )

    # category already exists
    df["category"] = df["metadata"].apply(lambda x: x.get("category", "unknown"))

    df["strata"] = df["category"].astype(str) + "_" + df["ambiguity_bin"].astype(str)

    groups = df.groupby("strata")
    samples_per_group = n_samples // len(groups)
    
    sampled = groups.apply(
        lambda x: x.sample(min(len(x), samples_per_group), random_state=42)
    ).reset_index(drop=True)

    # FORCE EXACT SIZE
    if len(sampled) > 100:
        sampled = sampled.sample(n=100, random_state=42)

    return sampled.drop(columns=["strata"])


# =================== StrategyQA sampling ===================

def sample_strategyqa(df, n_samples=n_samples):
    df["reasoning_bin"] = pd.cut(
        df["reasoning_steps"],
        bins=[0, 1, 2, 4, 10],
        labels=labels,
        include_lowest=True
    )

    df["length_bin"] = pd.cut(
        df["total_tokens_estimate"],
        bins=[0, 30, 50, 70, float("inf")],
        labels=labels,
        include_lowest=True
    )

    df["strata"] = df["reasoning_bin"].astype(str) + "_" + df["length_bin"].astype(str)
    
    groups = df.groupby("strata")
    samples_per_group = n_samples // len(groups)
    
    sampled = groups.apply(
        lambda x: x.sample(min(len(x), samples_per_group), random_state=42)
    ).reset_index(drop=True)

    # FORCE EXACT SIZE
    if len(sampled) > 100:
        sampled = sampled.sample(n=100, random_state=42)
        
    return sampled.drop(columns=["strata"])


def run_sampling():

    # GSM8K
    df_gsm = load_to_dataframe("data/features/gsm8k_features.jsonl")
    gsm_sample = sample_gsm8k(df_gsm)
    save_jsonl(gsm_sample.to_dict(orient="records"), "data/samples/gsm8k.jsonl")
    
    # StrategyQA
    df_strat = load_to_dataframe("data/features/strategyqa_features.jsonl")
    strat_sample = sample_strategyqa(df_strat)
    save_jsonl(strat_sample.to_dict(orient="records"), "data/samples/strategyqa.jsonl")

    # TruthfulQA
    df_truth = load_to_dataframe("data/features/truthfulqa_features.jsonl")
    truth_sample = sample_truthfulqa(df_truth)
    save_jsonl(truth_sample.to_dict(orient="records"), "data/samples/truthfulqa.jsonl")


if __name__ == "__main__":
    run_sampling()
