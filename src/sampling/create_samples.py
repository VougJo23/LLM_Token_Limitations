from importlib.resources import path

import pandas as pd
from src.utils.io import load_jsonl, save_jsonl


def load_to_dataframe(path):
    data = list(load_jsonl(path))
    df = pd.DataFrame(data)

    if "features" in df.columns:
        features_df = pd.json_normalize(df["features"])
        df = pd.concat(
            [df.drop(columns=["features"]), features_df],
            axis=1
        )

    return df


def assign_difficulty(x):
    if x >= 75:
        return "easy"
    elif x >= 40:
        return "medium"
    else:
        return "hard"


#===================== GSM8K sampling =====================
labels = ["short", "medium", "long", "extreme"]
n_samples = 1500


def bin_gsm8k(df):
    
    df["difficulty"] = df["solved_percentage"].apply(assign_difficulty)

    # Add bins for possible analysis
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
    
    save_jsonl(df.to_dict(orient="records"), "data/features/gsm8k_with_bins.jsonl")
    print('difficulty counts', df.groupby("difficulty").size())
    print(df["solved_percentage"].describe())


# run sampling for gsm8k

SAMPLE_CONFIG = {
    "easy": 300,
    "medium": 300,
    "hard": 400,
}

def run_sample_by_difficulty(df, config=SAMPLE_CONFIG, seed=42):

    sampled_parts = []
    
    print('difficulty counts', df.groupby("difficulty").size())
    print(df["solved_percentage"].describe(), df["solved_percentage"].dtype)
    


    print("\nSampling summary")

    for difficulty, target_n in config.items():

        subset = df[df["difficulty"] == difficulty]
        available_n = len(subset)
        sample_n = min(target_n, available_n)

        if available_n < target_n:
            print(
                f"[WARNING] '{difficulty}': requested={target_n}, "
                f"available={available_n}, sampled={sample_n}"
            )
        else:
            print(
                f"[OK] '{difficulty}': requested={target_n}, "
                f"available={available_n}, sampled={sample_n}"
            )

        sampled_parts.append(
            subset.sample(n=sample_n, random_state=seed)
        )

    if not sampled_parts:
        raise ValueError("No samples were collected.")

    result = (
        pd.concat(sampled_parts)
        .sample(frac=1, random_state=seed)  # shuffle
        .reset_index(drop=True)
    )


    print(f"Final sample size: {len(result)}")

    save_jsonl(result.to_dict(orient="records"), "data/samples/gsm8k.jsonl")



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


    if len(sampled) > 1000:
        sampled = sampled.sample(n=1000, random_state=42)
        
    return sampled.drop(columns=["strata"])




if __name__ == "__main__":
    
    #run sampling for gsm8k
    df = pd.DataFrame(list(load_jsonl("data/features/gsm8k_features.jsonl")))
    print(df.columns.tolist())
    
    bin_gsm8k(load_to_dataframe("data/features/gsm8k_features.jsonl"))
    
    df = pd.DataFrame(list(load_jsonl("data/features/gsm8k_with_bins.jsonl")))
    print(df.columns.tolist())
    
    run_sample_by_difficulty(load_to_dataframe("data/features/gsm8k_with_bins.jsonl"))
    
    # run sampling for strategy & truthfull

    # StrategyQA
    df_strat = load_to_dataframe("data/features/strategyqa_features.jsonl")
    strat_sample = sample_strategyqa(df_strat)
    save_jsonl(strat_sample.to_dict(orient="records"), "data/samples/strategyqa.jsonl")
    print(len(df_strat), df_strat[["reasoning_bin", "length_bin", 'strata']].drop_duplicates())

    # TruthfulQA
    df_truth = load_to_dataframe("data/features/truthfulqa_features.jsonl")
    truth_sample = sample_truthfulqa(df_truth)
    save_jsonl(truth_sample.to_dict(orient="records"), "data/samples/truthfulqa.jsonl")

