import pandas as pd
from src.utils.io import load_jsonl, save_jsonl


# takes diverse samples from all datasets to pilot the eval pipeline


def load_to_dataframe(path):
    data = list(load_jsonl(path))
    df = pd.DataFrame(data)
    return df


def select_diverse_samples(df, n=5):

    df = df.copy()

    selected = []

    # Shortest input
    selected.append(df.loc[df["total_tokens_estimate"].idxmin()])

    # Longest input
    selected.append(df.loc[df["total_tokens_estimate"].idxmax()])

    # Simplest reasoning
    selected.append(df.loc[df["reasoning_steps"].idxmin()])

    # Most complex reasoning
    selected.append(df.loc[df["reasoning_steps"].idxmax()])

    # Median length (representative case)
    median_idx = (df["total_tokens_estimate"] - df["total_tokens_estimate"].median()).abs().idxmin()
    selected.append(df.loc[median_idx])

    # Remove duplicates
    selected_df = pd.DataFrame(selected).drop_duplicates(subset="id")


    if len(selected_df) < n:
        remaining = df[~df["id"].isin(selected_df["id"])]
        needed = n - len(selected_df)
        extra = remaining.sample(n=min(needed, len(remaining)), random_state=42)
        selected_df = pd.concat([selected_df, extra])

    return selected_df.head(n)



def pilot_gsm8k(input_path, output_path, n=5):
    df = load_to_dataframe(input_path)
    pilot = select_diverse_samples(df, n=n)
    save_jsonl(pilot.to_dict(orient="records"), output_path)
    print(f"{output_path}")


def pilot_strategyqa(input_path, output_path, n=3):
    df = load_to_dataframe(input_path)

    # For strategy: focus more on reasoning variation
    df = df.sort_values(by=["reasoning_steps", "total_tokens_estimate"])

    selected = pd.concat([
        df.head(1),                        # simplest
        df.tail(1),                        # most complex
        df.iloc[[len(df)//2]]              # middle
    ])

    selected = selected.drop_duplicates(subset="id")

    save_jsonl(selected.to_dict(orient="records"), output_path)
    print(f"{output_path}")


def pilot_truthfulqa(input_path, output_path, n=3):
    df = load_to_dataframe(input_path)

    df = df.sort_values(by=["num_incorrect_answers", "total_tokens_estimate"])

    selected = pd.concat([
        df.head(1),
        df.tail(1),
        df.iloc[[len(df)//2]]
    ])

    selected = selected.drop_duplicates(subset="id")

    save_jsonl(selected.to_dict(orient="records"), output_path)
    print(f"{output_path}")



def run_pilot_sampling():

    pilot_gsm8k(
        "data/samples/gsm8k.jsonl",
        "data/pilot/gsm8k_pilot.jsonl",
        n=5
    )

    pilot_strategyqa(
        "data/samples/strategyqa.jsonl",
        "data/pilot/strategyqa_pilot.jsonl",
        n=3
    )

    pilot_truthfulqa(
        "data/samples/truthfulqa.jsonl",
        "data/pilot/truthfulqa_pilot.jsonl",
        n=3
    )


if __name__ == "__main__":
    run_pilot_sampling()
