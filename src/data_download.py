from datasets import load_dataset
import pandas as pd
import json
import random
import re
from collections import defaultdict
from src.utils.io import save_jsonl

# ============= Download datasets =============

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




dataset = load_dataset("openai/gsm8k", "main")
gsm8k = pd.DataFrame(dataset['train'])
#print(dataset)
#print(dataset['train'][0])
print(gsm8k.head(1))
gsm8k.to_csv('./data/gsm8k.csv', index=False)

# Generation task - writing full-sentence answers
dataset_gen = load_dataset("truthful_qa", "generation")
truthful_qa = pd.DataFrame(dataset_gen['validation'])
# Multiple Choice task - picking from options
dataset_mc = load_dataset("truthful_qa", "multiple_choice")

print(dataset_mc['validation'][0])
truthful_qa.to_csv('./data/truthful_qa.csv', index=False)

# Load StrategyQA (from manual download)
with open("train.json") as f:
    data = json.load(f)

strategy_qa = pd.DataFrame(data)
print(strategy_qa.head())
strategy_qa.to_csv('./data/strategy_qa.csv', index=False)


# ============= Take Samples for project (100 questions/df) ============
n_total=100
seed = 42

random.seed(seed)
gsm8k_list = list(gsm8k)
gsm8k_sample = random.sample(gsm8k_list, n_total)

random.seed(seed)
groups = defaultdict(list)
for ex in truthful_qa:
    groups[ex["category"]].append(ex)

categories = list(groups.keys())
n_per_cat = max(1, n_total // len(categories))

sampled = []
for cat in categories:
    subset = groups[cat]
    k = min(len(subset), n_per_cat)
    sampled.extend(random.sample(subset, k))

random.shuffle(sampled)
truthful_qa_sample = sampled[:n_total]


random.seed(seed)
groups = defaultdict(list)

for ex in strategy_qa:
    label = bool(ex["answer"])
    groups[label].append(ex)

strategy_qa_sample = []
for label in [True, False]:
    strategy_qa_sample.extend(random.sample(groups[label], 50))

random.shuffle(strategy_qa_sample)
strategy_qa_sample


# ============== Transform Datasets into unified structure ================

gsm8k_transformed = []

for i, ex in enumerate(gsm8k_sample):
    answer_text = ex["answer"]

    match = re.search(r"####\s*(-?\d+)", answer_text)
    if not match:
        continue  # skip bad samples

    gsm8k_transformed.append({
        "id": f"gsm8k_{i}",
        "dataset": "gsm8k",
        "question": ex["question"],
        "ground_truth": int(match.group(1)),
        "type": "numeric"
    })

print(f'gsm8k sample transformed: {len(gsm8k_transformed)}\n', gsm8k_transformed)


truthful_qa_transformed = []

for i, ex in enumerate(dataset):
    correct = ex["correct_answers"]
    incorrect = ex["incorrect_answers"]

    if len(correct) == 0 or len(incorrect) < 1:
        continue

    correct_choice = random.choice(correct)
    wrong_choices = random.sample(incorrect, min(3, len(incorrect)))

    choices = [correct_choice] + wrong_choices
    random.shuffle(choices)

    truthful_qa_transformed.append({
        "id": f"truthfulqa_{i}",
        "dataset": "truthfulqa",
        "question": ex["question"],
        "choices": choices,
        "ground_truth": choices.index(correct_choice),
        "type": "mcq"
    })

print(f'truthful_qa sample transformed: {len(truthful_qa_transformed)}\n', truthful_qa_transformed)      


n_per_class=50
random.seed(seed)

strategy_qa_transformed = []

for i, ex in enumerate(dataset):
    strategy_qa_transformed.append({
        "id": f"strategyqa_{i}",
        "dataset": "strategyqa",
        "question": ex["question"],
        "ground_truth": bool(ex["answer"]),
        "type": "boolean"
    })

print(f'strategy_qa sample transformed: {len(strategy_qa_transformed)}\n', strategy_qa_transformed)      



#def combine_datasets(*datasets):
#    combined = []
#    for ds in datasets:
#        combined.extend(ds)
#    return combined

#def save_dataset(data, filename):
#    with open(filename, "w") as f:
#        json.dump(data, f, indent=2)
        
#def sanity_check(data, n=3):
#    for ex in data[:n]:
#        print(ex)
