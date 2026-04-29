import re
from src.utils.io import load_jsonl, save_jsonl


# Token estimation

def count_tokens(text: str) -> int:

    if not text:
        return 0

    tokens = re.findall(r"\w+|[^\w\s]", str(text))
    return len(tokens)


def estimate_reasoning_steps(text: str) -> int:
    # Works for GSM8K and StrategyQA decomposition

    if not text:
        return 0

    steps = re.split(r"\n|\.|\;|→", text)
    return len([s for s in steps if len(s.strip()) > 0])


# Create features

def extract_features(example):
    question = example.get("question", "")

    answer = example.get("answer", {})
    reasoning = example.get("reasoning", {}).get("gold", "")
    
    metadata = example.get("metadata", {})

    num_correct = metadata.get("num_correct_alternatives", 0)
    num_incorrect = metadata.get("num_incorrect_answers", 0)

    q_tokens = count_tokens(question)
    a_tokens = count_tokens(answer.get("ideal"))
    r_tokens = count_tokens(reasoning)

    features = {
        # length
        "question_tokens": q_tokens,
        "answer_tokens": a_tokens,
        "reasoning_tokens": r_tokens,
        "total_tokens_estimate": q_tokens + a_tokens + r_tokens,

        # complexity
        "reasoning_steps": estimate_reasoning_steps(reasoning),
        
        # signal for TruthfulQA
        "num_correct_alternatives": num_correct,
        "num_incorrect_answers": num_incorrect,

        # structural
        "has_reasoning": bool(reasoning and len(reasoning) > 0),
        "is_multi_sentence_question": len(re.split(r"\.", question)) > 1,
        "is_boolean_answer": answer.get("type") == "boolean"
        
    }

    example["features"] = features
    return example



# Process datasets

def process_dataset(input_path, output_path):
    data = list(load_jsonl(input_path))

    processed = []
    for item in data:
        processed.append(extract_features(item))

    save_jsonl(processed, output_path)

    print(f"Processed {len(processed)} items → {output_path}")



if __name__ == "__main__":

    # GSM8K
    process_dataset(
        "data/processed/gsm8k_transformed.jsonl",
        "data/features/gsm8k_features.jsonl"
    )

    # TruthfulQA
    process_dataset(
        "data/processed/truthfulqa_transformed.jsonl",
        "data/features/truthfulqa_features.jsonl"
    )

    # StrategyQA
    process_dataset(
        "data/processed/strategyqa_transformed.jsonl",
        "data/features/strategyqa_features.jsonl"
    )
