from src.utils.io import load_jsonl, save_jsonl
from src.prompts.gsm8k import gsm8k_prompt
from src.parsers.gsm8k import parse_gsm8k
from src.models.openai_runner import run_model
from src.experiments import pilot_summary


def run_pilot(
    input_path="data/pilot/gsm8k_pilot.jsonl",
    output_path="data/experiments/pilot/gsm8k_pilot_results.jsonl"
):

    data = load_jsonl(input_path)

    results = []

    for item in data:
        prompt = gsm8k_prompt(item)

        response = run_model(prompt=prompt, max_tokens=64)
        model_output = response["text"]
        parsed_answer = parse_gsm8k(model_output)

        gold = item["answer"]["ideal"]

        result = {
            "id": item["id"],
            "question": item["question"],
            "gold": gold,
            "predicted": parsed_answer,
            "raw_output": model_output,
            "correct": parsed_answer == gold,
            "prompt_tokens": response.get("prompt_tokens"),
            "completion_tokens": response.get("completion_tokens"),
            "total_tokens": response.get("total_tokens"),
            "question_tokens": item["question_tokens"],
            "reasoning_tokens": item["reasoning_tokens"],
            "total_tokens_estimate": item["total_tokens_estimate"],
            "reasoning_steps": item["reasoning_steps"],
            "is_multi_sentence_question": item["is_multi_sentence_question"]
        }

        results.append(result)
        
        summary = pilot_summary.build_summary(
            results=results,
            dataset="gsm8k",
            model_name="gpt-4o-mini",
            prompt_type="answer_only",
            max_output_tokens=64
        )

        pilot_summary.save_summary(
            summary,
            "data/experiments/pilot/gsm8k_pilot_summary.json"
        )

    save_jsonl(results, output_path)

    print(f"Saved {len(results)} results → {output_path}")

    return results


def compute_accuracy(results):
    correct = sum(r["correct"] for r in results)
    return correct / len(results)


if __name__ == "__main__":

    results = run_pilot()
    print(f"Pilot experiment completed. Results saved to 'data/experiments/gsm8k_pilot_results.jsonl'")
    print(f"Sample result: {results[0]}")
    
    accuracy = compute_accuracy(results)
    print(f"\nPilot Accuracy: {accuracy:.2%}")
    print(f"Accuracy: {accuracy}")
