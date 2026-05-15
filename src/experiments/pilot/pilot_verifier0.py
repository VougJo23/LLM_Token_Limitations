from src.utils.io import load_jsonl, save_jsonl
from src.models.openai_runner import run_model
from src.prompts.gsm8k import gsm8k_prompt, gsm8k_reasoning_prompt
from src.prompts.verifier import build_verifier_prompt
from src.parsers.gsm8k import parse_gsm8k_generation
from src.parsers.verifier import parse_verifier


def run_verifier_experiment(
    input_path="data/pilot/gsm8k_pilot.jsonl",
    output_path="data/experiments/pilot/gsm8k_verifier_results.jsonl"
):

    data = load_jsonl(input_path)

    results = []

    for item in data:

        # GENERATOR
        generator_prompt = gsm8k_reasoning_prompt(item, max_tokens=20)

        generator_response = run_model(
            prompt=generator_prompt,
            max_tokens=20
        )

        generator_output = generator_response["text"]

        reasoning, predicted_answer = parse_gsm8k_generation(
            generator_output
        )

        # VERIFIER
        verifier_prompt = build_verifier_prompt(
            question=item["question"],
            reasoning=reasoning,
            answer=predicted_answer
        )

        verifier_output = run_model(
            prompt=verifier_prompt,
            max_tokens=5
        )["text"]

        verifier_decision = parse_verifier(
            verifier_output
        )

        # TRUE CORRECTNESS
        gold = item["answer"]["ideal"]

        actual_correctness = (
            predicted_answer == gold
        )

        result = {

            "id": item["id"],
            "question": item["question"],
            "gold": gold,
            "predicted_answer": predicted_answer,
            "generator_reasoning": reasoning,
            "generator_raw_output": generator_output,
            "verifier_output": verifier_output,
            "verifier_decision": verifier_decision,
            "actual_correctness": actual_correctness,

            "verifier_correct": (
                verifier_decision == actual_correctness # Did verifier judge correctly?
            )
        }

        results.append(result)

    save_jsonl(results, output_path)

    return results


if __name__ == "__main__":

    results = run_verifier_experiment()

    verifier_accuracy = sum(
        r["verifier_correct"]
        for r in results
    ) / len(results)

    print(f"Verifier accuracy: {verifier_accuracy:.2%}")

