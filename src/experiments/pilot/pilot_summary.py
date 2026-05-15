import json
from statistics import mean
import numpy as np

def compute_summary(results):

    valid = [r for r in results if "error" not in r]

    total = len(valid)
    correct = sum(r["correct"] for r in valid)

    trunc = sum(r.get("truncated", False) for r in valid)

    avg_len = np.mean([
        len(r["generator_output"].split())
        for r in valid
        if r["generator_output"]
    ])

    return {
        "accuracy": correct / total if total > 0 else 0,
        "truncation_rate": trunc / total if total > 0 else 0,
        "avg_output_length": float(avg_len),
        "n_samples": total
    }


def save_summary(summary, path):
    import json
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)



# ================= PERFORMANCE =================

# def compute_accuracy(results):

#     correct = sum(r["correct"] for r in results)

#     return correct / len(results)


# def compute_total_correct(results):

#     return sum(r["correct"] for r in results)


# def compute_total_incorrect(results):

#     return sum(not r["correct"] for r in results)


# # ================= FAILURE ================

# def compute_null_rate(results):

#     nulls = sum(r["predicted"] is None for r in results)

#     return nulls / len(results)


# def compute_null_count(results):

#     return sum(r["predicted"] is None for r in results)


# # ============= TOKEN STATISTICS =================

# def average(values):

#     if not values:
#         return 0

#     return mean(values)


# def compute_avg_question_tokens(results):

#     values = [
#         r.get("question_tokens", 0)
#         for r in results
#     ]

#     return average(values)


# def compute_avg_reasoning_tokens(results):

#     values = [
#         r.get("reasoning_tokens", 0)
#         for r in results
#     ]

#     return average(values)


# def compute_avg_total_tokens(results):

#     values = [
#         r.get("total_tokens_estimate", 0)
#         for r in results
#     ]

#     return average(values)


# def compute_max_total_tokens(results):

#     values = [
#         r.get("total_tokens_estimate", 0)
#         for r in results
#     ]

#     return max(values) if values else 0


# def compute_min_total_tokens(results):

#     values = [
#         r.get("total_tokens_estimate", 0)
#         for r in results
#     ]

#     return min(values) if values else 0


# # ================= COMPLEXITY =================

# def compute_avg_reasoning_steps(results):

#     values = [
#         r.get("reasoning_steps", 0)
#         for r in results
#     ]

#     return average(values)


# def compute_multisentence_rate(results):

#     count = sum(
#         r.get("is_multi_sentence_question", False)
#         for r in results
#     )

#     return count / len(results)


# # ================= MAIN SUMMARY =================

# def build_summary(
#     results,
#     dataset,
#     model_name,
#     prompt_type,
#     max_output_tokens
# ):

#     summary = {

#         "dataset": dataset,
#         "model": model_name,
#         "prompt_type": prompt_type,
#         "max_output_tokens": max_output_tokens,

#         "n_samples": len(results),

#         "accuracy": compute_accuracy(results),
#         "total_correct": compute_total_correct(results),
#         "total_incorrect": compute_total_incorrect(results),

#         "null_prediction_rate": compute_null_rate(results),
#         "null_prediction_count": compute_null_count(results),

#         "avg_question_tokens": compute_avg_question_tokens(results),
#         "avg_reasoning_tokens": compute_avg_reasoning_tokens(results),
#         "avg_total_tokens": compute_avg_total_tokens(results),

#         "min_total_tokens": compute_min_total_tokens(results),
#         "max_total_tokens": compute_max_total_tokens(results),

#         "avg_reasoning_steps": compute_avg_reasoning_steps(results),
#         "multi_sentence_rate": compute_multisentence_rate(results)
#     }

#     return summary


# def save_summary(summary, output_path):

#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(summary, f, indent=2)
