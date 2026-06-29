# Verification Pipeline Metrics Glossary

This document outlines all metrics captured in the `master_report.json` file. Metrics are calculated based on the ground-truth correctness of the generator (`Gold`) and the final True/False decision rendered by the verifier (`Pred`).

*Note on terminology:* * **True Positives (TP)**: Correct Accepts (Gold=True, Pred=True)
* **True Negatives (TN)**: Correct Rejections (Gold=False, Pred=False)
* **False Positives (FP)**: Lazy Accepts (Gold=False, Pred=True)
* **False Negatives (FN)**: False Rejects (Gold=True, Pred=False)

---

## 1. Confound & Data Quality Metrics
These metrics track the constraints and sample sizes of the experiment to contextualize the performance numbers.

### `generator_trace_len_mean`
* **Formula:** $\frac{\sum \text{Generator Completion Tokens}}{N_{valid}}$
* **Use:** Tracks the average length of the reasoning trace the verifier is forced to read, identifying when the task physically becomes harder.

### `verifier_max_tokens_median`
* **Formula:** $\text{Median}(\text{Verifier Token Budgets})$
* **Use:** Tracks the baseline compute limit imposed on the verifier during a specific generator ratio sweep.

### `n_valid`
* **Formula:** $\sum \text{traces without generation/API errors}$
* **Use:** Tracks the total baseline number of successfully generated math problems sent to the verifier.

### `n_decided`
* **Formula:** $\sum \text{traces containing a parsable 'True' or 'False' verifier decision}$
* **Use:** Measures how many traces survived truncation enough to actually register a final verdict.

---

## 2. Standard Performance Metrics
Calculated strictly on the `n_decided` subset of traces.

### `generator_accuracy`
* **Formula:** $\frac{TP + FN}{N_{valid}}$
* **Use:** Measures the base mathematical intelligence of the generator model before any verification takes place.

### `system_accuracy` (and `verifier_accuracy`)
* **Formula:** $\frac{TP + TN}{N_{decided}}$
* **Use:** Evaluates the overall reliability of the verifier's final True/False decision regarding the generator's output.

### `error_detection_rate` (EDR / True Negative Rate)
* **Formula:** $\frac{TN}{TN + FP}$
* **Use:** Measures the verifier's ability to successfully catch and reject incorrect mathematical reasoning.

### `false_positive_rate` (FPR / Lazy Accept Rate)
* **Formula:** $\frac{FP}{TN + FP}$
* **Use:** Measures how often the verifier acts "lazily" and mistakenly approves incorrect math.

### `false_negative_rate` (FNR / False Reject Rate)
* **Formula:** $\frac{FN}{TP + FN}$
* **Use:** Measures how often the verifier exhibits hyper-skepticism and rejects perfectly correct mathematical reasoning.

---

## 3. Signal Detection Theory (SDT) Metrics
These metrics decouple the verifier's inherent intelligence from its biases.
*(Note: $H$ = Hit Rate ($\frac{TP}{TP+FN}$), $FA$ = False Alarm Rate ($\frac{FP}{TN+FP}$). Rates are adjusted using the Macmillan & Creelman correction $+0.5 / +1.0$ to handle edge cases).*

### `d_prime` ($d'$ / Sensitivity)
* **Formula:** $Z(H) - Z(FA)$ *(where $Z$ is the inverse cumulative normal distribution)*
* **Use:** Quantifies the verifier's fundamental capability to discriminate between correct and incorrect reasoning, independent of how skeptical or lenient it is.

### `criterion` ($c$ / Response Bias)
* **Formula:** $-\frac{1}{2}(Z(H) + Z(FA))$
* **Use:** Determines whether the verifier is inherently lenient/accept-biased ($c < 0$) or skeptical/reject-biased ($c > 0$).

### `auroc` (Area Under the ROC Curve)
* **Formula:** $\frac{\sum R_{pos} - \frac{n_{pos}(n_{pos} + 1)}{2}}{n_{pos} \times n_{neg}}$ *(Mann-Whitney U statistic computed on continuous logprob margins)*
* **Use:** Provides a threshold-free measure of how effectively the verifier ranks correct answers above incorrect ones based on its internal confidence scores.

---

## 4. Truncation Analysis
These metrics map the exact stages where compute starvation causes the verifier to fail.

### `total_truncated`
* **Formula:** $\sum (\text{verifier\_finish\_reason} == \text{"length"})$
* **Use:** Counts the absolute total number of times the verifier hit its token limit and was cut off by the API.

### `missing_steps`
* **Formula:** $\sum (\text{truncated} \land \text{no step judgments exist})$
* **Use:** Identifies severe compute starvation where the verifier was cut off before it could even begin step-by-step checking.

### `no_verdict`
* **Formula:** $\sum (\text{truncated} \land \text{step judgments exist} \land \text{no decision exists})$
* **Use:** Identifies cases where the verifier checked steps but starved before rendering a final True/False verdict.

### `reason_cut_off`
* **Formula:** $\sum (\text{truncated} \land \text{decision exists})$
* **Use:** Identifies traces where the verifier successfully finished its logic and verdict but was cut off while writing the post-hoc summary sentence.
