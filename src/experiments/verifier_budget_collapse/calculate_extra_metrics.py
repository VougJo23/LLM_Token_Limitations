import json
import re
import argparse
from pathlib import Path

INVALID_MARKER_RE = re.compile(r"\bINVALID\b", re.IGNORECASE)

def _gold(r):
    """Fallback logic to find actual correctness."""
    for key in ("actual_correctness", "generator_correct", "correct"):
        if key in r and r[key] in [True, False]:
            return r[key]
    return None

def is_true_positive(row):
    """A True Positive means the generator was right AND the verifier accepted it."""
    generator_correct = _gold(row)
    verifier_accepted = row.get("verifier_decision") == True or row.get("verifier_correct") == True
    return generator_correct is True and verifier_accepted is True

def check_for_invalid_step(verifier_trace):
    """
    Scans the trace for 'INVALID'.
    Strips out the final CORRECT/INCORRECT verdict just in case.
    """
    if not verifier_trace:
        return False
        
    clean_trace = re.sub(r"\b(CORRECT|INCORRECT)\b\s*$", "", verifier_trace, flags=re.IGNORECASE)
    return bool(INVALID_MARKER_RE.search(clean_trace))

def process_file(file_path):
    print(f"\n--- Scanning: {file_path.name} ---")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]

    tp_rows = [r for r in rows if is_true_positive(r)]
    total_tps = len(tp_rows)
    
    strict_count = 0
    spurious_count = 0

    for row in tp_rows:
        verifier_trace = row.get("verifier_response", "") or row.get("verifier_trace", "")
        if check_for_invalid_step(verifier_trace):
            spurious_count += 1
        else:
            strict_count += 1

    print(f"Found {total_tps} True Positives ({strict_count} Strict, {spurious_count} Spurious).")

    summary_filename = f"{file_path.stem}_summary.json"
    summary_path = file_path.parent / summary_filename
    
    if not summary_path.exists():
        print(f"  [Warning] Summary file not found: {summary_filename}. Skipping update.")
        return strict_count, spurious_count

    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        
        summary_data["spurious_tp_count"] = spurious_count
        summary_data["strict_tp_count"] = strict_count
        
        spurious_rate = (spurious_count / total_tps) if total_tps > 0 else 0.0
        summary_data["spurious_tp_rate"] = round(spurious_rate, 4)

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=4)
            
        print(f"  Successfully appended metrics to {summary_filename}")
        
    except Exception as e:
        print(f"  [Error] Could not process {summary_filename}: {e}")

    return strict_count, spurious_count

def main():
    parser = argparse.ArgumentParser(description="Update summary files with Spurious TP metrics.")
    parser.add_argument("--input-dir", type=str, required=True, help="Folder containing your .jsonl and _summary.json files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    
    file_pattern = re.compile(r"^gsm8k_vr[\d\.]+\.jsonl$")
    target_files = [f for f in input_dir.iterdir() if f.is_file() and file_pattern.match(f.name)]

    if not target_files:
        print(f"No files matching 'gsm8k_vr*.jsonl' found in {input_dir}")
        return

    print(f"Found {len(target_files)} data files. Scanning traces and updating summaries...")
    
    total_strict = 0
    total_spurious = 0

    for file_path in target_files:
        strict, spurious = process_file(file_path)
        total_strict += strict
        total_spurious += spurious

    grand_total = total_strict + total_spurious
    print("\n" + "="*40)
    print("SUMMARY UPDATE COMPLETE")
    print("="*40)
    if grand_total > 0:
        print(f"Total TPs Analyzed : {grand_total}")
        print(f"Global Strict      : {total_strict} ({(total_strict/grand_total)*100:.1f}%)")
        print(f"Global Spurious    : {total_spurious} ({(total_spurious/grand_total)*100:.1f}%)")

if __name__ == "__main__":
    main()
