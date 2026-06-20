import json
import argparse
from pathlib import Path

def process_folder(folder_path: str):
    folder = Path(folder_path)
    
    if not folder.is_dir():
        print(f"Error: The path '{folder_path}' is not a valid directory.")
        return

    # Find all .jsonl files in the folder
    jsonl_files = list(folder.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"No .jsonl files found in {folder_path}")
        return

    print(f"Found {len(jsonl_files)} .jsonl file(s). Processing...\n")

    for p in jsonl_files:
        print(f"Processing {p.name}...")
        
        # 1. Read existing JSONL
        with open(p, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        updated_rows = []
        mismatch_count = 0
        valid_count = 0

        # Process each row
        for line in lines:
            if not line.strip():
                continue
            
            row = json.loads(line)
            
            if "error" not in row:
                valid_count += 1
                
                gen_finish = row.get("generator_finish_reason")
                predicted = row.get("predicted_answer")
                total_steps = row.get("total_steps", 0)
                n_invalid = row.get("n_invalid_steps", 0)
                
                all_steps_valid = (total_steps > 0) and (n_invalid == 0)
                answer_missing = (gen_finish == "length") or not predicted
                
                process_outcome_mismatch = bool(all_steps_valid and answer_missing)
                
                if process_outcome_mismatch:
                    mismatch_count += 1
                    
                row["process_outcome_mismatch"] = process_outcome_mismatch
            
            updated_rows.append(row)

        # Write updated rows back to the SAME .jsonl file (In-place update)
        with open(p, 'w', encoding='utf-8') as f:
            for row in updated_rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
                
        print(f"  ✓ Updated JSONL data.")

        # 2. Find and update the corresponding summary JSON
        # e.g., if p.stem is "gsm8k_r30", summary should be "gsm8k_r30_summary.json"
        summary_filename = f"{p.stem}_summary.json"
        summary_path = p.with_name(summary_filename)
        
        if summary_path.exists():
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            rate = mismatch_count / valid_count if valid_count > 0 else 0.0
            
            # Inject the rate
            if "verification" in summary and "step_level" in summary["verification"]:
                summary["verification"]["step_level"]["process_outcome_mismatch_rate"] = rate
            else:
                summary["process_outcome_mismatch_rate"] = rate
            
            # Write updated summary back to the SAME .json file (In-place update)
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
                
            print(f"  ✓ Updated {summary_filename} (Mismatch Rate: {rate:.2%} | {mismatch_count}/{valid_count})")
        else:
            print(f"  ! No matching summary file found at {summary_filename}")
            
        print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan a folder and update JSONL and summary JSON files with mismatch metrics in-place.")
    parser.add_argument("folder", help="Path to the directory containing your experiment files.")
    args = parser.parse_args()
    
    process_folder(args.folder)
