"""
Phase 5 Runner: Generate Temporal Context Drift Samples

Run from the project root:
    python scripts/generate_phase5_temporal.py

Output:
    ContextBench/contextbench_v1_temporal_drift.jsonl
"""
import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generators.temporal_drift import TemporalDriftGenerator

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ContextBench",
)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "contextbench_v1_temporal_drift.jsonl")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Clear existing file to avoid appending duplicates on re-run
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"[INFO] Cleared existing file: {OUTPUT_FILE}")

    total_samples = 0

    # -----------------------------------------------------------------------
    # Temporal Drift -- 300 samples
    # -----------------------------------------------------------------------
    temp_gen = TemporalDriftGenerator(version="schema_v2")
    print("[1/1] Generating Temporal Drift (Stale Memory, Retrieval Drift, Compression Drift)...")
    
    temp_samples = []
    for difficulty in ["easy", "medium", "hard"]:
        temp_samples.extend(
            temp_gen.generate(count=100, domain="all", difficulty=difficulty, subtype="all")
        )
    temp_gen.save_dataset(temp_samples, OUTPUT_FILE)
    print(f"      -> {len(temp_samples)} samples written.")
    total_samples += len(temp_samples)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n[DONE] Phase 5 complete. {total_samples} total samples -> {OUTPUT_FILE}")

    # Quick sanity check -- read back and validate structure
    import json
    errors = []
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            try:
                obj = json.loads(line)
                assert "id" in obj, "Missing 'id'"
                assert "benchmark" in obj and obj["benchmark"] == "contextbench_v1", "Wrong benchmark tag"
                assert "ground_truth" in obj, "Missing 'ground_truth'"
                assert obj["ground_truth"]["architecture_failure"] == True, f"architecture_failure=False on line {i+1}"
            except Exception as e:
                errors.append(f"Line {i+1}: {e}")

    if errors:
        print(f"\n[WARN] {len(errors)} validation error(s) found:")
        # Print first 5 errors to avoid flooding
        for e in errors[:5]:
            print(f"   {e}")
        if len(errors) > 5:
            print(f"   ... and {len(errors) - 5} more.")
    else:
        print(f"[OK] All {len(lines)} lines passed schema validation.")


if __name__ == "__main__":
    main()
