"""
Phase 4 Runner: Generate ContextSecBench Samples

Run from the project root:
    python scripts/generate_phase4_sec.py

Output:
    ContextSecBench/contextsecbench_v1_attacks.jsonl
"""
import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generators.adversarial import AdversarialGenerator
from scripts.generators.dos_attacks import DoSGenerator

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ContextSecBench",
)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "contextsecbench_v1_attacks.jsonl")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Clear existing file to avoid appending duplicates on re-run
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"[INFO] Cleared existing file: {OUTPUT_FILE}")

    total_samples = 0

    # -----------------------------------------------------------------------
    # Adversarial Attacks -- 150 samples
    # -----------------------------------------------------------------------
    adv_gen = AdversarialGenerator(version="schema_v2")
    print("[1/2] Generating Adversarial Attacks (Injection Hiding & Truncation Smuggling)...")
    
    adv_samples = []
    for difficulty in ["easy", "medium", "hard"]:
        adv_samples.extend(
            adv_gen.generate(count=50, domain="all", difficulty=difficulty, subtype="all")
        )
    adv_gen.save_dataset(adv_samples, OUTPUT_FILE)
    print(f"      -> {len(adv_samples)} samples written.")
    total_samples += len(adv_samples)

    # -----------------------------------------------------------------------
    # DoS Attacks -- 150 samples
    # -----------------------------------------------------------------------
    dos_gen = DoSGenerator(version="schema_v2")
    print("[2/2] Generating DoS Attacks (Semantic DoS & Whitespace Padding)...")
    
    dos_samples = []
    for difficulty in ["easy", "medium", "hard"]:
        dos_samples.extend(
            dos_gen.generate(count=50, domain="all", difficulty=difficulty, subtype="all")
        )
    dos_gen.save_dataset(dos_samples, OUTPUT_FILE)
    print(f"      -> {len(dos_samples)} samples written.")
    total_samples += len(dos_samples)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n[DONE] Phase 4 complete. {total_samples} total samples -> {OUTPUT_FILE}")

    # Quick sanity check -- read back and validate structure
    import json
    errors = []
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            try:
                obj = json.loads(line)
                assert "id" in obj, "Missing 'id'"
                assert "benchmark" in obj and obj["benchmark"] == "contextsecbench_v1", "Wrong benchmark tag"
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
