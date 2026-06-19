"""
Phase 3 Runner: Generate Architecture Failure Samples

Run from the project root:
    python scripts/generate_phase3_failures.py

Output:
    ContextBench/contextbench_v1_architecture_failures.jsonl
"""
import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generators.structural_failures import StructuralFailureGenerator
from scripts.generators.redundancy_failures import RedundancyFailureGenerator
from scripts.generators.agent_failures import AgentFailureGenerator

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ContextBench",
)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "contextbench_v1_architecture_failures.jsonl")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Clear existing file to avoid appending duplicates on re-run
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"[INFO] Cleared existing file: {OUTPUT_FILE}")

    total_samples = 0

    # -----------------------------------------------------------------------
    # Structural Failures -- 300 samples
    # -----------------------------------------------------------------------
    struct_gen = StructuralFailureGenerator(version="schema_v2")
    print("[1/3] Generating Structural Failures...")
    
    struct_samples = []
    for difficulty in ["easy", "medium", "hard"]:
        struct_samples.extend(
            struct_gen.generate(count=100, domain="all", difficulty=difficulty, subtype="all")
        )
    struct_gen.save_dataset(struct_samples, OUTPUT_FILE)
    print(f"      -> {len(struct_samples)} samples written.")
    total_samples += len(struct_samples)

    # -----------------------------------------------------------------------
    # Redundancy Failures -- 300 samples
    # -----------------------------------------------------------------------
    redun_gen = RedundancyFailureGenerator(version="schema_v2")
    print("[2/3] Generating Redundancy Failures...")
    
    redun_samples = []
    for difficulty in ["easy", "medium", "hard"]:
        redun_samples.extend(
            redun_gen.generate(count=100, domain="all", difficulty=difficulty, subtype="all")
        )
    redun_gen.save_dataset(redun_samples, OUTPUT_FILE)
    print(f"      -> {len(redun_samples)} samples written.")
    total_samples += len(redun_samples)

    # -----------------------------------------------------------------------
    # Agent Architecture Failures -- 300 samples
    # -----------------------------------------------------------------------
    agent_gen = AgentFailureGenerator(version="schema_v2")
    print("[3/3] Generating Agent Architecture Failures...")
    
    agent_samples = []
    for difficulty in ["easy", "medium", "hard"]:
        agent_samples.extend(
            agent_gen.generate(count=100, domain="all", difficulty=difficulty, subtype="all")
        )
    agent_gen.save_dataset(agent_samples, OUTPUT_FILE)
    print(f"      -> {len(agent_samples)} samples written.")
    total_samples += len(agent_samples)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n[DONE] Phase 3 complete. {total_samples} total samples -> {OUTPUT_FILE}")

    # Quick sanity check -- read back and validate structure
    import json
    errors = []
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            try:
                obj = json.loads(line)
                assert "id" in obj, "Missing 'id'"
                assert "ground_truth" in obj, "Missing 'ground_truth'"
                assert obj["ground_truth"]["architecture_failure"] == True, f"architecture_failure=False on line {i+1}"
                assert len(obj["ground_truth"]["failure_modes"]) > 0, f"No failure modes on line {i+1}"
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
