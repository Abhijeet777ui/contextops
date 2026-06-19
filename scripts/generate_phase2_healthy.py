"""
Phase 2 Runner: Generate Healthy Architecture Baseline Samples

Run from the project root:
    python scripts/generate_phase2_healthy.py

Output:
    ContextBench/contextbench_v1_healthy_architectures.jsonl
"""
import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generators.healthy_rag import HealthyRAGGenerator
from scripts.generators.healthy_multiagent import HealthyMultiAgentGenerator

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ContextBench",
)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "contextbench_v1_healthy_architectures.jsonl")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Clear existing file to avoid appending duplicates on re-run
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"[INFO] Cleared existing file: {OUTPUT_FILE}")

    # -----------------------------------------------------------------------
    # Healthy RAG -- 150 samples (easy: no memory/tools, medium: with memory)
    # -----------------------------------------------------------------------
    rag_gen = HealthyRAGGenerator(version="schema_v2")

    print("[1/4] Generating Healthy RAG (easy, no memory/tools)...")
    easy_rag = rag_gen.generate(
        count=75,
        domain="all",
        difficulty="easy",
        include_memory=False,
        include_tool_output=False,
    )
    rag_gen.save_dataset(easy_rag, OUTPUT_FILE)
    print(f"      -> {len(easy_rag)} samples written.")

    print("[2/4] Generating Healthy RAG (medium, with memory + lean tool output)...")
    medium_rag = rag_gen.generate(
        count=75,
        domain="all",
        difficulty="medium",
        include_memory=True,
        include_tool_output=True,
    )
    rag_gen.save_dataset(medium_rag, OUTPUT_FILE)
    print(f"      -> {len(medium_rag)} samples written.")

    # -----------------------------------------------------------------------
    # Healthy Multi-Agent -- 150 samples
    # -----------------------------------------------------------------------
    agent_gen = HealthyMultiAgentGenerator(version="schema_v2")

    print("[3/4] Generating Healthy Multi-Agent (executor snapshots)...")
    executor_samples = agent_gen.generate(
        count=75,
        domain="all",
        difficulty="medium",
    )
    agent_gen.save_dataset(executor_samples, OUTPUT_FILE)
    print(f"      -> {len(executor_samples)} samples written.")

    print("[4/4] Generating Healthy Multi-Agent (validator snapshots)...")
    validator_samples = agent_gen.generate(
        count=75,
        domain="all",
        difficulty="medium",
    )
    agent_gen.save_dataset(validator_samples, OUTPUT_FILE)
    print(f"      -> {len(validator_samples)} samples written.")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total = len(easy_rag) + len(medium_rag) + len(executor_samples) + len(validator_samples)
    print(f"\n[DONE] Phase 2 complete. {total} total samples -> {OUTPUT_FILE}")

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
                assert obj["ground_truth"]["severity"] == "healthy", f"Non-healthy severity on line {i+1}"
                assert obj["ground_truth"]["architecture_failure"] == False, f"architecture_failure=True on line {i+1}"
            except Exception as e:
                errors.append(f"Line {i+1}: {e}")

    if errors:
        print(f"\n[WARN] {len(errors)} validation error(s) found:")
        for e in errors:
            print(f"   {e}")
    else:
        print(f"[OK] All {len(lines)} lines passed schema validation.")


if __name__ == "__main__":
    main()
