"""
ContextOps Deterministic Benchmark Runner

Executes all JSON benchmarks in this directory against the core inspect engine.
Validates that expected signals are detected and unexpected signals are missing.
"""

import json
import os
import sys
from typing import Any

from contextops.api.inspect import inspect_context


def run_benchmark(filepath: str) -> dict[str, Any]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    name = data.get("name", os.path.basename(filepath))
    desc = data.get("description", "")
    raw_input = data["input"]
    expected_signals = set(data.get("expected_signals", []))
    unexpected_signals = set(data.get("unexpected_signals", []))
    
    # Execute the engine
    result = inspect_context(raw_input)
    result_dict = result.to_dict()
    
    # Extract detected signals from findings
    detected_signals = set()
    for f in result_dict["findings"].get("redundancy", []):
        detected_signals.add(f["classification"].upper())
    for f in result_dict["findings"].get("structure", []):
        detected_signals.add("STRUCTURE_IMBALANCE")
        
    missing_expected = expected_signals - detected_signals
    found_unexpected = unexpected_signals & detected_signals
    
    passed = len(missing_expected) == 0 and len(found_unexpected) == 0
    
    return {
        "name": name,
        "description": desc,
        "score": result_dict["score"],
        "expected_signals": list(expected_signals),
        "detected_signals": list(detected_signals),
        "missing_expected": list(missing_expected),
        "found_unexpected": list(found_unexpected),
        "status": "PASS" if passed else "FAIL"
    }


def main():
    benchmarks_dir = os.path.dirname(os.path.abspath(__file__))
    benchmark_files = [f for f in os.listdir(benchmarks_dir) if f.endswith(".json") and f != "results.json"]
    benchmark_files.sort()
    
    if not benchmark_files:
        print("No benchmark JSON files found.")
        sys.exit(1)
        
    print(f"Running {len(benchmark_files)} benchmarks...\n")
    
    all_passed = True
    results = []
    
    for f in benchmark_files:
        filepath = os.path.join(benchmarks_dir, f)
        res = run_benchmark(filepath)
        results.append(res)
        
        status_color = "\033[92m" if res["status"] == "PASS" else "\033[91m"
        reset_color = "\033[0m"
        
        print(f"[{status_color}{res['status']}{reset_color}] {res['name']}")
        print(f"  Description: {res['description']}")
        print(f"  Score:       {res['score']}")
        
        if res["status"] == "FAIL":
            all_passed = False
            if res["missing_expected"]:
                print(f"  \033[91mMISSING EXPECTED:\033[0m {res['missing_expected']}")
            if res["found_unexpected"]:
                print(f"  \033[91mFOUND UNEXPECTED:\033[0m {res['found_unexpected']}")
        print()
        
    # Write full output
    with open(os.path.join(benchmarks_dir, "results.json"), "w", encoding="utf-8") as out:
        json.dump(results, out, indent=2)
        
    if not all_passed:
        print("\033[91mSome benchmarks failed.\033[0m")
        sys.exit(1)
    else:
        print("\033[92mAll benchmarks passed!\033[0m")
        sys.exit(0)

if __name__ == "__main__":
    main()
