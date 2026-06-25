"""
Quality Floor Calibration Script for ContextBench Leaderboard.

Runs ContextOps across all ContextBench_v1 samples, collects the full score
distribution, and determines the empirically justified quality floor threshold.

The threshold is the score at which:
  - >= 90% of healthy architecture samples score ABOVE it
  - The maximum separation between healthy and failure distributions occurs

Output:
  - Prints the calibrated threshold with supporting statistics
  - Saves raw score data to scripts/calibration_results.json for audit
"""

import os
import sys
import json
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextops.core.normalizer import normalize
from contextops.core.engine import analyze
from contextops.core.config import ContextOpsConfig


def load_dataset(filepath: str) -> list[dict]:
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def score_sample(sample: dict, config: ContextOpsConfig) -> int:
    """Run ContextOps on a single ContextBench sample and return the score."""
    context_data = sample["context"]
    raw_input = {
        "system": context_data.get("system_prompt", ""),
        "messages": context_data.get("messages", []),
        "retrieval": context_data.get("retrieval_chunks", []),
        "tools": context_data.get("tool_outputs", []),
        "memory": context_data.get("memory", []),
    }
    bundle = normalize(raw_input)
    result = analyze(bundle, config=config)
    return result.score


def find_optimal_threshold(healthy_scores: list[int], failure_scores: list[int]) -> dict:
    """
    Find the threshold that maximizes separation between healthy and failure
    distributions while keeping >= 90% of healthy samples above the floor.
    """
    best_threshold = 0
    best_separation = -1.0
    results_by_threshold = {}

    for threshold in range(0, 101):
        healthy_pass = sum(1 for s in healthy_scores if s >= threshold)
        healthy_pass_rate = healthy_pass / max(1, len(healthy_scores))

        failure_fail = sum(1 for s in failure_scores if s < threshold)
        failure_fail_rate = failure_fail / max(1, len(failure_scores))

        # Separation = how well this threshold divides the two populations
        separation = healthy_pass_rate + failure_fail_rate - 1.0  # Youden's J

        results_by_threshold[threshold] = {
            "healthy_pass_rate": round(healthy_pass_rate, 4),
            "failure_disqualified_rate": round(failure_fail_rate, 4),
            "youden_j": round(separation, 4),
        }

        # Only consider thresholds where >= 90% of healthy samples pass
        if healthy_pass_rate >= 0.90 and separation > best_separation:
            best_separation = separation
            best_threshold = threshold

    return {
        "optimal_threshold": best_threshold,
        "best_youden_j": round(best_separation, 4),
        "detail_at_optimal": results_by_threshold[best_threshold],
    }


def print_histogram(scores: list[int], label: str, width: int = 50):
    """Print a simple text histogram of scores in 10-point bands."""
    bands = {}
    for lo in range(0, 100, 10):
        hi = lo + 9
        count = sum(1 for s in scores if lo <= s <= hi)
        bands[f"{lo:3d}-{hi:3d}"] = count

    max_count = max(bands.values()) if bands.values() else 1
    print(f"\n  {label} (n={len(scores)})")
    print(f"  {'Band':>7s}  {'Count':>5s}  Distribution")
    print(f"  {'-'*7}  {'-'*5}  {'-'*width}")
    for band, count in bands.items():
        bar_len = int((count / max_count) * width) if max_count > 0 else 0
        bar = "#" * bar_len
        print(f"  {band}  {count:5d}  {bar}")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = ContextOpsConfig.default()

    # Load datasets
    healthy_path = os.path.join(base_dir, "ContextBench", "contextbench_v1_healthy_architectures.jsonl")
    failure_paths = [
        os.path.join(base_dir, "ContextBench", "contextbench_v1_architecture_failures.jsonl"),
        os.path.join(base_dir, "ContextBench", "contextbench_v1_temporal_drift.jsonl"),
    ]

    print("=" * 60)
    print("  CONTEXTBENCH QUALITY FLOOR CALIBRATION")
    print("=" * 60)

    # Score healthy architectures
    print("\nScoring healthy architectures...")
    healthy_samples = load_dataset(healthy_path)
    healthy_scores = []
    for i, sample in enumerate(healthy_samples):
        score = score_sample(sample, config)
        healthy_scores.append(score)
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(healthy_samples)}]")

    # Score failure architectures
    print("\nScoring failure architectures...")
    failure_scores = []
    for fpath in failure_paths:
        if not os.path.exists(fpath):
            print(f"  [WARN] Skipping missing file: {fpath}")
            continue
        failure_samples = load_dataset(fpath)
        for i, sample in enumerate(failure_samples):
            score = score_sample(sample, config)
            failure_scores.append(score)
            if (i + 1) % 100 == 0:
                print(f"  [{i + 1}/{len(failure_samples)}] ({os.path.basename(fpath)})")

    # Print distributions
    print("\n" + "=" * 60)
    print("  SCORE DISTRIBUTIONS")
    print("=" * 60)
    print_histogram(healthy_scores, "Healthy Architectures")
    print_histogram(failure_scores, "Failure Architectures")

    # Statistics
    print("\n" + "=" * 60)
    print("  SUMMARY STATISTICS")
    print("=" * 60)
    print(f"\n  Healthy:  mean={statistics.mean(healthy_scores):.1f}  "
          f"median={statistics.median(healthy_scores):.0f}  "
          f"min={min(healthy_scores)}  max={max(healthy_scores)}  "
          f"stdev={statistics.stdev(healthy_scores):.1f}")
    print(f"  Failure:  mean={statistics.mean(failure_scores):.1f}  "
          f"median={statistics.median(failure_scores):.0f}  "
          f"min={min(failure_scores)}  max={max(failure_scores)}  "
          f"stdev={statistics.stdev(failure_scores):.1f}")

    # Find optimal threshold
    calibration = find_optimal_threshold(healthy_scores, failure_scores)

    print("\n" + "=" * 60)
    print("  CALIBRATED QUALITY FLOOR")
    print("=" * 60)
    t = calibration["optimal_threshold"]
    d = calibration["detail_at_optimal"]
    print(f"\n  Optimal Threshold:           {t}")
    print(f"  Youden's J statistic:        {calibration['best_youden_j']}")
    print(f"  Healthy pass rate at {t}:     {d['healthy_pass_rate'] * 100:.1f}%")
    print(f"  Failure disqualified at {t}:  {d['failure_disqualified_rate'] * 100:.1f}%")

    # Save full results for audit
    output_path = os.path.join(base_dir, "scripts", "calibration_results.json")
    audit_data = {
        "calibrated_threshold": t,
        "youden_j": calibration["best_youden_j"],
        "healthy_pass_rate": d["healthy_pass_rate"],
        "failure_disqualified_rate": d["failure_disqualified_rate"],
        "healthy_scores": healthy_scores,
        "failure_scores": failure_scores,
        "healthy_stats": {
            "mean": round(statistics.mean(healthy_scores), 2),
            "median": statistics.median(healthy_scores),
            "min": min(healthy_scores),
            "max": max(healthy_scores),
            "stdev": round(statistics.stdev(healthy_scores), 2),
            "n": len(healthy_scores),
        },
        "failure_stats": {
            "mean": round(statistics.mean(failure_scores), 2),
            "median": statistics.median(failure_scores),
            "min": min(failure_scores),
            "max": max(failure_scores),
            "stdev": round(statistics.stdev(failure_scores), 2),
            "n": len(failure_scores),
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
    print(f"\n  Full calibration data saved to: {output_path}")
    print()


if __name__ == "__main__":
    main()
