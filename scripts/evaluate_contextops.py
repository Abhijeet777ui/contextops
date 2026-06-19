"""
ContextOps Evaluator on ContextBench

This script runs the core ContextOps engine against the newly generated
ContextBench and ContextSecBench datasets. It translates the ContextBench
schema into the ContextOps normalizer format, runs the analyzer, and
compares the output to the ground_truth expected properties.
"""

import os
import sys
import json
from dataclasses import dataclass
from typing import List, Dict, Any

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextops.core.normalizer import normalize
from contextops.core.engine import analyze
from contextops.core.config import ContextOpsConfig


@dataclass
class EvaluationMetrics:
    total: int = 0
    correct_redundancy: int = 0
    correct_density: int = 0
    correct_structure: int = 0
    correct_concentration: int = 0

    false_positives_overall: int = 0
    true_positives_overall: int = 0
    
    # Track overall architecture failures expected vs predicted
    expected_failures: int = 0
    predicted_failures: int = 0


def load_dataset(filepath: str) -> List[Dict[str, Any]]:
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def evaluate_sample(sample: Dict[str, Any], config: ContextOpsConfig) -> Dict[str, bool]:
    """Runs ContextOps on a single sample and returns predicted boolean flags."""
    context_data = sample["context"]
    
    # Translate ContextBench schema to ContextOps normalizer format
    raw_input = {
        "system": context_data.get("system_prompt", ""),
        "messages": context_data.get("messages", []),
        "retrieval": context_data.get("retrieval_chunks", []),
        "tools": context_data.get("tool_outputs", []),
        "memory": context_data.get("memory", []),
    }

    bundle = normalize(raw_input)
    result = analyze(bundle, config=config)

    # Map ContextOps results to boolean properties using calibrated thresholds
    # Minor inefficiencies shouldn't count as a complete failure flag
    predicted = {
        "contains_redundancy": result.score_breakdown.redundancy_penalty > 10.0,
        "contains_density_bloat": (
            result.score_breakdown.density_penalty > 10.0 or 
            (result.density_signal and result.density_signal.total_density_signal > 0.2)
        ),
        "contains_structure_imbalance": result.score_breakdown.structure_penalty > 5.0,
        "contains_source_concentration": result.score_breakdown.concentration_penalty > 10.0,
    }
    
    # A global failure is when the score drops below 80
    predicted["architecture_failure"] = result.score < 80
    return predicted


def process_dataset(name: str, filepath: str, is_failure_dataset: bool) -> EvaluationMetrics:
    print(f"Loading {name}...")
    if not os.path.exists(filepath):
        print(f"  [WARN] File not found: {filepath}")
        return EvaluationMetrics()
        
    samples = load_dataset(filepath)
    metrics = EvaluationMetrics(total=len(samples))
    config = ContextOpsConfig.default()

    for i, sample in enumerate(samples):
        expected = sample["ground_truth"]["expected_properties"]
        expected_failure = sample["ground_truth"]["architecture_failure"]
        
        predicted = evaluate_sample(sample, config)
        
        if expected["contains_redundancy"] == predicted["contains_redundancy"]:
            metrics.correct_redundancy += 1
        if expected["contains_density_bloat"] == predicted["contains_density_bloat"]:
            metrics.correct_density += 1
        if expected["contains_structure_imbalance"] == predicted["contains_structure_imbalance"]:
            metrics.correct_structure += 1
        if expected["contains_source_concentration"] == predicted["contains_source_concentration"]:
            metrics.correct_concentration += 1

        if expected_failure:
            metrics.expected_failures += 1
            if predicted["architecture_failure"]:
                metrics.predicted_failures += 1
                metrics.true_positives_overall += 1
        else:
            if predicted["architecture_failure"]:
                metrics.false_positives_overall += 1

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(samples)} samples...")

    return metrics


def print_report(name: str, metrics: EvaluationMetrics, is_failure_dataset: bool):
    print(f"\n{'='*50}")
    print(f" REPORT: {name}")
    print(f"{'='*50}")
    print(f" Total Samples Evaluated: {metrics.total}")
    
    if metrics.total == 0:
        return

    print("\n --- Property-Level Accuracy ---")
    print(f" Redundancy Detection:    {metrics.correct_redundancy / metrics.total * 100:.1f}%")
    print(f" Density Bloat Detection: {metrics.correct_density / metrics.total * 100:.1f}%")
    print(f" Structure Imbalance:     {metrics.correct_structure / metrics.total * 100:.1f}%")
    print(f" Source Concentration:    {metrics.correct_concentration / metrics.total * 100:.1f}%")
    
    print("\n --- Overall Architecture Failure Detection ---")
    if not is_failure_dataset:
        fpr = (metrics.false_positives_overall / metrics.total) * 100
        print(f" False Positive Rate (FPR): {fpr:.1f}% ({metrics.false_positives_overall}/{metrics.total})")
        if fpr > 0:
            print("   -> (ContextOps flagged healthy architectures as failures)")
        else:
            print("   -> [EXCELLENT] ContextOps perfectly ignored healthy architectures.")
    else:
        tpr = (metrics.true_positives_overall / metrics.total) * 100
        print(f" True Positive Rate (Recall): {tpr:.1f}% ({metrics.true_positives_overall}/{metrics.total})")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    datasets = [
        {
            "name": "Healthy Architectures",
            "path": os.path.join(base_dir, "ContextBench", "contextbench_v1_healthy_architectures.jsonl"),
            "is_failure": False
        },
        {
            "name": "Architecture Failures",
            "path": os.path.join(base_dir, "ContextBench", "contextbench_v1_architecture_failures.jsonl"),
            "is_failure": True
        },
        {
            "name": "Temporal Drift Failures",
            "path": os.path.join(base_dir, "ContextBench", "contextbench_v1_temporal_drift.jsonl"),
            "is_failure": True
        },
        {
            "name": "ContextSecBench (Adversarial & DoS)",
            "path": os.path.join(base_dir, "ContextSecBench", "contextsecbench_v1_attacks.jsonl"),
            "is_failure": True
        }
    ]

    print("Starting ContextOps Evaluation on ContextBench...")
    for ds in datasets:
        metrics = process_dataset(ds["name"], ds["path"], ds["is_failure"])
        print_report(ds["name"], metrics, ds["is_failure"])


if __name__ == "__main__":
    main()
