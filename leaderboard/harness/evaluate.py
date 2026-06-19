import os
import sys
import json
import time
import argparse
from typing import Dict, Any

# Ensure contextops is in path (assuming this runs in the workspace root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from contextops.core.normalizer import normalize
from contextops.core.engine import analyze
from contextops.core.config import ContextOpsConfig

# This would dynamically load the submitted optimization function
# For the harness, we expect a module named 'submission' with 'optimize_context'
try:
    from submission import optimize_context
except ImportError:
    # Dummy implementation for testing the harness
    def optimize_context(ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Naive truncator just for harness testing."""
        return ctx

def count_tokens(ctx: Dict[str, Any]) -> int:
    # Very rough simulation of token counting for the harness wrapper
    s = json.dumps(ctx)
    return len(s) // 4

def run_evaluation(repo_url: str, team_name: str, is_adversarial: bool = False):
    print(f"Evaluating submission: {team_name} ({repo_url})")
    
    # Load dataset
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if is_adversarial:
        dataset_path = os.path.join(base_dir, "ContextSecBench", "contextsecbench_v1_attacks.jsonl")
    else:
        dataset_path = os.path.join(base_dir, "ContextBench", "contextbench_v1_healthy_architectures.jsonl")
        
    samples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                
    # Evaluate
    config = ContextOpsConfig.default()
    config.roast_enabled = True
    
    total_input_tokens = 0
    total_output_tokens = 0
    total_latency = 0
    total_quality = 0
    
    print(f"Running across {len(samples)} samples...")
    for sample in samples:
        input_ctx = sample["context"]
        total_input_tokens += count_tokens(input_ctx)
        
        start_time = time.time()
        output_ctx = optimize_context(input_ctx)
        latency_ms = (time.time() - start_time) * 1000
        
        total_output_tokens += count_tokens(output_ctx)
        total_latency += latency_ms
        
        # Analyze quality
        bundle = normalize(output_ctx)
        result = analyze(bundle, config=config)
        total_quality += result.score
        
    # Aggregate
    avg_quality = total_quality / len(samples)
    avg_latency = total_latency / len(samples)
    compression_ratio = total_output_tokens / max(1, total_input_tokens)
    
    print("\nResults:")
    print(f"Quality Score: {avg_quality:.1f}")
    print(f"Compression Ratio: {compression_ratio:.3f}")
    print(f"Latency: {avg_latency:.1f}ms")
    
    # Check gates
    if avg_quality < 78:
        print("FAIL: Disqualified (Score < 78)")
        return
        
    if compression_ratio > 0.85: # Just for demo
        print("FAIL: Disqualified (Not enough compression)")
        return
        
    # Calculate Leaderboard Score
    quality_norm = avg_quality / 100.0
    compression_reward = 1.0 - compression_ratio
    
    latency_mult = 1.0
    if avg_latency > 100: latency_mult = 0.95
    if avg_latency > 500: latency_mult = 0.85
    if avg_latency > 2000: latency_mult = 0.70
    
    final_score = (0.50 * quality_norm + 0.35 * compression_reward + 0.15 * latency_mult) * 100
    
    print(f"FINAL SCORE: {final_score:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    
    run_evaluation(args.repo, args.name)
