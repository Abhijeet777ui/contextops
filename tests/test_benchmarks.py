"""Tests for tracking benchmark scores over time (Snapshot Regression Testing)."""

import json
from pathlib import Path
from contextops.api.inspect import inspect_context

SNAPSHOTS = {
    "01_simple_redundancy.json": {
        "score": 75,
        "redundancy_penalty": 6.62,
        "structure_penalty": 10.75,
        "top_recommendation_issue": "Redundant context: 100% similarity between 'refunds_doc' and 'customer_service_macro_1' \u2014 redundant context from independent sources"
    },
    "02_semantic_duplicates.json": {
        "score": 80,
        "redundancy_penalty": 6.05,
        "structure_penalty": 5.29,
        "top_recommendation_issue": "Retrieval dominance"
    },
    "03_rag_overload.json": {
        "score": 63,
        "redundancy_penalty": 8.46,
        "structure_penalty": 17.04,
        "top_recommendation_issue": "Retrieval dominance"
    },
    "04_agent_trace_noise.json": {
        "score": 93,
        "redundancy_penalty": 0.0,
        "structure_penalty": 5.0,
        "top_recommendation_issue": "System prompt bloat"
    },
    "05_structured_data_case.json": {
        "score": 70,
        "redundancy_penalty": 0.0,
        "structure_penalty": 15.0,
        "top_recommendation_issue": "Retrieval dominance"
    },
    "06_fake_good_context.json": {
        "score": 88,
        "redundancy_penalty": 6.95,
        "structure_penalty": 0.0,
        "top_recommendation_issue": "Redundant context: 13% similarity between 'system_prompt' and 'message_0' \u2014 redundant context from independent sources"
    },
    "07_instruction_conflict.json": {
        "score": 78,
        "redundancy_penalty": 8.04,
        "structure_penalty": 7.16,
        "top_recommendation_issue": "Suspicious Threshold Padding detected (divergence: 0.0350)"
    }
}



def test_benchmark_snapshots() -> None:
    """
    Ensure benchmark scores, penalties, and recommendations do not change 
    silently. If they do change due to a deliberate algorithm update, 
    these snapshots must be manually updated.
    """
    benchmarks_dir = Path("benchmarks")
    
    for file_name, snapshot in SNAPSHOTS.items():
        file_path = benchmarks_dir / file_name
        assert file_path.exists(), f"Benchmark {file_name} missing"
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        context_data = data.get("input", data)
        result = inspect_context(context_data)
        
        top_rec = result.recommendations[0].issue if result.recommendations else None
        
        # Verify score
        assert result.score == snapshot["score"], \
            f"Score changed for {file_name}: {result.score} != {snapshot['score']}"
            
        # Verify penalties
        assert result.score_breakdown.redundancy_penalty == snapshot["redundancy_penalty"], \
            f"Redundancy changed for {file_name}: {result.score_breakdown.redundancy_penalty} != {snapshot['redundancy_penalty']}"
            
        assert result.score_breakdown.structure_penalty == snapshot["structure_penalty"], \
            f"Structure changed for {file_name}: {result.score_breakdown.structure_penalty} != {snapshot['structure_penalty']}"
            
        # Verify top recommendation
        assert top_rec == snapshot["top_recommendation_issue"], \
            f"Rec changed for {file_name}: '{top_rec}' != '{snapshot['top_recommendation_issue']}'"
