"""Tests for tracking benchmark scores over time (Snapshot Regression Testing)."""

import json
from pathlib import Path
from contextops.api.inspect import inspect_context

SNAPSHOTS = {
    "01_simple_redundancy.json": {
        # Phase 3.1/3.3: density_penalty changed (log-scale + length smoothing). Score unchanged at 75.
        "score": 75,
        "redundancy_penalty": 6.62,
        "structure_penalty": 10.75,
        "top_recommendation_issue": "Redundant context: 100% similarity \u2014 exact duplicate content"
    },
    "02_semantic_duplicates.json": {
        # Phase 3.1/3.3: density_penalty 1.29 → 2.25 (log formula raises low-signal penalty slightly).
        # Score 87 → 86. redundancy and structure penalties unchanged.
        "score": 86,
        "redundancy_penalty": 0.0,
        "structure_penalty": 5.29,
        "top_recommendation_issue": "Retrieval dominance"
    },
    "03_rag_overload.json": {
        # Phase 0 Bug 3: score improved 70 → 76 (linear × 130 → log-scale divergence amplifier).
        # Phase 2 Bug 2: redundancy_penalty 2.18 → 2.09 (per-item max-waste cluster_score fix).
        # Phase 3.1/3.3: density_penalty shift + length smoothing. Score 76 → 75.
        "score": 75,
        "redundancy_penalty": 2.09,
        "structure_penalty": 17.04,
        "top_recommendation_issue": "Retrieval dominance"
    },
    "04_agent_trace_noise.json": {
        # Phase 3.1/3.3: density_penalty changed. Score 93 → 92.
        "score": 92,
        "redundancy_penalty": 0.0,
        "structure_penalty": 5.0,
        "top_recommendation_issue": "System prompt bloat"
    },
    "05_structured_data_case.json": {
        # Phase 3.1/3.3: density_penalty changed. Score 74 → 72.
        "score": 72,
        "redundancy_penalty": 0.0,
        "structure_penalty": 15.0,
        "top_recommendation_issue": "Retrieval dominance"
    },
    "06_fake_good_context.json": {
        # Phase 3.1/3.3: density_penalty changed. Score 95 → 93.
        "score": 93,
        "redundancy_penalty": 0.0,
        "structure_penalty": 0.0,
        "top_recommendation_issue": None
    },
    "07_instruction_conflict.json": {
        # Phase 3.1/3.3: density changed. Phase 3.2: padding anomaly now advisory-only
        # (impact_score=0.0, LOW severity) — top_rec is now 'Retrieval dominance'.
        # Score 91 → 90.
        "score": 90,
        "redundancy_penalty": 0.0,
        "structure_penalty": 7.16,
        "top_recommendation_issue": "Retrieval dominance"
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
