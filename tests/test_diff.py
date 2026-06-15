import json
from pathlib import Path
from contextops.api.diff import diff_contexts, get_recommendation_id
from contextops.core.models import Recommendation, FindingSeverity

def test_get_recommendation_id_stability():
    """Verify that the ID generation is stable and normalizes whitespace/case."""
    rec1 = Recommendation(
        issue="  Retrieval   dominance ",
        impact_score=5.0,
        token_savings=0,
        fix="Fix it",
        severity=FindingSeverity.MEDIUM
    )
    rec2 = Recommendation(
        issue="retrieval dominance",
        impact_score=10.0,  # Different impact shouldn't change the ID
        token_savings=100,  # Different savings shouldn't change the ID
        fix="Different fix",
        severity=FindingSeverity.HIGH
    )
    
    id1 = get_recommendation_id(rec1)
    id2 = get_recommendation_id(rec2)
    
    assert id1 == id2
    assert len(id1) == 12

def test_diff_logic():
    """Test the diff logic on standard benchmarks."""
    benchmarks_dir = Path(__file__).resolve().parent.parent / "benchmarks"
    file_a = benchmarks_dir / "diff_case_a.json"
    file_b = benchmarks_dir / "diff_case_b.json"
    
    # Run the diff
    with open(file_a, "r", encoding="utf-8") as f:
        data_a = json.load(f)
    with open(file_b, "r", encoding="utf-8") as f:
        data_b = json.load(f)
        
    diff_result = diff_contexts(data_a, data_b)
    
    # 1. Verify numeric deltas
    # A has exact redundancy. B has no redundancy but has structure imbalance.
    # Score of A should be lower or different than B. Tokens of A < Tokens of B.
    assert diff_result.token_delta > 0
    assert "redundancy" in diff_result.structure_delta
    assert "density" in diff_result.structure_delta
    
    # 2. Verify recommendation lifecycle
    # A has Redundant Context. B doesn't.
    # So Redundant Context should be in resolved.
    resolved_issues = [r.issue.lower() for r in diff_result.resolved_recommendations]
    assert any("redundant context" in issue for issue in resolved_issues)
    
    # B may have at most minor redundancy findings (low-signal overlaps)
    # from the improved RS sensitivity, but no structural issues.
    new_issues = [r.issue.lower() for r in diff_result.new_recommendations]
    assert all("redundant context" in issue for issue in new_issues), \
        f"Unexpected non-redundancy new issues in diff_case_b: {new_issues}"
    
    # 3. Verify net impact
    assert diff_result.net_impact in ["IMPROVEMENT", "DEGRADATION", "NEUTRAL"]
