"""Tests for the API schema and JSON output stability."""

from contextops.api.inspect import inspect_context


def test_analysis_result_schema() -> None:
    """Test that the core JSON schema fields always exist."""
    raw_input = {
        "system": "Hello world.",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    
    result = inspect_context(raw_input)
    data = result.to_dict()
    
    # Assert top-level keys
    assert "score" in data
    assert "mode" in data
    assert "config_version" in data
    assert "score_breakdown" in data
    assert "token_breakdown" in data
    assert "findings" in data
    assert "recommendations" in data
    assert "metadata" in data
    
    assert data["mode"] == "strict"
    assert data["config_version"] == "1.0"
    
    # Assert score_breakdown
    breakdown = data["score_breakdown"]
    assert "redundancy_penalty" in breakdown
    assert "density_penalty" in breakdown
    assert "structure_penalty" in breakdown
    assert "concentration_penalty" in breakdown
    assert "total_penalty" in breakdown
