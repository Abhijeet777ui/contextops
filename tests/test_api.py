"""
Tests for the Public API (inspect_context).

This is the surface that external users and the future LangChain callback
will hit directly. It must be bulletproof against every kind of input
a real user would throw at it.

Coverage:
  - All input formats route correctly (string, list, dict)
  - AnalysisResult contract (fields exist, types correct, serialization works)
  - Edge cases: empty input, huge input, unicode, missing keys
  - Config passthrough
  - to_dict() serialization roundtrip
  - Score invariants hold through the full API path
"""

import json
import math

import pytest

from contextops.api.inspect import inspect_context
from contextops.core.config import ContextOpsConfig
from contextops.core.models import AnalysisResult


# ═══════════════════════════════════════════════════════════════════════════
# BASIC API CONTRACT — inspect_context() returns well-formed AnalysisResult
# ═══════════════════════════════════════════════════════════════════════════

class TestAPIContract:
    """The API must return a valid AnalysisResult for any supported input."""

    def test_string_input_returns_analysis_result(self):
        result = inspect_context("You are a helpful assistant.")
        assert isinstance(result, AnalysisResult)
        assert 0 <= result.score <= 100

    def test_message_list_input(self):
        result = inspect_context([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is AI?"},
        ])
        assert isinstance(result, AnalysisResult)
        assert result.score >= 0

    def test_dict_input(self):
        result = inspect_context({
            "system": "You are an assistant.",
            "messages": [{"role": "user", "content": "Hello"}],
            "chunks": [{"content": "Some data", "source": "doc.md"}],
        })
        assert isinstance(result, AnalysisResult)
        assert result.token_breakdown.total_tokens > 0

    def test_benchmark_wrapper_dict(self):
        """Benchmark-style dicts with 'input' wrapper must work."""
        result = inspect_context({
            "name": "test_case",
            "input": {
                "system": "sys",
                "messages": [{"role": "user", "content": "hi"}],
                "chunks": [],
                "memory": [],
                "tools": [],
            }
        })
        assert isinstance(result, AnalysisResult)

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError):
            inspect_context(42)


# ═══════════════════════════════════════════════════════════════════════════
# RESULT FIELDS — every field must exist and have correct types
# ═══════════════════════════════════════════════════════════════════════════

class TestResultFields:
    """Validate AnalysisResult structure from a real API call."""

    @pytest.fixture
    def result(self):
        return inspect_context({
            "system": "You are a coding assistant.",
            "messages": [{"role": "user", "content": "Write a function."}],
            "chunks": [
                {"content": "Python is a high-level language.", "source": "wiki.md"},
                {"content": "Functions use the def keyword.", "source": "tutorial.md"},
            ],
        })

    def test_score_is_int(self, result):
        assert isinstance(result.score, int)

    def test_score_in_range(self, result):
        assert 0 <= result.score <= 100

    def test_score_breakdown_exists(self, result):
        sb = result.score_breakdown
        assert hasattr(sb, "redundancy_penalty")
        assert hasattr(sb, "density_penalty")
        assert hasattr(sb, "structure_penalty")
        assert hasattr(sb, "concentration_penalty")
        assert hasattr(sb, "total_penalty")

    def test_all_penalties_non_negative(self, result):
        sb = result.score_breakdown
        assert sb.redundancy_penalty >= 0
        assert sb.density_penalty >= 0
        assert sb.structure_penalty >= 0
        assert sb.concentration_penalty >= 0

    def test_penalty_sum_consistency(self, result):
        sb = result.score_breakdown
        expected = (
            sb.redundancy_penalty
            + sb.density_penalty
            + sb.structure_penalty
            + sb.concentration_penalty
        )
        assert abs(sb.total_penalty - expected) < 0.01

    def test_score_equals_100_minus_penalty(self, result):
        expected = max(0, min(100, round(100 - result.score_breakdown.total_penalty)))
        assert abs(result.score - expected) <= 1

    def test_token_breakdown_exists(self, result):
        tb = result.token_breakdown
        assert tb.total_tokens > 0
        assert isinstance(tb.by_type, dict)
        assert tb.estimated_reduction_pct >= 0.0
        assert isinstance(tb.wasted_tokens, int)

    def test_no_nan_or_inf(self, result):
        """No numeric field may be NaN or Infinity."""
        sb = result.score_breakdown
        tb = result.token_breakdown
        values = [
            result.score,
            sb.redundancy_penalty, sb.density_penalty,
            sb.structure_penalty, sb.concentration_penalty,
            sb.total_penalty,
            tb.total_tokens, tb.wasted_tokens, tb.estimated_reduction_pct,
        ]
        for v in values:
            assert not math.isnan(v), f"NaN found: {v}"
            assert not math.isinf(v), f"Inf found: {v}"

    def test_metadata_contains_item_count(self, result):
        assert "item_count" in result.metadata
        assert result.metadata["item_count"] == 4  # system + user + 2 chunks

    def test_metadata_contains_model(self, result):
        assert "model" in result.metadata

    def test_density_signal_exists(self, result):
        ds = result.density_signal
        assert ds is not None
        assert 0.0 <= ds.format_overhead <= 1.0
        assert 0.0 <= ds.whitespace_waste <= 1.0
        assert 0.0 <= ds.entropy_compression <= 1.0
        assert 0.0 <= ds.total_density_signal <= 1.0

    def test_recommendations_are_list(self, result):
        assert isinstance(result.recommendations, list)

    def test_redundancy_findings_are_list(self, result):
        assert isinstance(result.redundancy_findings, list)

    def test_structure_findings_are_list(self, result):
        assert isinstance(result.structure_findings, list)


# ═══════════════════════════════════════════════════════════════════════════
# to_dict() SERIALIZATION — must produce valid JSON-serializable output
# ═══════════════════════════════════════════════════════════════════════════

class TestSerialization:
    """AnalysisResult.to_dict() must produce JSON-safe output."""

    def test_to_dict_is_json_serializable(self):
        result = inspect_context("You are an AI.")
        d = result.to_dict()
        # Must not raise
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    def test_to_dict_roundtrip(self):
        result = inspect_context({
            "system": "sys",
            "chunks": [{"content": "data", "source": "doc"}],
        })
        d = result.to_dict()
        parsed = json.loads(json.dumps(d))
        assert parsed["score"] == result.score
        assert "score_breakdown" in parsed
        assert "token_breakdown" in parsed

    def test_to_dict_contains_all_top_level_keys(self):
        result = inspect_context("test")
        d = result.to_dict()
        required_keys = {"score", "mode", "config_version", "score_breakdown",
                         "token_breakdown", "findings", "recommendations", "metadata"}
        assert required_keys.issubset(d.keys())

    def test_to_dict_findings_structure(self):
        result = inspect_context({
            "chunks": [
                {"content": "same data here", "source": "a.md"},
                {"content": "same data here", "source": "b.md"},
            ]
        })
        d = result.to_dict()
        assert "redundancy" in d["findings"]
        assert "structure" in d["findings"]
        assert isinstance(d["findings"]["redundancy"], list)
        assert isinstance(d["findings"]["structure"], list)

    def test_to_dict_density_signal_present(self):
        result = inspect_context("test content")
        d = result.to_dict()
        assert "density_signal" in d
        assert "density_effect" in d


# ═══════════════════════════════════════════════════════════════════════════
# EDGE CASES — stress the full API path
# ═══════════════════════════════════════════════════════════════════════════

class TestAPIEdgeCases:
    """Push the API to its limits with unusual inputs."""

    def test_single_empty_chunk(self):
        """Empty chunk must not crash the full pipeline."""
        result = inspect_context({"chunks": [{"content": "", "source": "empty.md"}]})
        assert isinstance(result, AnalysisResult)
        assert 0 <= result.score <= 100

    def test_single_whitespace_chunk(self):
        result = inspect_context({"chunks": [{"content": "   \n\t  ", "source": "ws.md"}]})
        assert isinstance(result, AnalysisResult)

    def test_many_identical_chunks(self):
        """20 identical chunks must flag redundancy, not crash."""
        chunks = [{"content": "The same text repeated many times.", "source": f"doc_{i}"} for i in range(20)]
        result = inspect_context({"chunks": chunks})
        assert result.score < 90  # should be penalized
        assert len(result.redundancy_findings) > 0

    def test_very_long_system_prompt(self):
        """Extremely long system prompt must flag system prompt bloat."""
        result = inspect_context({
            "system": "Follow these instructions carefully. " * 500,
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert isinstance(result, AnalysisResult)
        issues = [r.issue for r in result.recommendations]
        assert any("system" in issue.lower() or "bloat" in issue.lower() for issue in issues)

    def test_unicode_full_pipeline(self):
        """Unicode through the entire pipeline must not corrupt or crash."""
        result = inspect_context({
            "system": "あなたは日本語アシスタントです。",
            "messages": [{"role": "user", "content": "こんにちは"}],
            "chunks": [{"content": "日本語のデータ", "source": "日本語.md"}],
        })
        assert isinstance(result, AnalysisResult)
        assert result.token_breakdown.total_tokens > 0

    def test_all_retrieval_flags_dominance(self):
        """Context that is 100% retrieval must flag 'Retrieval dominance'."""
        result = inspect_context({
            "chunks": [
                {"content": "chunk one data", "source": "a"},
                {"content": "chunk two data", "source": "b"},
            ]
        })
        structure_issues = [f.issue for f in result.structure_findings]
        assert "Retrieval dominance" in structure_issues

    def test_only_system_no_structure_penalty(self):
        """A single system prompt should not trigger structure penalties."""
        result = inspect_context("Just a simple system prompt.")
        assert result.score_breakdown.structure_penalty == 0.0

    def test_empty_message_list(self):
        """Empty message list must produce a valid (empty) result."""
        result = inspect_context([])
        assert isinstance(result, AnalysisResult)
        assert result.token_breakdown.total_tokens == 0

    def test_deeply_nested_tool_output(self):
        """Tool with large JSON-like output must not crash."""
        nested_json = json.dumps({"level1": {"level2": {"level3": list(range(100))}}})
        result = inspect_context({
            "system": "You are an agent.",
            "tools": [{"output": nested_json, "name": "api_call"}],
        })
        assert isinstance(result, AnalysisResult)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG PASSTHROUGH — custom configs must propagate
# ═══════════════════════════════════════════════════════════════════════════

class TestConfigPassthrough:
    """Custom ContextOpsConfig must affect the analysis."""

    def test_relaxed_retrieval_threshold_suppresses_finding(self):
        """Setting retrieval_max_ratio=1.0 must suppress retrieval dominance."""
        data = {"chunks": [
            {"content": "chunk a", "source": "a"},
            {"content": "chunk b", "source": "b"},
        ]}
        config = ContextOpsConfig(retrieval_max_ratio=1.0)
        result = inspect_context(data, config=config)
        structure_issues = [f.issue for f in result.structure_findings]
        assert "Retrieval dominance" not in structure_issues

    def test_strict_threshold_catches_more(self):
        """Lowering threshold must catch imbalances that default wouldn't."""
        data = {
            "system": "system prompt",
            "chunks": [
                {"content": "chunk data one", "source": "a"},
            ],
        }
        # With default (0.70), retrieval at ~50% wouldn't trigger
        result_default = inspect_context(data)
        # With threshold=0.30, it would trigger
        config = ContextOpsConfig(retrieval_max_ratio=0.30)
        result_strict = inspect_context(data, config=config)

        default_issues = [f.issue for f in result_default.structure_findings]
        strict_issues = [f.issue for f in result_strict.structure_findings]

        assert "Retrieval dominance" not in default_issues
        assert "Retrieval dominance" in strict_issues

    def test_custom_config_mode_is_custom(self):
        """Custom config must set mode='custom' in the result."""
        config = ContextOpsConfig(retrieval_max_ratio=0.90, mode="custom")
        result = inspect_context("test", config=config)
        assert result.mode == "custom"


# ═══════════════════════════════════════════════════════════════════════════
# DETERMINISM — same input, same output, every time
# ═══════════════════════════════════════════════════════════════════════════

class TestAPIDeterminism:
    """The full API pipeline must be fully deterministic."""

    def test_same_string_same_score(self):
        r1 = inspect_context("Test prompt for determinism check.")
        r2 = inspect_context("Test prompt for determinism check.")
        assert r1.score == r2.score

    def test_same_dict_same_breakdown(self):
        data = {
            "system": "sys",
            "messages": [{"role": "user", "content": "q"}],
            "chunks": [{"content": "data", "source": "doc.md"}],
        }
        r1 = inspect_context(data)
        r2 = inspect_context(data)
        assert r1.score == r2.score
        assert r1.score_breakdown.redundancy_penalty == r2.score_breakdown.redundancy_penalty
        assert r1.score_breakdown.density_penalty == r2.score_breakdown.density_penalty
        assert r1.score_breakdown.structure_penalty == r2.score_breakdown.structure_penalty
        assert r1.score_breakdown.concentration_penalty == r2.score_breakdown.concentration_penalty
