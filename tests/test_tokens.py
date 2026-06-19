"""
Tests for the Token Analyzer.

The token counter is the foundation of every penalty calculation:
  - Redundancy penalty uses wasted_tokens
  - Structure penalty uses per-type token ratios
  - Cost estimates are derived from token counts

If token counting is wrong, every score is wrong.

Coverage:
  - count_tokens() for various inputs (empty, short, long, unicode)
  - analyze_tokens() side-effects (setting item.token_count)
  - TokenBreakdown correctness (by_type aggregation, cost math)
  - Model fallback behavior
  - Edge cases: empty bundles, single items, zero-length content
"""

import pytest

from contextops.analyzers.tokens import count_tokens, analyze_tokens, DEFAULT_COST_PER_1K_TOKENS
from contextops.core.models import ContextBundle, ContextItem, ContextType


# ═══════════════════════════════════════════════════════════════════════════
# count_tokens() — UNIT TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestCountTokens:
    """Unit tests for the count_tokens() function."""

    def test_empty_string_is_zero(self):
        assert count_tokens("") == 0

    def test_single_word(self):
        tokens = count_tokens("hello")
        assert tokens == 1

    def test_simple_sentence(self):
        tokens = count_tokens("The quick brown fox jumps over the lazy dog.")
        assert tokens > 0
        assert tokens < 20  # sanity bound

    def test_returns_int(self):
        result = count_tokens("some text")
        assert isinstance(result, int)

    def test_whitespace_only(self):
        """Whitespace-only strings may tokenize to something — just must not crash."""
        tokens = count_tokens("   \n\t\n   ")
        assert isinstance(tokens, int)
        assert tokens >= 0

    def test_unicode_text(self):
        """Chinese / Japanese text must tokenize without error."""
        tokens = count_tokens("これはテストです。机器学习是人工智能的一部分。")
        assert tokens > 0

    def test_emoji(self):
        tokens = count_tokens("🎯🔥💯")
        assert tokens > 0

    def test_long_text(self):
        """1000-word text must tokenize in reasonable time and produce reasonable count."""
        text = " ".join(["word"] * 1000)
        tokens = count_tokens(text)
        # "word" repeated 1000 times — each "word" is 1 token, plus spaces
        assert tokens > 500
        assert tokens < 2000

    def test_code_snippet(self):
        """Code with special characters must tokenize correctly."""
        code = '''def foo(x: int) -> str:\n    return f"result={x * 2}"\n'''
        tokens = count_tokens(code)
        assert tokens > 5

    def test_json_string(self):
        text = '{"key": "value", "nested": {"a": [1, 2, 3]}}'
        tokens = count_tokens(text)
        assert tokens > 5

    def test_unknown_model_falls_back(self):
        """Unknown model name must not crash — should fall back to cl100k_base."""
        tokens = count_tokens("hello world", model="nonexistent-model-xyz")
        assert tokens > 0

    def test_deterministic(self):
        """Same input must always produce same token count."""
        text = "The quick brown fox."
        t1 = count_tokens(text)
        t2 = count_tokens(text)
        assert t1 == t2


# ═══════════════════════════════════════════════════════════════════════════
# analyze_tokens() — INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestAnalyzeTokens:
    """Tests for the analyze_tokens() function."""

    def test_empty_bundle(self):
        """Empty bundle must produce zero-everything breakdown."""
        bundle = ContextBundle(items=[])
        tb = analyze_tokens(bundle)
        assert tb.total_tokens == 0
        assert tb.estimated_cost_usd == 0.0
        assert tb.wasted_tokens == 0
        assert tb.by_type == {}

    def test_single_item(self):
        bundle = ContextBundle(items=[
            ContextItem(type=ContextType.SYSTEM, content="You are helpful.")
        ])
        tb = analyze_tokens(bundle)
        assert tb.total_tokens > 0
        assert "system" in tb.by_type
        assert tb.by_type["system"] == tb.total_tokens

    def test_sets_token_count_on_items(self):
        """analyze_tokens must set token_count on each ContextItem as a side-effect."""
        item = ContextItem(type=ContextType.MESSAGE, content="Hello world")
        assert item.token_count == 0  # default
        bundle = ContextBundle(items=[item])
        analyze_tokens(bundle)
        assert item.token_count > 0

    def test_by_type_aggregation(self):
        """Tokens must be correctly aggregated by type."""
        bundle = ContextBundle(items=[
            ContextItem(type=ContextType.SYSTEM, content="system prompt text here"),
            ContextItem(type=ContextType.MESSAGE, content="user question"),
            ContextItem(type=ContextType.MESSAGE, content="assistant answer"),
            ContextItem(type=ContextType.RETRIEVAL, content="retrieved chunk data"),
        ])
        tb = analyze_tokens(bundle)

        assert "system" in tb.by_type
        assert "message" in tb.by_type
        assert "retrieval" in tb.by_type
        # Sum of by_type must equal total
        assert sum(tb.by_type.values()) == tb.total_tokens

    def test_cost_calculation(self):
        """Cost must be (total_tokens / 1000) * cost_per_1k."""
        bundle = ContextBundle(items=[
            ContextItem(type=ContextType.SYSTEM, content="x " * 500)
        ])
        cost_per_1k = 0.01
        tb = analyze_tokens(bundle, cost_per_1k=cost_per_1k)
        expected_cost = (tb.total_tokens / 1000) * cost_per_1k
        assert abs(tb.estimated_cost_usd - expected_cost) < 1e-10

    def test_custom_cost_per_1k(self):
        """Custom cost_per_1k must propagate correctly."""
        bundle = ContextBundle(items=[
            ContextItem(type=ContextType.MESSAGE, content="test input")
        ])
        tb_cheap = analyze_tokens(bundle, cost_per_1k=0.001)
        tb_expensive = analyze_tokens(bundle, cost_per_1k=0.1)
        assert tb_expensive.estimated_cost_usd > tb_cheap.estimated_cost_usd

    def test_wasted_tokens_default_zero(self):
        """wasted_tokens must always be 0 from analyze_tokens (set later by engine)."""
        bundle = ContextBundle(items=[
            ContextItem(type=ContextType.RETRIEVAL, content="some data"),
            ContextItem(type=ContextType.RETRIEVAL, content="some data"),  # duplicate
        ])
        tb = analyze_tokens(bundle)
        assert tb.wasted_tokens == 0  # engine sets this, not analyze_tokens

    def test_all_context_types(self):
        """All 5 context types must appear in by_type when present."""
        bundle = ContextBundle(items=[
            ContextItem(type=ContextType.SYSTEM, content="sys"),
            ContextItem(type=ContextType.MESSAGE, content="msg"),
            ContextItem(type=ContextType.RETRIEVAL, content="ret"),
            ContextItem(type=ContextType.MEMORY, content="mem"),
            ContextItem(type=ContextType.TOOL, content="tool"),
        ])
        tb = analyze_tokens(bundle)
        assert len(tb.by_type) == 5
        for type_key in ["system", "message", "retrieval", "memory", "tool"]:
            assert type_key in tb.by_type
            assert tb.by_type[type_key] > 0

    def test_empty_content_items(self):
        """Items with empty content must tokenize to 0 and not crash."""
        bundle = ContextBundle(items=[
            ContextItem(type=ContextType.SYSTEM, content=""),
            ContextItem(type=ContextType.MESSAGE, content=""),
        ])
        tb = analyze_tokens(bundle)
        assert tb.total_tokens == 0

    def test_many_items_performance(self):
        """100 items must complete without issue."""
        items = [
            ContextItem(type=ContextType.RETRIEVAL, content=f"Chunk {i} with some content text.")
            for i in range(100)
        ]
        bundle = ContextBundle(items=items)
        tb = analyze_tokens(bundle)
        assert tb.total_tokens > 0
        assert tb.by_type["retrieval"] == tb.total_tokens
