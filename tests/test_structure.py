"""
Tests for the structure analyzer and config handling.

Coverage:
  - All structure imbalance triggers:
      - Retrieval dominance
      - System prompt bloat
      - Memory explosion
      - Tool output sprawl
  - Low diversity penalty
  - Config threshold overrides (both directions)
  - Edge cases: 0 tokens, no findings
"""

import pytest

from contextops.analyzers.structure import analyze_structure, ISSUES
from contextops.core.config import ContextOpsConfig
from contextops.core.models import ContextBundle, ContextItem, ContextType, FindingSeverity


# ═══════════════════════════════════════════════════════════════════════════
# IMBALANCE TRIGGERS
# ═══════════════════════════════════════════════════════════════════════════

class TestImbalanceTriggers:
    """Ensure every type of imbalance is correctly detected when thresholds are exceeded."""

    def test_retrieval_dominance(self):
        """Retrieval exceeding 0.70 defaults should trigger."""
        bundle = ContextBundle(items=[
            ContextItem(type=ContextType.RETRIEVAL, content="chunk", token_count=85),
            ContextItem(type=ContextType.SYSTEM, content="system", token_count=15),
        ])
        findings = analyze_structure(bundle)
        # Should have retrieval dominance and potentially low diversity
        issues = [f.issue for f in findings]
        assert "Retrieval dominance" in issues
        f = next(f for f in findings if f.issue == "Retrieval dominance")
        assert f.actual_ratio == 0.85
        assert f.severity == FindingSeverity.HIGH

    def test_system_prompt_bloat(self):
        """System prompt exceeding 0.50 defaults should trigger."""
        bundle = ContextBundle(items=[
            ContextItem(type=ContextType.SYSTEM, content="sys", token_count=60),
            ContextItem(type=ContextType.MESSAGE, content="msg", token_count=20),
            ContextItem(type=ContextType.RETRIEVAL, content="chunk", token_count=20),
        ])
        findings = analyze_structure(bundle)
        issues = [f.issue for f in findings]
        assert "System prompt bloat" in issues
        f = next(f for f in findings if f.issue == "System prompt bloat")
        assert f.severity == FindingSeverity.MEDIUM

    def test_memory_explosion(self):
        """Memory exceeding 0.50 defaults should trigger."""
        bundle = ContextBundle(items=[
            ContextItem(type=ContextType.MEMORY, content="mem1", token_count=40),
            ContextItem(type=ContextType.MEMORY, content="mem2", token_count=20),
            ContextItem(type=ContextType.SYSTEM, content="sys", token_count=40),
        ])
        findings = analyze_structure(bundle)
        issues = [f.issue for f in findings]
        assert "Memory explosion" in issues
        f = next(f for f in findings if f.issue == "Memory explosion")
        assert f.severity == FindingSeverity.HIGH

    def test_tool_output_sprawl(self):
        """Tool output exceeding 0.60 defaults should trigger."""
        bundle = ContextBundle(items=[
            ContextItem(type=ContextType.TOOL, content="tool", token_count=70),
            ContextItem(type=ContextType.SYSTEM, content="sys", token_count=30),
        ])
        findings = analyze_structure(bundle)
        issues = [f.issue for f in findings]
        assert "Tool output sprawl" in issues
        f = next(f for f in findings if f.issue == "Tool output sprawl")
        assert f.severity == FindingSeverity.MEDIUM


# ═══════════════════════════════════════════════════════════════════════════
# LOW DIVERSITY
# ═══════════════════════════════════════════════════════════════════════════

class TestLowDiversity:
    """Ensure 'Low context diversity' is triggered correctly."""

    def test_low_diversity_single_type(self):
        """Test that a bundle with only one type (and >1 item) triggers low diversity finding."""
        bundle = ContextBundle(items=[
            ContextItem(type=ContextType.RETRIEVAL, content="chunk1", token_count=10),
            ContextItem(type=ContextType.RETRIEVAL, content="chunk2", token_count=10),
        ])
        
        config = ContextOpsConfig(retrieval_max_ratio=1.0) # suppress threshold error
        findings = analyze_structure(bundle, config=config)
        
        assert len(findings) == 1
        assert findings[0].issue == "Low context diversity"
        assert findings[0].severity == FindingSeverity.LOW

    def test_single_item_no_diversity_penalty(self):
        """A bundle with exactly 1 item should not get a low diversity penalty."""
        bundle = ContextBundle(items=[
            ContextItem(type=ContextType.SYSTEM, content="sys", token_count=100)
        ])
        config = ContextOpsConfig(system_max_ratio=1.0)
        findings = analyze_structure(bundle, config=config)
        assert len(findings) == 0

    def test_sufficient_diversity(self):
        """A bundle with 2 or more types should not get low diversity."""
        bundle = ContextBundle(items=[
            ContextItem(type=ContextType.SYSTEM, content="sys", token_count=50),
            ContextItem(type=ContextType.MESSAGE, content="msg", token_count=50),
        ])
        findings = analyze_structure(bundle)
        issues = [f.issue for f in findings]
        assert "Low context diversity" not in issues


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG OVERRIDES
# ═══════════════════════════════════════════════════════════════════════════

class TestConfigOverrides:
    """Test that custom thresholds correctly suppress or trigger findings."""

    def test_structure_threshold_override_suppress(self):
        bundle = ContextBundle(items=[
            ContextItem(type=ContextType.RETRIEVAL, content="chunk", token_count=85),
            ContextItem(type=ContextType.SYSTEM, content="system", token_count=15),
        ])
        
        # Custom config allows up to 0.90 retrieval
        config = ContextOpsConfig(retrieval_max_ratio=0.90)
        findings_custom = analyze_structure(bundle, config=config)
        # Should only have low diversity, not retrieval dominance
        issues = [f.issue for f in findings_custom]
        assert "Retrieval dominance" not in issues

    def test_structure_threshold_override_trigger(self):
        """Lowering a threshold should cause a finding to appear."""
        bundle = ContextBundle(items=[
            ContextItem(type=ContextType.SYSTEM, content="sys", token_count=30),
            ContextItem(type=ContextType.MESSAGE, content="msg", token_count=70),
        ])
        
        # Default system threshold is 0.50, so 0.30 actual is fine
        findings_default = analyze_structure(bundle)
        issues = [f.issue for f in findings_default]
        assert "System prompt bloat" not in issues

        # Override to 0.20
        config = ContextOpsConfig(system_max_ratio=0.20)
        findings_custom = analyze_structure(bundle, config=config)
        issues_custom = [f.issue for f in findings_custom]
        assert "System prompt bloat" in issues_custom


# ═══════════════════════════════════════════════════════════════════════════
# EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    
    def test_zero_tokens(self):
        """A bundle with 0 tokens should return no findings and not crash."""
        bundle = ContextBundle(items=[
            ContextItem(type=ContextType.SYSTEM, content="", token_count=0),
            ContextItem(type=ContextType.RETRIEVAL, content="", token_count=0),
        ])
        findings = analyze_structure(bundle)
        assert len(findings) == 0

    def test_empty_bundle(self):
        """An empty bundle should return no findings."""
        bundle = ContextBundle(items=[])
        findings = analyze_structure(bundle)
        assert len(findings) == 0

    def test_perfectly_balanced(self):
        """A well-balanced bundle should have no findings."""
        bundle = ContextBundle(items=[
            ContextItem(type=ContextType.SYSTEM, content="sys", token_count=20),
            ContextItem(type=ContextType.MESSAGE, content="msg", token_count=20),
            ContextItem(type=ContextType.RETRIEVAL, content="chunk", token_count=40),
            ContextItem(type=ContextType.TOOL, content="tool", token_count=20),
        ])
        findings = analyze_structure(bundle)
        assert len(findings) == 0
