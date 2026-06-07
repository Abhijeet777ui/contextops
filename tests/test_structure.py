"""Tests for the structure analyzer and config handling."""

from contextops.analyzers.structure import analyze_structure
from contextops.core.config import ContextOpsConfig
from contextops.core.models import ContextBundle, ContextItem, ContextType, FindingSeverity


def test_structure_threshold_override() -> None:
    """Test that custom thresholds correctly suppress or trigger findings."""
    bundle = ContextBundle(items=[
        ContextItem(type=ContextType.RETRIEVAL, content="chunk", token_count=85),
        ContextItem(type=ContextType.SYSTEM, content="system", token_count=15),
    ])
    
    # By default, retrieval > 0.70 triggers high severity
    findings_default = analyze_structure(bundle)
    assert len(findings_default) == 1
    assert findings_default[0].issue == "Retrieval dominance"
    assert findings_default[0].actual_ratio == 0.85
    
    # Custom config allows up to 0.90 retrieval
    config = ContextOpsConfig(retrieval_max_ratio=0.90)
    findings_custom = analyze_structure(bundle, config=config)
    assert len(findings_custom) == 0


def test_low_diversity() -> None:
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
