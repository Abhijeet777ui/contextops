import pytest
from contextops.analyzers.density import compute_density_signal
from contextops.core.models import ContextBundle, ContextItem, ContextType


def _make_item(content: str, ctx_type: ContextType = ContextType.RETRIEVAL) -> ContextItem:
    return ContextItem(
        type=ctx_type,
        content=content,
        token_count=len(content.split()),
    )


def test_compute_density_signal_empty():
    bundle = ContextBundle(items=[])
    signal = compute_density_signal(bundle)
    assert signal.format_overhead == 0.0
    assert signal.whitespace_waste == 0.0
    assert signal.entropy_compression == 0.0
    assert signal.total_density_signal == 0.0


def test_compute_density_signal_high_whitespace():
    item = _make_item("def foo():\n\n\n\n    pass\n\n\n\n")
    bundle = ContextBundle(items=[item])
    signal = compute_density_signal(bundle)
    assert signal.whitespace_waste > 0.0


def test_compute_density_signal_high_format():
    item = _make_item('{ "key1": "value1", "key2": "value2", "key3": "value3" }')
    bundle = ContextBundle(items=[item])
    signal = compute_density_signal(bundle)
    assert signal.format_overhead > 0.0


def test_compute_density_signal_high_entropy():
    # Single word repeated — max entropy compression
    item = _make_item("aaaa aaaa aaaa aaaa aaaa aaaa aaaa aaaa")
    bundle = ContextBundle(items=[item])
    signal = compute_density_signal(bundle)
    assert signal.entropy_compression > 0.0


def test_fo_and_wl_are_orthogonal():
    """FO and WL must be non-overlapping: FO measures syntax, WL measures whitespace."""
    text = "hello { world } \n\n"
    item = _make_item(text)
    bundle = ContextBundle(items=[item])
    signal = compute_density_signal(bundle)

    # Both can be > 0 but their sum must not exceed 1.0
    # (payload takes the remainder)
    assert signal.format_overhead + signal.whitespace_waste <= 1.0, (
        "FO + WL exceeded 1.0 — signals are overlapping (not orthogonal)."
    )


def test_fo_wl_ec_sum_within_range():
    """Each component must independently be in [0, 1]."""
    item = _make_item("{ key: value }\n\n{ key: value }\n\n")
    bundle = ContextBundle(items=[item])
    signal = compute_density_signal(bundle)

    assert 0.0 <= signal.format_overhead <= 1.0
    assert 0.0 <= signal.whitespace_waste <= 1.0
    assert 0.0 <= signal.entropy_compression <= 1.0
    assert 0.0 <= signal.total_density_signal <= 1.0
