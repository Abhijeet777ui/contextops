"""
Token Analyzer.

Uses tiktoken to count tokens per ContextItem and estimate costs.
Nothing fancy — just reliable counting and cost math.
"""

from __future__ import annotations

import tiktoken

from contextops.core.models import ContextBundle, TokenBreakdown




def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count tokens for a given text using tiktoken.

    Args:
        text: The text to tokenize.
        model: The model name for the encoding. Defaults to gpt-4o.

    Returns:
        The number of tokens.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base (covers GPT-4, GPT-3.5, etc.)
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def analyze_tokens(
    bundle: ContextBundle,
    model: str = "gpt-4o",
) -> TokenBreakdown:
    """
    Count tokens for every item in the bundle and produce a breakdown.

    Side effect: sets token_count on each ContextItem in the bundle.

    Args:
        bundle: The context bundle to analyze.
        model: Model name for tiktoken encoding selection.

    Returns:
        A TokenBreakdown with totals, per-type distribution, and cost estimate.
    """
    by_type: dict[str, int] = {}
    total = 0

    for item in bundle.items:
        tokens = count_tokens(item.content, model=model)
        item.token_count = tokens
        total += tokens

        type_key = item.type.value
        by_type[type_key] = by_type.get(type_key, 0) + tokens

    return TokenBreakdown(
        total_tokens=total,
        by_type=by_type,
        estimated_reduction_pct=0.0,  # filled in later by engine
        wasted_tokens=0,  # filled in later by the engine after redundancy analysis
    )
