"""
ContextOps Programmatic API.

This is the primary interface for using ContextOps as a library.
Import and call `inspect_context()` with any supported input format.

Example:
    from contextops.api.inspect import inspect_context

    result = inspect_context({
        "system": "You are a helpful assistant.",
        "chunks": ["chunk 1", "chunk 2"],
    })
    print(result.score)
    print(json.dumps(result.to_dict(), indent=2))
"""

from __future__ import annotations

from typing import Any

from contextops.core.config import ContextOpsConfig
from contextops.core.engine import analyze
from contextops.core.models import AnalysisResult
from contextops.core.normalizer import normalize


def inspect_context(
    raw_input: str | list[dict[str, Any]] | dict[str, Any],
    model: str = "gpt-4o",
    cost_per_1k: float = 0.005,
    config: ContextOpsConfig | None = None,
) -> AnalysisResult:
    """
    Analyze an LLM context and return a full AnalysisResult.

    This is the main entry point for the ContextOps library.

    Args:
        raw_input: Raw LLM context in any supported format:
            - str: treated as a system prompt
            - list[dict]: OpenAI-style message list
            - dict: structured dict with system/messages/chunks/memory/tools
        model: Model name for tiktoken encoding.
        cost_per_1k: Cost per 1K input tokens in USD.
        config: Optional custom threshold configuration.

    Returns:
        AnalysisResult containing score, breakdown, findings, and recommendations.
    """
    bundle = normalize(raw_input)
    return analyze(bundle, model=model, cost_per_1k=cost_per_1k, config=config)
