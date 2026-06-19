"""
ContextOps Core Data Models.

These are the canonical data structures that every module in the system
operates on. ContextItem and ContextBundle are the internal representation
of LLM context — everything gets normalized into this form.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from contextops.core.roast import RoastResult


class ContextType(str, Enum):
    """Classification of a context item's origin."""
    SYSTEM = "system"
    MEMORY = "memory"
    RETRIEVAL = "retrieval"
    TOOL = "tool"
    MESSAGE = "message"


class RedundancyClassification(str, Enum):
    """How we classify detected overlap between context items."""
    EXPECTED_OVERLAP = "expected_overlap"       # adjacent chunks, normal RAG behavior
    REDUNDANT_CONTEXT = "redundant_context"     # unnecessary duplication, real waste
    BOILERPLATE = "boilerplate"                 # repeated template/instructions


class FindingSeverity(str, Enum):
    """Severity level for analysis findings."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ContextItem:
    """
    A single unit of context fed into an LLM.

    This is the atomic unit of the entire system. Every chunk, message,
    system prompt, memory entry, or tool output becomes a ContextItem.
    """
    type: ContextType
    content: str
    token_count: int = 0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.type, str):
            self.type = ContextType(self.type)


@dataclass
class ContextBundle:
    """
    The complete context being sent to an LLM.

    This is a list of ContextItems. Every analyzer, the scoring engine,
    and the recommendation engine operate exclusively on ContextBundle.
    """
    items: list[ContextItem] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        """Total token count across all items."""
        return sum(item.token_count for item in self.items)

    @property
    def item_count(self) -> int:
        """Number of context items."""
        return len(self.items)

    def items_by_type(self, context_type: ContextType) -> list[ContextItem]:
        """Filter items by their context type."""
        return [item for item in self.items if item.type == context_type]


# ── Analysis Result Models ──────────────────────────────────────────────


@dataclass
class RedundancyFinding:
    """A detected redundancy between two context items."""
    item_a_id: str
    item_b_id: str
    similarity_score: float           # 0.0 to 1.0
    classification: RedundancyClassification
    estimated_waste_tokens: int
    detail: str = ""


@dataclass
class StructureFinding:
    """A detected structural imbalance in the context distribution."""
    issue: str
    context_type: ContextType
    actual_ratio: float               # 0.0 to 1.0
    threshold: float                  # the threshold that was exceeded
    severity: FindingSeverity = FindingSeverity.MEDIUM


@dataclass
class Recommendation:
    """An actionable fix the user can apply."""
    issue: str
    impact_score: float               # estimated score improvement
    token_savings: int                # estimated tokens saved
    fix: str                          # human-readable fix instruction
    severity: FindingSeverity = FindingSeverity.MEDIUM


@dataclass
class DensitySignal:
    """Shadow metric measuring structural token waste."""
    format_overhead: float        # 0.0 to 1.0
    whitespace_waste: float       # 0.0 to 1.0
    entropy_compression: float    # 0.0 to 1.0
    total_density_signal: float   # 0.0 to 1.0
    system_divergence: float = 0.0 # 0.0 to 1.0


@dataclass
class ScoreBreakdown:
    """Decomposed penalty breakdown for the context score."""
    redundancy_penalty: float = 0.0   # 0–30
    density_penalty: float = 0.0   # 0–30
    structure_penalty: float = 0.0    # 0–20
    concentration_penalty: float = 0.0    # 0–20

    @property
    def total_penalty(self) -> float:
        return (
            self.redundancy_penalty
            + self.density_penalty
            + self.structure_penalty
            + self.concentration_penalty
        )

    @property
    def score(self) -> int:
        """Final 0–100 context score."""
        return max(0, min(100, round(100 - self.total_penalty)))


@dataclass
class TokenBreakdown:
    """Per-type token distribution."""
    total_tokens: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    estimated_cost_usd: float = 0.0
    wasted_tokens: int = 0


@dataclass
class AnalysisResult:
    """
    The complete output of a ContextOps analysis.

    This is the JSON-primary API contract. The CLI renderer reads this.
    CI mode reads this. Everything derives from this object.
    """
    score: int
    score_breakdown: ScoreBreakdown
    token_breakdown: TokenBreakdown
    redundancy_findings: list[RedundancyFinding] = field(default_factory=list)
    structure_findings: list[StructureFinding] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    mode: str = "strict"
    config_version: str = "1.0"
    density_signal: DensitySignal | None = None
    density_effect: Literal["shadow", "active"] = "shadow"
    # Roast is random per-run — explicitly non-deterministic.
    # Only populated when roast_enabled=True in config.
    roast: "RoastResult | None" = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for JSON output."""
        res = {
            "score": self.score,
            "mode": self.mode,
            "config_version": self.config_version,
            "score_breakdown": {
                "redundancy_penalty": round(self.score_breakdown.redundancy_penalty, 2),
                "density_penalty": round(self.score_breakdown.density_penalty, 2),
                "structure_penalty": round(self.score_breakdown.structure_penalty, 2),
                "concentration_penalty": round(self.score_breakdown.concentration_penalty, 2),
                "total_penalty": round(self.score_breakdown.total_penalty, 2),
            },
            "token_breakdown": {
                "total_tokens": self.token_breakdown.total_tokens,
                "by_type": self.token_breakdown.by_type,
                "estimated_cost_usd": round(self.token_breakdown.estimated_cost_usd, 6),
                "wasted_tokens": self.token_breakdown.wasted_tokens,
            },
            "findings": {
                "redundancy": [
                    {
                        "item_a": f.item_a_id,
                        "item_b": f.item_b_id,
                        "similarity": round(f.similarity_score, 3),
                        "classification": f.classification.value,
                        "waste_tokens": f.estimated_waste_tokens,
                        "detail": f.detail,
                    }
                    for f in self.redundancy_findings
                ],
                "structure": [
                    {
                        "issue": f.issue,
                        "type": f.context_type.value,
                        "actual_ratio": round(f.actual_ratio, 3),
                        "threshold": f.threshold,
                        "severity": f.severity.value,
                    }
                    for f in self.structure_findings
                ],
            },
            "recommendations": [
                {
                    "issue": r.issue,
                    "impact": f"+{round(r.impact_score, 1)} points",
                    "token_savings": r.token_savings,
                    "fix": r.fix,
                    "severity": r.severity.value,
                }
                for r in self.recommendations
            ],
            "metadata": self.metadata,
        }
        
        if self.density_signal:
            res["density_signal"] = {
                "format_overhead": round(self.density_signal.format_overhead, 3),
                "whitespace_waste": round(self.density_signal.whitespace_waste, 3),
                "entropy_compression": round(self.density_signal.entropy_compression, 3),
                "total_density_signal": round(self.density_signal.total_density_signal, 3),
            }
            res["density_effect"] = self.density_effect

        if self.roast is not None:
            res["roast"] = self.roast.to_dict()
            
        return res
