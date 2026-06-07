"""
ContextOps Stability API.

Runs deterministic perturbations against a context bundle to verify the scoring
engine behaves logically. Testing properties and invariants over specific scores.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any

from contextops.core.engine import analyze
from contextops.core.models import ContextItem, ContextType, RedundancyClassification
from contextops.core.normalizer import normalize


@dataclass
class InvariantResult:
    """The outcome of a single stability invariant check."""
    name: str
    passed: bool
    severity: str = "critical"
    diagnostic_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class StabilityReport:
    """Complete stability report containing all invariant checks."""
    base_score: int = 0
    base_tokens: int = 0
    base_waste_tokens: int = 0
    invariants: list[InvariantResult] = field(default_factory=list)

    @property
    def score_percentage(self) -> int:
        """Percentage of invariants that passed."""
        if not self.invariants:
            return 0
        passed = sum(1 for inv in self.invariants if inv.passed)
        return int((passed / len(self.invariants)) * 100)


def run_stability_report(raw_input: str | list[dict] | dict) -> StabilityReport:
    """
    Run the formal sanity-check layer for the scoring engine.
    
    Applies deterministic mutations to the context bundle and verifies
    that the system behaves logically.
    """
    base_bundle = normalize(raw_input)
    base_result = analyze(base_bundle)
    base_score = base_result.score

    invariants = []

    # 1. Shuffle Invariant
    # ContextOps should care about content, not ordering.
    shuffled_bundle = copy.deepcopy(base_bundle)
    shuffled_bundle.items = sorted(shuffled_bundle.items, key=lambda x: x.id, reverse=True)
    shuffle_result = analyze(shuffled_bundle)
    invariants.append(InvariantResult(
        name="Shuffle Invariant",
        passed=(shuffle_result.score == base_score),
        severity="critical",
    ))

    # 2. Duplicate Injection
    # Injecting an exact duplicate must be detected and penalized.
    dup_passed = True
    dup_diagnostic = {}
    if base_bundle.items:
        retrieval_items = base_bundle.items_by_type(ContextType.RETRIEVAL)
        if not retrieval_items:
            retrieval_items = base_bundle.items

        dup_bundle = copy.deepcopy(base_bundle)
        item_to_dup = copy.deepcopy(retrieval_items[0])
        item_to_dup.id = item_to_dup.id + "_dup"
        dup_bundle.items.append(item_to_dup)

        dup_result = analyze(dup_bundle)
        score_delta = dup_result.score - base_score

        dup_passed = (dup_result.score < base_score)
        dup_diagnostic = {
            "Score Delta": f"{score_delta:+d}",
            "Expected Direction": "Decrease",
        }
    else:
        dup_diagnostic = {"Note": "No items to duplicate"}

    invariants.append(InvariantResult(
        name="Duplicate Injection",
        passed=dup_passed,
        severity="critical",
        diagnostic_info=dup_diagnostic
    ))

    # 3. Noise Injection
    # Pure synthetic noise shouldn't magically improve the score.
    noise_bundle = copy.deepcopy(base_bundle)
    noise_content = " ".join([f"TOKEN_{i:04d}" for i in range(1, 101)])
    noise_item = ContextItem(
        type=ContextType.RETRIEVAL,
        content=noise_content,
        source="synthetic_noise"
    )
    noise_bundle.items.append(noise_item)
    noise_result = analyze(noise_bundle)
    noise_score_delta = noise_result.score - base_score
    invariants.append(InvariantResult(
        name="Noise Injection",
        passed=(noise_result.score <= base_score),
        severity="important",
        diagnostic_info={
            "Score Delta": f"{noise_score_delta:+d}",
            "Expected Direction": "<= 0",
        }
    ))

    # 4. Chunk Split Invariant
    # Splitting content shouldn't dramatically alter conclusions.
    split_passed = True
    split_diagnostic = {}
    if base_bundle.items:
        split_bundle = copy.deepcopy(base_bundle)
        non_system_indices = [
            i for i, item in enumerate(split_bundle.items) 
            if item.type != ContextType.SYSTEM
        ]
        if non_system_indices:
            longest_idx = max(non_system_indices, key=lambda i: len(split_bundle.items[i].content))
            longest_item = split_bundle.items.pop(longest_idx)

            mid = len(longest_item.content) // 2
            part1 = longest_item.content[:mid]
            part2 = longest_item.content[mid:]

            item1 = ContextItem(type=longest_item.type, content=part1, source=longest_item.source)
            item2 = ContextItem(type=longest_item.type, content=part2, source=longest_item.source)

            split_bundle.items.append(item1)
            split_bundle.items.append(item2)

            split_result = analyze(split_bundle)
            split_delta = split_result.score - base_score

            DEFAULT_SPLIT_TOLERANCE = 10
            split_passed = (abs(split_delta) <= DEFAULT_SPLIT_TOLERANCE)
            split_diagnostic = {
                "Base Score": base_score,
                "Split Score": split_result.score,
                "Delta": f"{split_delta:+d}",
            }
        else:
            split_diagnostic = {"Note": "No non-system items to split"}
    else:
        split_diagnostic = {"Note": "No items to split"}

    invariants.append(InvariantResult(
        name="Chunk Split Invariant",
        passed=split_passed,
        severity="important",
        diagnostic_info=split_diagnostic
    ))

    # 5. Boilerplate Invariant
    # Tests the core philosophy: expected repetition vs real waste.
    bp_bundle = copy.deepcopy(base_bundle)
    bp_item = ContextItem(
        type=ContextType.SYSTEM,
        content="You are a helpful assistant. Please follow all instructions carefully.",
        source="system"
    )
    bp_item_dup = copy.deepcopy(bp_item)
    bp_item_dup.id = bp_item.id + "_dup"

    bp_bundle.items.extend([bp_item, bp_item_dup])
    bp_result = analyze(bp_bundle)

    # Only check findings between the two injected boilerplate items
    bp_ids = {bp_item.id, bp_item_dup.id}
    bp_pair_findings = [
        f for f in bp_result.redundancy_findings
        if f.item_a_id in bp_ids and f.item_b_id in bp_ids
    ]

    detected_bp = any(
        f.classification == RedundancyClassification.BOILERPLATE
        for f in bp_pair_findings
    )
    detected_redundant = any(
        f.classification == RedundancyClassification.REDUNDANT_CONTEXT
        for f in bp_pair_findings
    )

    invariants.append(InvariantResult(
        name="Boilerplate Invariant",
        passed=(detected_bp and not detected_redundant),
        severity="critical",
        diagnostic_info={
            "Detected as BOILERPLATE": detected_bp,
            "Detected as REDUNDANT_CONTEXT": detected_redundant,
        }
    ))

    # 6. Semantic Blindness Guard
    # Semantic blindness is a feature, not a bug. Verify it stays that way.
    SEMANTIC_BLINDNESS_CASES = [
        (
            "The startup raised one million dollars.",
            "The company secured $1M in funding."
        ),
        (
            "The API request timed out after thirty seconds.",
            "The endpoint exceeded its 30-second timeout threshold."
        ),
        (
            "The quick brown fox jumps over the lazy dog.",
            "A fast dark-colored canine leaped above a resting dog."
        ),
    ]

    sb_passed = True
    sb_diagnostics = []

    for case_idx, (text1, text2) in enumerate(SEMANTIC_BLINDNESS_CASES):
        sb_bundle = copy.deepcopy(base_bundle)
        item1 = ContextItem(type=ContextType.RETRIEVAL, content=text1, source=f"sb_a_{case_idx}")
        item2 = ContextItem(type=ContextType.RETRIEVAL, content=text2, source=f"sb_b_{case_idx}")
        sb_bundle.items.extend([item1, item2])

        sb_result = analyze(sb_bundle)

        has_redundancy = any(
            f.classification == RedundancyClassification.REDUNDANT_CONTEXT
            and ((f.item_a_id == item1.id and f.item_b_id == item2.id) or
                 (f.item_a_id == item2.id and f.item_b_id == item1.id))
            for f in sb_result.redundancy_findings
        )

        if has_redundancy:
            sb_passed = False
            sb_diagnostics.append(f"Case {case_idx+1} triggered redundancy")

    invariants.append(InvariantResult(
        name="Semantic Blindness Guard",
        passed=sb_passed,
        severity="important",
        diagnostic_info={
            "Redundancy Detected": not sb_passed,
            "Details": ", ".join(sb_diagnostics) if sb_diagnostics else "Clean across all cases",
        }
    ))

    return StabilityReport(
        base_score=base_score,
        base_tokens=base_result.token_breakdown.total_tokens,
        base_waste_tokens=base_result.token_breakdown.wasted_tokens,
        invariants=invariants
    )
