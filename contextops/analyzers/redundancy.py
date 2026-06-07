"""
Redundancy Analyzer — Unified Signal Architecture.

All redundancy detection is driven by a single, deterministic
Redundancy Signal function RS(i, j) → float ∈ [0, 1].

Design principles:
  - Single source of truth: findings drive both UI output AND penalty math.
  - No split-brain: there is ONE waste calculation, not two.
  - No hard threshold cliff: RS is continuous, not binary.
  - Semantic-blind: tokens + character overlap only. No embeddings.
  - Short-text aware: length weighting prevents sparse-data instability.

RS(i,j) = (0.6 × Jaccard(tokens) + 0.3 × ngram_overlap(N=4) + 0.1 × char_overlap)
         × length_weight
         × classification_modifier
"""

from __future__ import annotations

import re
import string

from contextops.core.models import (
    ContextBundle,
    ContextItem,
    RedundancyClassification,
    RedundancyFinding,
)


# ── Constants ────────────────────────────────────────────────────────────

# Minimum RS to produce a finding (replaces hard threshold cliff).
# A continuous signal: RS = 0.12 means "very mild overlap", RS = 1.0 = exact copy.
RS_MINIMUM: float = 0.12

# Short texts need at least this many tokens to produce a full-weight RS.
# Below this, the length weight dampens the signal proportionally.
RS_LENGTH_NORMALIZER: int = 12

# N-gram window for per-pair N-gram overlap. Smaller than the old 8/12/16
# global scan so short sentences (4–7 words) are still captured.
NGRAM_N: int = 4

# Max items before switching to hash-bucket pairwise strategy (O(n²) cap).
MAX_PAIRWISE_ITEMS: int = 50

# Boilerplate modifier: overlap between two boilerplate items carries near-zero weight.
BOILERPLATE_RS_MODIFIER: float = 0.0

# Adjacent-chunk modifier: overlap between adjacent chunks from the same
# source is expected — penalise at 20% of full RS.
ADJACENT_RS_MODIFIER: float = 0.2

# Words that indicate boilerplate when they dominate the content.
BOILERPLATE_SIGNALS: set[str] = {
    "please", "always", "must", "never", "ensure", "remember",
    "important", "note", "follow", "instructions", "guidelines",
    "rules", "format", "respond", "output", "do not",
}

# Lightweight synonym map for intent normalisation (deterministic, no embeddings).
SYNONYM_MAP: dict[str, str] = {
    "concise": "short",
    "brief": "short",
    "minimal": "short",
    "quickly": "fast",
    "rapidly": "fast",
    "accurate": "correct",
    "exact": "correct",
    "precise": "correct",
}


# ── Tokenisation helpers ─────────────────────────────────────────────────

def _get_source_base(s: str) -> str:
    """Strip trailing numeric suffixes to get a canonical source base name."""
    base = re.sub(r'[_\-]?\d+(?:\.[a-zA-Z0-9]+)?$', '', s)
    return base if base else s


def _get_ordered_tokens(text: str) -> list[str]:
    """Split text into an ordered list of lowercase words, mapping synonyms."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    return [SYNONYM_MAP.get(w, w) for w in words]


def _tokenize_words(text: str) -> set[str]:
    """Split text into a lowercase word set, stripping punctuation."""
    return set(_get_ordered_tokens(text))


# ── RS sub-components ────────────────────────────────────────────────────

def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two word sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def _compute_ngram_overlap(tokens_a: list[str], tokens_b: list[str], n: int = NGRAM_N) -> float:
    """
    Compute N-gram overlap ratio between two token lists.

    Falls back to bigrams if either list is shorter than N.
    Returns 0.0 if too short even for bigrams.
    """
    if n > 2 and (len(tokens_a) < n or len(tokens_b) < n):
        return _compute_ngram_overlap(tokens_a, tokens_b, n=2)
    if len(tokens_a) < 2 or len(tokens_b) < 2:
        return 0.0

    ngrams_a = {tuple(tokens_a[i:i + n]) for i in range(len(tokens_a) - n + 1)}
    ngrams_b = {tuple(tokens_b[i:i + n]) for i in range(len(tokens_b) - n + 1)}

    if not ngrams_a or not ngrams_b:
        return 0.0

    union = ngrams_a | ngrams_b
    if not union:
        return 0.0
    return len(ngrams_a & ngrams_b) / len(union)


def _compute_char_overlap(text_a: str, text_b: str) -> float:
    """Compute character-level Jaccard similarity (case-insensitive)."""
    chars_a = set(text_a.lower())
    chars_b = set(text_b.lower())
    if not chars_a and not chars_b:
        return 0.0
    return len(chars_a & chars_b) / len(chars_a | chars_b)


def _compute_rs(item_a: ContextItem, item_b: ContextItem) -> float:
    """
    Compute the unified Redundancy Signal RS(i,j) ∈ [0, 1].

    RS = (0.6 × Jaccard(tokens) + 0.3 × ngram_overlap + 0.1 × char_overlap)
         × length_weight

    length_weight = min(1.0, min_tokens / RS_LENGTH_NORMALIZER)

    Signal contract:
      - Reads ONLY from item content and tokens.
      - No embeddings, no external APIs, no difflib.
      - Fully deterministic.
    """
    tokens_a = _get_ordered_tokens(item_a.content)
    tokens_b = _get_ordered_tokens(item_b.content)

    words_a = set(tokens_a)
    words_b = set(tokens_b)

    jaccard = _jaccard_similarity(words_a, words_b)
    ngram = _compute_ngram_overlap(tokens_a, tokens_b)
    char_overlap = _compute_char_overlap(item_a.content, item_b.content)

    # Length-aware weight: short text is sparse — downscale to avoid false positives.
    # Use actual word count (always available) as the length measure.
    min_len = min(len(tokens_a), len(tokens_b))
    length_weight = min(1.0, min_len / RS_LENGTH_NORMALIZER)

    rs = (0.6 * jaccard + 0.3 * ngram + 0.1 * char_overlap) * length_weight
    return round(min(1.0, rs), 4)


# ── Classification helpers ───────────────────────────────────────────────

def _is_adjacent_source(item_a: ContextItem, item_b: ContextItem) -> bool:
    """
    Check if two items are from adjacent positions in the same source.

    Adjacent chunks from the same document are expected to overlap.
    This prevents false positives in RAG pipelines with sliding windows.
    """
    if not item_a.source or not item_b.source:
        return False

    idx_a = item_a.metadata.get("index") or item_a.metadata.get("chunk_index")
    idx_b = item_b.metadata.get("index") or item_b.metadata.get("chunk_index")

    if idx_a is not None and idx_b is not None:
        try:
            return abs(int(idx_a) - int(idx_b)) <= 1
        except (ValueError, TypeError):
            pass

    base_a = _get_source_base(item_a.source)
    base_b = _get_source_base(item_b.source)
    return base_a == base_b and item_a.source != item_b.source


def _is_boilerplate(item: ContextItem) -> bool:
    """
    Check if an item's content is primarily boilerplate instructions.

    Requires at least 3 words so ultra-short phrases don't mis-fire.
    """
    words = _get_ordered_tokens(item.content)
    if len(words) < 3:
        return False
    signal_count = sum(1 for w in words if w in BOILERPLATE_SIGNALS)
    return (signal_count / len(words)) > 0.25


def _classify(item_a: ContextItem, item_b: ContextItem) -> RedundancyClassification:
    """
    Classify the type of redundancy between two items.

    Rules (in priority order):
    1. Adjacent chunks from same source → EXPECTED_OVERLAP
    2. Both are boilerplate → BOILERPLATE
    3. Everything else with RS >= RS_MINIMUM → REDUNDANT_CONTEXT
    """
    if _is_adjacent_source(item_a, item_b):
        return RedundancyClassification.EXPECTED_OVERLAP
    if _is_boilerplate(item_a) and _is_boilerplate(item_b):
        return RedundancyClassification.BOILERPLATE
    return RedundancyClassification.REDUNDANT_CONTEXT


def _get_rs_modifier(classification: RedundancyClassification) -> float:
    """Return the RS modifier for a given classification."""
    if classification == RedundancyClassification.BOILERPLATE:
        return BOILERPLATE_RS_MODIFIER
    if classification == RedundancyClassification.EXPECTED_OVERLAP:
        return ADJACENT_RS_MODIFIER
    return 1.0


# ── Main entry point ─────────────────────────────────────────────────────

def analyze_redundancy(bundle: ContextBundle) -> tuple[list[RedundancyFinding], int]:
    """
    Detect redundant pairs in a ContextBundle using the unified RS signal.

    Architecture:
      RS(i,j) → classification → modifier → finding → waste aggregation

    Findings are the SINGLE SOURCE OF TRUTH for:
      - Human-readable output (issue + fix)
      - Waste token count (fed back into engine for scoring)

    Returns:
        tuple of (list[RedundancyFinding], final_wasted_tokens)
        where final_wasted_tokens is the sum of REDUNDANT_CONTEXT finding waste.
    """
    findings: list[RedundancyFinding] = []
    items = bundle.items

    # Determine which pairs to compare.
    if len(items) <= MAX_PAIRWISE_ITEMS:
        pairs_to_check = [
            (i, j)
            for i in range(len(items))
            for j in range(i + 1, len(items))
        ]
    else:
        # Hash-bucket fast path: group items by stripped content hash.
        from hashlib import md5
        buckets: dict[str, list[int]] = {}
        for idx, item in enumerate(items):
            h = md5(item.content.strip().lower().encode()).hexdigest()[:16]
            buckets.setdefault(h, []).append(idx)

        pairs_to_check = []
        # Always compare within same hash bucket (exact/near duplicates)
        for indices in buckets.values():
            for a_pos in range(len(indices)):
                for b_pos in range(a_pos + 1, len(indices)):
                    pairs_to_check.append((indices[a_pos], indices[b_pos]))

        # Add a bounded sample of cross-bucket pairs for fuzzy detection.
        bucket_keys = list(buckets.keys())
        cross_pairs_added = 0
        max_cross_pairs = MAX_PAIRWISE_ITEMS * 10
        for bi in range(len(bucket_keys)):
            if cross_pairs_added >= max_cross_pairs:
                break
            for bj in range(bi + 1, len(bucket_keys)):
                if cross_pairs_added >= max_cross_pairs:
                    break
                pairs_to_check.append(
                    (buckets[bucket_keys[bi]][0], buckets[bucket_keys[bj]][0])
                )
                cross_pairs_added += 1

    for i_idx, j_idx in pairs_to_check:
        item_a = items[i_idx]
        item_b = items[j_idx]

        # Skip empty content.
        if not item_a.content.strip() or not item_b.content.strip():
            continue

        # Compute the unified RS signal.
        rs = _compute_rs(item_a, item_b)

        # Below the minimum RS floor → not a meaningful overlap.
        if rs < RS_MINIMUM:
            continue

        classification = _classify(item_a, item_b)
        modifier = _get_rs_modifier(classification)
        rs_effective = round(rs * modifier, 4)

        # Waste = smaller item's tokens × effective RS.
        # This makes waste proportional to signal strength, not binary.
        raw_waste = min(item_a.token_count, item_b.token_count)
        waste = int(raw_waste * rs_effective)

        detail = _build_detail(item_a, item_b, rs, classification)

        findings.append(RedundancyFinding(
            item_a_id=item_a.id,
            item_b_id=item_b.id,
            similarity_score=rs,
            classification=classification,
            estimated_waste_tokens=waste,
            detail=detail,
        ))

    # Sort for CI determinism: waste desc, similarity desc, id_a, id_b.
    findings.sort(
        key=lambda f: (-f.estimated_waste_tokens, -f.similarity_score, f.item_a_id, f.item_b_id)
    )

    # Findings are the single source of truth for total waste.
    # Only REDUNDANT_CONTEXT findings count as true waste.
    final_wasted_tokens = sum(
        f.estimated_waste_tokens
        for f in findings
        if f.classification == RedundancyClassification.REDUNDANT_CONTEXT
    )

    return findings, final_wasted_tokens


# ── Detail builder ───────────────────────────────────────────────────────

def _build_detail(
    item_a: ContextItem,
    item_b: ContextItem,
    rs: float,
    classification: RedundancyClassification,
) -> str:
    """Build a human-readable explanation of the finding."""
    sim_pct = f"{rs * 100:.0f}%"

    if classification == RedundancyClassification.EXPECTED_OVERLAP:
        return (
            f"{sim_pct} similarity between '{item_a.source}' and '{item_b.source}' "
            f"— expected overlap (adjacent chunks)"
        )
    elif classification == RedundancyClassification.BOILERPLATE:
        return (
            f"{sim_pct} similarity — both items contain boilerplate instructions"
        )
    else:
        src_a = item_a.source or item_a.type.value
        src_b = item_b.source or item_b.type.value
        return (
            f"{sim_pct} similarity between '{src_a}' and '{src_b}' "
            f"— redundant context from independent sources"
        )
