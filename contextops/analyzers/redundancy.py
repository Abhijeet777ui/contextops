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

import hashlib
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
RS_MINIMUM: float = 0.20

# Short texts need at least this many tokens to produce a full-weight RS.
# Below this, the length weight dampens the signal proportionally.
RS_LENGTH_NORMALIZER: int = 6

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


def _minhash_signature(text: str, num_hashes: int = 90) -> list[int]:
    """
    Compute a deterministic MinHash signature using 3-character shingling.
    This captures morphological structure to survive LLM paraphrasing.
    """
    text = text.lower().strip()
    
    # CAVEAT: Skip shingling entirely for short chunks (< 50 chars) 
    # to avoid false positive collisions on common English trigrams.
    if len(text) < 50:
        return []
        
    # Generate 3-character shingles
    shingles = set(text[i:i+3] for i in range(len(text) - 2))
    if not shingles:
        return []
    
    sig = []
    for i in range(num_hashes):
        min_h = float('inf')
        for s in shingles:
            # Deterministic hash: MD5 of salt + shingle
            h = int(hashlib.md5(f"{i}_{s}".encode()).hexdigest()[:8], 16)
            if h < min_h:
                min_h = h
        sig.append(min_h)
    return sig

def _minhash_similarity(sig_a: list[int], sig_b: list[int]) -> float:
    """Compute Jaccard similarity estimate from MinHash signatures."""
    if not sig_a or not sig_b:
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)

def _compute_rs(
    item_a: ContextItem,
    item_b: ContextItem,
    strict_semantic: bool = False
) -> tuple[float, bool]:
    """
    Compute the unified Redundancy Signal RS(i,j) in [0, 1].

    Returns:
        (rs, minhash_dominant): rs is the signal value; minhash_dominant
        is True when strict-semantic MinHash was the primary driver of RS.
        This allows the caller to bypass the adjacency modifier for semantically
        flagged pairs.
    """
    tokens_a = _get_ordered_tokens(item_a.content)
    tokens_b = _get_ordered_tokens(item_b.content)

    words_a = set(tokens_a)
    words_b = set(tokens_b)

    jaccard = _jaccard_similarity(words_a, words_b)
    ngram = _compute_ngram_overlap(tokens_a, tokens_b)
    char_overlap = _compute_char_overlap(item_a.content, item_b.content)

    min_len = min(len(tokens_a), len(tokens_b))
    length_weight = min(1.0, min_len / RS_LENGTH_NORMALIZER)

    rs = (0.6 * jaccard + 0.3 * ngram + 0.1 * char_overlap) * length_weight
    minhash_dominant = False
    
    # If opt-in strict semantic mode is enabled, evaluate MinHash similarity
    # to catch adversarial paraphrasing (Semantic DoS bypass).
    if strict_semantic:
        sig_a = _minhash_signature(item_a.content)
        sig_b = _minhash_signature(item_b.content)
        
        # Only compare if both items were long enough to generate signatures
        if sig_a and sig_b:
            mh_sim = _minhash_similarity(sig_a, sig_b)
            # 3-char shingling needs a lower threshold multiplier than exact word match.
            # We scale the mh_sim up slightly so ~40% shingle overlap triggers a high signal.
            scaled_mh_sim = min(1.0, mh_sim * 2.5)
            minhash_rs = scaled_mh_sim * length_weight
            if minhash_rs > rs:
                # MinHash dominates — flag this so the caller can force REDUNDANT_CONTEXT
                # classification regardless of source adjacency.
                rs = minhash_rs
                minhash_dominant = True

    return round(min(1.0, rs), 4), minhash_dominant


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

def analyze_redundancy(bundle: ContextBundle, strict_semantic: bool = False) -> tuple[list[RedundancyFinding], int]:
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

    # ── 1. Candidate Generation (O(N)) ──────────────────────────────────────
    candidates = set()

    if strict_semantic:
        # EXACT TOKEN INVERTED INDEX PATH
        # Guarantees 100% exact Jaccard recall for security-sensitive paths.
        inverted_index: dict[str, list[int]] = {}
        for item_idx, item in enumerate(items):
            tokens = _tokenize_words(item.content)
            for token in tokens:
                if token not in inverted_index:
                    inverted_index[token] = []
                for matching_idx in inverted_index[token]:
                    # Keep pairs sorted (lower_idx, higher_idx)
                    pair = (matching_idx, item_idx) if matching_idx < item_idx else (item_idx, matching_idx)
                    candidates.add(pair)
                inverted_index[token].append(item_idx)
    else:
        # FAST LSH PATH
        # 30 bands of 3 hashes for ~98% recall at 50% true Jaccard similarity.
        signatures = [_minhash_signature(item.content, num_hashes=90) for item in items]
        for band_idx in range(30):
            bucket: dict[tuple[int, ...], list[int]] = {}
            for item_idx, sig in enumerate(signatures):
                if not sig:
                    continue  # Skip ultra-short items
                band_tuple = tuple(sig[band_idx * 3 : (band_idx + 1) * 3])
                if band_tuple in bucket:
                    for matching_idx in bucket[band_tuple]:
                        pair = (matching_idx, item_idx) if matching_idx < item_idx else (item_idx, matching_idx)
                        candidates.add(pair)
                bucket.setdefault(band_tuple, []).append(item_idx)

    # Always add adjacent items (sliding window fallback)
    for i in range(len(items) - 1):
        candidates.add((i, i + 1))

    # Sort for deterministic evaluation order
    pairs_to_check = sorted(list(candidates))

    # ── 2. Pairwise Evaluation ─────────────────────────────────────────────
    for i_idx, j_idx in pairs_to_check:
        item_a = items[i_idx]
        item_b = items[j_idx]

        # Skip empty content.
        if not item_a.content.strip() or not item_b.content.strip():
            continue

        # Compute the unified RS signal.
        rs, minhash_dominant = _compute_rs(item_a, item_b, strict_semantic=strict_semantic)

        # Below the minimum RS floor → not a meaningful overlap.
        if rs < RS_MINIMUM:
            continue

        # If MinHash drove the signal (strict-semantic mode), force REDUNDANT_CONTEXT
        # classification regardless of source adjacency — the semantic signal overrides
        # the structural adjacency heuristic.
        if minhash_dominant:
            classification = RedundancyClassification.REDUNDANT_CONTEXT
        else:
            classification = _classify(item_a, item_b)
        modifier = _get_rs_modifier(classification)
        rs_effective = round(rs * modifier, 4)

        # Waste = smaller item's tokens × effective RS.
        # This makes waste proportional to signal strength, not binary.
        raw_waste = min(item_a.token_count, item_b.token_count)
        waste = max(1, int(raw_waste * rs_effective)) if raw_waste > 0 else 0

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
