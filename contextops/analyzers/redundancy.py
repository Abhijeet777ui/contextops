"""
Redundancy Analyzer.

Detects near-duplicate, overlapping, and boilerplate context items.
Uses a deterministic hybrid heuristic approach:
  1. Exact match detection (fast path)
  2. Jaccard similarity on word sets

Critical design rule: NEVER blindly flag overlap as waste.
Adjacent chunks from the same source get EXPECTED_OVERLAP.
Only independent sources with high similarity get REDUNDANT_CONTEXT.
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


# ── Thresholds (fixed, deterministic, CI-safe) ──────────────────────────

EXACT_MATCH_THRESHOLD: float = 1.0
HIGH_SIMILARITY_THRESHOLD: float = 0.75
MODERATE_SIMILARITY_THRESHOLD: float = 0.45

# Words that indicate boilerplate when they dominate the content
BOILERPLATE_SIGNALS: set[str] = {
    "please", "always", "must", "never", "ensure", "remember",
    "important", "note", "follow", "instructions", "guidelines",
    "rules", "format", "respond", "output", "do not",
}

# Lightweight synonym mapping for intent duplication
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


def _get_source_base(s: str) -> str:
    """Strip trailing numeric suffixes and extensions to get a canonical source base name.

    Examples:
        'doc_1'  -> 'doc'
        'page1.md' -> 'page'
        'chunk-3' -> 'chunk'
        'readme.md' -> 'readme.md'  (no suffix stripped)
    """
    base = re.sub(r'[_\-]?\d+(?:\.[a-zA-Z0-9]+)?$', '', s)
    return base if base else s


def _get_ordered_tokens(text: str) -> list[str]:
    """Split text into an ordered list of lowercase words, mapping synonyms."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    return [SYNONYM_MAP.get(w, w) for w in words]

def _tokenize_words(text: str) -> set[str]:
    """Split text into a lowercase word set, stripping punctuation and mapping synonyms."""
    return set(_get_ordered_tokens(text))


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """
    Compute Jaccard similarity between two word sets.

    Returns 0.0 if both sets are empty, otherwise |intersection| / |union|.
    Deterministic. No randomness. CI-safe.
    """
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def _is_adjacent_source(item_a: ContextItem, item_b: ContextItem) -> bool:
    """
    Check if two items are from adjacent positions in the same source.

    Adjacent chunks from the same document are expected to overlap.
    This prevents false positives in RAG pipelines with sliding windows.
    """
    if not item_a.source or not item_b.source:
        return False

    # Same source base (e.g., "chunk_3" and "chunk_4" from same doc)
    source_a = item_a.source
    source_b = item_b.source

    # Check if metadata indicates adjacency
    idx_a = item_a.metadata.get("index") or item_a.metadata.get("chunk_index")
    idx_b = item_b.metadata.get("index") or item_b.metadata.get("chunk_index")

    if idx_a is not None and idx_b is not None:
        try:
            return abs(int(idx_a) - int(idx_b)) <= 1
        except (ValueError, TypeError):
            pass

    # Check if sources share a base name (e.g., "docs/api.md", "page1.md", "doc-2")
    base_a = _get_source_base(source_a)
    base_b = _get_source_base(source_b)
    return base_a == base_b and source_a != source_b


def _is_boilerplate(item: ContextItem) -> bool:
    """
    Check if an item's content is primarily boilerplate instructions.

    Returns True if a high proportion of word occurrences match boilerplate signals.
    Uses an ordered token list (not a set) so word frequency is correctly measured.
    e.g. "please please please do this" → 3/5 = 60% signal density, correctly fires.
    """
    words = _get_ordered_tokens(item.content)   # list, preserves frequency
    if len(words) < 5:
        return False
    signal_count = sum(1 for w in words if w in BOILERPLATE_SIGNALS)
    return (signal_count / len(words)) > 0.25


def _classify(
    item_a: ContextItem,
    item_b: ContextItem,
    similarity: float,
) -> RedundancyClassification:
    """
    Classify the type of redundancy between two items.

    Rules:
    1. Adjacent chunks from same source → EXPECTED_OVERLAP
    2. Both are boilerplate → BOILERPLATE
    3. Everything else with high similarity → REDUNDANT_CONTEXT
    """
    if _is_adjacent_source(item_a, item_b):
        return RedundancyClassification.EXPECTED_OVERLAP

    if _is_boilerplate(item_a) and _is_boilerplate(item_b):
        return RedundancyClassification.BOILERPLATE

    return RedundancyClassification.REDUNDANT_CONTEXT


def analyze_redundancy(bundle: ContextBundle) -> tuple[list[RedundancyFinding], int]:
    """
    Detect redundant pairs in a ContextBundle and calculate global final waste.

    Uses a hybrid approach:
    1. Fast path chunk-to-chunk exact/Jaccard similarity for human-readable findings.
    2. Global Multi-Scale N-grams (8, 12, 16) for authoritative penalty math.

    Returns:
        tuple of (list of RedundancyFinding, final_wasted_tokens integer)
    """
    findings: list[RedundancyFinding] = []
    items = bundle.items
    
    # --- 1. Raw Signal Extraction (Multi-Scale N-grams) ---
    SCALES = [8, 12, 16]
    
    item_tokens = []
    total_token_count = 0
    for item in items:
        tokens = _get_ordered_tokens(item.content)
        item_tokens.append((item, tokens, total_token_count))
        total_token_count += len(tokens)
    
    # Fast-path: if content is very large and all items are unique, skip N-gram scan
    MAX_NGRAM_TOKENS = 10000  # Only run N-gram scan below this token count
    
    skip_ngram = False
    if total_token_count > MAX_NGRAM_TOKENS:
        # Check if there are any content hash collisions (potential duplicates)
        from hashlib import md5 as _md5
        content_hashes = set()
        has_collision = False
        for item in items:
            h = _md5(item.content.strip().lower().encode()).hexdigest()
            if h in content_hashes:
                has_collision = True
                break
            content_hashes.add(h)
        
        if not has_collision:
            # All items are unique — no token-level redundancy possible
            skip_ngram = True
    
    scale_waste_counts = {}
    
    if not skip_ngram:
        for N in SCALES:
            token_redundancy_score = [0] * total_token_count
            seen_ngrams = {}
            
            for item, tokens, offset in item_tokens:
                if len(tokens) < N:
                    continue
                for i in range(len(tokens) - N + 1):
                    ngram = tuple(tokens[i:i+N])
                    seen_ngrams.setdefault(ngram, []).append((offset + i, item))
                    
            for ngram, occurrences in seen_ngrams.items():
                if len(occurrences) > 1:
                    occ_items = [occ[1] for occ in occurrences]
                    
                    # Respect Boilerplate and Expected Overlap
                    if all(_is_boilerplate(it) for it in occ_items):
                        continue
                        
                    is_waste = False
                    for i in range(len(occ_items)):
                        for j in range(i+1, len(occ_items)):
                            if occ_items[i].id != occ_items[j].id and not _is_adjacent_source(occ_items[i], occ_items[j]):
                                is_waste = True
                                break
                        if is_waste: break
                            
                    # Self-duplication is waste unless boilerplate
                    if not is_waste and all(occ_items[0].id == it.id for it in occ_items):
                        is_waste = True
                        
                    if is_waste:
                        for start_idx, _ in occurrences:
                            for j in range(start_idx, start_idx + N):
                                token_redundancy_score[j] += 1
                                
            scale_waste_counts[N] = sum(token_redundancy_score)
        
    # --- 2. Structural Aggregation (Weighted Summation & Compression) ---
    weighted_sum = (
        0.4 * scale_waste_counts.get(8, 0) +
        0.35 * scale_waste_counts.get(12, 0) +
        0.25 * scale_waste_counts.get(16, 0)
    )
    import math
    final_wasted_tokens = int(math.sqrt(weighted_sum)) if weighted_sum > 0 else 0

    # --- 3. Generate Human-Readable Findings ---
    # For large bundles, limit pairwise comparisons using content-hash bucketing.
    # Exact duplicates are grouped first (O(n)), then cross-group Jaccard is
    # limited to a sample to keep total work bounded.

    MAX_PAIRWISE_ITEMS = 50  # Only do full O(n²) when n ≤ 50

    if len(items) <= MAX_PAIRWISE_ITEMS:
        pairs_to_check = [(i, j) for i in range(len(items)) for j in range(i + 1, len(items))]
    else:
        # Hash-bucket fast path: group items by stripped content hash
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

        # Add a bounded sample of cross-bucket pairs for fuzzy detection
        all_indices = list(range(len(items)))
        bucket_keys = list(buckets.keys())
        cross_pairs_added = 0
        max_cross_pairs = MAX_PAIRWISE_ITEMS * 10  # cap at ~500
        for bi in range(len(bucket_keys)):
            if cross_pairs_added >= max_cross_pairs:
                break
            for bj in range(bi + 1, len(bucket_keys)):
                if cross_pairs_added >= max_cross_pairs:
                    break
                # Compare first item from each bucket
                pairs_to_check.append((buckets[bucket_keys[bi]][0], buckets[bucket_keys[bj]][0]))
                cross_pairs_added += 1

    for i_idx, j_idx in pairs_to_check:
        item_a = items[i_idx]
        item_b = items[j_idx]

        # Skip empty content
        if not item_a.content.strip() or not item_b.content.strip():
            continue

        # Fast path: exact match
        if item_a.content.strip() == item_b.content.strip():
            similarity = EXACT_MATCH_THRESHOLD
        else:
            words_a = _tokenize_words(item_a.content)
            words_b = _tokenize_words(item_b.content)
            similarity = _jaccard_similarity(words_a, words_b)

        # Only report if above moderate threshold
        if similarity < MODERATE_SIMILARITY_THRESHOLD:
            continue

        classification = _classify(item_a, item_b, similarity)

        # Estimate waste: the smaller item's tokens are "wasted" if redundant
        waste = min(item_a.token_count, item_b.token_count)
        if classification == RedundancyClassification.EXPECTED_OVERLAP:
            # Expected overlap is not full waste — discount by 80%
            waste = int(waste * 0.2)

        # Build human-readable detail
        detail = _build_detail(item_a, item_b, similarity, classification)

        findings.append(RedundancyFinding(
            item_a_id=item_a.id,
            item_b_id=item_b.id,
            similarity_score=similarity,
            classification=classification,
            estimated_waste_tokens=waste,
            detail=detail,
        ))

    # Sort strictly for CI determinism: waste desc, similarity desc, id_a, id_b
    findings.sort(
        key=lambda f: (-f.estimated_waste_tokens, -f.similarity_score, f.item_a_id, f.item_b_id)
    )
    return findings, final_wasted_tokens


def _build_detail(
    item_a: ContextItem,
    item_b: ContextItem,
    similarity: float,
    classification: RedundancyClassification,
) -> str:
    """Build a human-readable explanation of the finding."""
    sim_pct = f"{similarity * 100:.0f}%"

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
