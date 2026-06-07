"""
Density Analyzer — Phase 2.5 (Metric Orthogonalization).

Computes the structural Density Signal from raw context text.

Signal contract:
    - Reads ONLY from raw ContextBundle item content strings.
    - Does NOT read wasted_tokens, redundancy findings, or any other analyzer output.
    - Is the sole authoritative input for density_penalty in the scoring engine.

Three orthogonal character buckets (exhaustive, non-overlapping):
    payload_chars   = alphanumeric  (actual information)
    syntax_chars    = non-alphanum, non-whitespace (brackets, punctuation, markup)
    whitespace_chars = whitespace   (layout/formatting overhead)
    total_chars = payload + syntax + whitespace  (always sums to 1.0)
"""

from __future__ import annotations

import math
import re
from collections import Counter
from contextops.core.models import ContextBundle, DensitySignal


def normalize_density_input(text: str) -> list[str]:
    """
    Standardize text preprocessing for density metrics to prevent metric drift.

    Rules (frozen — do not change without updating all callers):
        1. Lowercase
        2. Replace non-alphanumeric (including underscore) with spaces
        3. Split on whitespace only

    The regex uses [^a-z0-9\\s] (not \\w) to ensure underscores are treated
    as punctuation, not part of identifiers. This makes snake_case and kebab-case
    consistent (both split into component words).
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text.split()


def _calc_format_overhead(text: str) -> float:
    """
    Format Overhead (FO): Ratio of syntax chars to total chars.

    FO = syntax_chars / total_chars
    where syntax_chars = non-alphanumeric AND non-whitespace characters
    (brackets, punctuation, markup, operators, etc.)

    Range: 0.0 (no syntax overhead) → 1.0 (all syntax, no payload or whitespace).
    Does NOT include whitespace — that is measured separately by WL.
    """
    total_chars = len(text)
    if total_chars == 0:
        return 0.0

    syntax_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return max(0.0, min(1.0, syntax_chars / total_chars))


def _calc_whitespace_waste(text: str) -> float:
    """
    Whitespace Waste (WL): Ratio of whitespace chars to total chars.

    WL = whitespace_chars / total_chars
    where whitespace_chars = space, tab, newline, carriage return, etc.

    Range: 0.0 (no whitespace) → 1.0 (all whitespace).
    Does NOT include syntax chars — that is measured separately by FO.
    """
    total_chars = len(text)
    if total_chars == 0:
        return 0.0

    whitespace_chars = sum(1 for c in text if c.isspace())
    return max(0.0, min(1.0, whitespace_chars / total_chars))


def _calc_entropy_compression(text: str) -> float:
    """
    Entropy Compression (EC): Statistical measure of repetitive boilerplate.

    EC = 1 - normalized_shannon_entropy

    High EC → low entropy → highly repetitive token distribution.
    Low EC  → high entropy → diverse vocabulary (good).

    Normalization: entropy / log2(unique_words) so range is always 0.0–1.0.
    """
    words = normalize_density_input(text)
    if not words:
        return 0.0

    total_words = len(words)
    word_counts = Counter(words)
    unique_words = len(word_counts)

    if unique_words <= 1:
        return 1.0  # single word repeated — maximum compression

    # Shannon entropy
    entropy = 0.0
    for count in word_counts.values():
        p = count / total_words
        entropy -= p * math.log2(p)

    max_entropy = math.log2(unique_words)
    normalized_entropy = entropy / max_entropy

    return max(0.0, min(1.0, 1.0 - normalized_entropy))


def compute_density_signal(bundle: ContextBundle) -> DensitySignal:
    """
    Compute the structural Density Signal from raw context content.

    Signal contract: reads ONLY raw item.content strings.
    Must NOT read token_count, wasted_tokens, or any analyzer output.

    Weights (initial): FO=0.4, WL=0.2, EC=0.4
    These are calibrated so typical clean context scores near 0.1–0.3,
    and heavily bloated context scores near 0.6–0.9.
    """
    if not bundle.items:
        return DensitySignal(0.0, 0.0, 0.0, 0.0)

    total_text = "\n".join(item.content for item in bundle.items)

    if not total_text.strip():
        return DensitySignal(0.0, 0.0, 0.0, 0.0)

    fo = _calc_format_overhead(total_text)
    wl = _calc_whitespace_waste(total_text)
    ec = _calc_entropy_compression(total_text)

    # Weights: w_fo=0.4, w_wl=0.2, w_ec=0.4
    total_signal = (0.4 * fo) + (0.2 * wl) + (0.4 * ec)

    return DensitySignal(
        format_overhead=round(fo, 3),
        whitespace_waste=round(wl, 3),
        entropy_compression=round(ec, 3),
        total_density_signal=round(total_signal, 3),
    )
