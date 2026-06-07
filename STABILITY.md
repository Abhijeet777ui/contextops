# ContextOps Stability Contract

> **Version:** 0.2.0  
> **Status:** Active  
> **Last Updated:** 2026-06-07  
> **Applies To:** All `0.2.x` releases

This document defines the behavioral guarantees that ContextOps makes to its users. These guarantees exist so that engineering teams can safely integrate ContextOps into CI/CD pipelines, dashboards, and automated workflows without fear of silent breakage.

If a guarantee listed in this document is violated by a release, that release is considered **defective** and will be patched or reverted.

---

## 1. Core Identity

ContextOps is a **deterministic, embedding-free context linter** for LLM input payloads.

It measures structural bloat — not semantic quality, not reasoning correctness, not output truth. It is designed to behave like ESLint for prompt engineering: fast, predictable, and CI-safe.

**ContextOps will NEVER:**

- Use LLM inference or embeddings in its analysis pipeline
- Introduce randomness or probabilistic scoring
- Require network access to produce a result
- Evaluate the semantic meaning of context content

---

## 2. Versioning Policy

ContextOps follows [Semantic Versioning 2.0.0](https://semver.org/) with the following strict interpretations for infrastructure use:

### Major Version (`X.0.0`)

A major version bump is required for:

- **Scoring formula changes** — Any modification to the mathematical functions that compute `redundancy_penalty`, `density_penalty`, `structure_penalty`, or `concentration_penalty`
- **Threshold changes** — Any change to default Jaccard thresholds, structure ratio limits, or N-gram scale weights
- **Output schema breaking changes** — Renaming, removing, or changing the type of any field in the JSON output
- **Behavioral changes** — Any change that would cause the same input to produce a different score

### Minor Version (`0.X.0`)

A minor version bump is used for:

- New CLI commands or flags (additive only)
- New optional fields in the JSON output (existing fields remain unchanged)
- New analyzers that contribute to scoring (behind feature flags until next major)
- Documentation and metadata additions

### Patch Version (`0.0.X`)

A patch version bump is used for:

- Bug fixes that restore behavior to match this contract
- Performance improvements that do not change output
- Internal refactoring with no external effect
- Documentation corrections

### The Golden Rule

> **If the same `ContextBundle` input produces a different `score` after an update, that update MUST be a major version bump.**

---

## 3. Determinism Guarantee

ContextOps is **fully deterministic**. This is the foundational guarantee upon which all CI integration depends.

### What This Means

Given the same input payload:

- The **score** will be identical across runs
- The **penalty breakdown** will be identical across runs
- The **recommendations** will be identical across runs (same issues, same order)
- The **findings** will be identical across runs (same pairs, same classifications)

### What This Requires

- No random number generation in any analysis path
- No dependency on system time, locale, or environment variables for scoring
- No floating-point non-determinism (all intermediate results are rounded to fixed precision before comparison)
- No dependency on dictionary ordering (all outputs are explicitly sorted)

### Verification

Determinism is enforced by the chaos test suite (`tests/test_chaos.py`), which runs every payload twice and asserts exact score equality. This test runs on every commit.

---

## 4. Output Schema Contract

The JSON output of `contextops inspect --json-output` conforms to the following schema. **No field in this schema will be renamed, removed, or have its type changed within a major version.**

```json
{
  "score": "<integer, 0–100>",
  "mode": "<string, 'strict' | 'lenient'>",
  "config_version": "<string, semver>",
  "score_breakdown": {
    "redundancy_penalty": "<float, >= 0>",
    "density_penalty": "<float, >= 0>",
    "structure_penalty": "<float, >= 0>",
    "concentration_penalty": "<float, >= 0>",
    "total_penalty": "<float, >= 0>"
  },
  "token_breakdown": {
    "total_tokens": "<integer, >= 0>",
    "by_type": "<dict[string, integer]>",
    "estimated_cost_usd": "<float, >= 0>",
    "wasted_tokens": "<integer, >= 0>"
  },
  "findings": {
    "redundancy": "<array of RedundancyFinding>",
    "structure": "<array of StructureFinding>"
  },
  "recommendations": "<array of Recommendation>",
  "metadata": {
    "item_count": "<integer>",
    "model": "<string>",
    "version": "<string, semver>"
  }
}
```

### Schema Evolution Rules

| Action | Allowed in Minor? | Allowed in Patch? |
|---|---|---|
| Add new top-level field | ✅ Yes | ❌ No |
| Add new nested field | ✅ Yes | ❌ No |
| Remove any field | ❌ No (major only) | ❌ No |
| Rename any field | ❌ No (major only) | ❌ No |
| Change field type | ❌ No (major only) | ❌ No |
| Change enum values | ❌ No (major only) | ❌ No |

---

## 5. Scoring Contract

### 5.1 Score Formula

The context score is computed as:

```
Context Score = max(0, min(100, round(100 - total_penalty)))
```

Where:

```
total_penalty = redundancy_penalty + density_penalty + structure_penalty + concentration_penalty
```

### 5.2 Penalty Ranges

| Penalty | Min | Max | What It Measures |
|---|---|---|---|
| `redundancy_penalty` | 0.0 | 30.0 | Lexical duplication across context items |
| `density_penalty` | 0.0 | 30.0 | Token waste from structural bloat |
| `structure_penalty` | 0.0 | 20.0 | Distribution imbalance across context types |
| `concentration_penalty` | 0.0 | 20.0 | Source dominance or highly imbalanced chunk distribution |

### 5.3 Mathematical Invariants

These invariants hold for **every** input, without exception:

1. `0 ≤ score ≤ 100`
2. `redundancy_penalty ≥ 0`
3. `density_penalty ≥ 0`
4. `structure_penalty ≥ 0`
5. `concentration_penalty ≥ 0`
6. `total_penalty == redundancy_penalty + density_penalty + structure_penalty + concentration_penalty` (within ±0.01 floating-point tolerance)
7. `score == max(0, min(100, round(100 - total_penalty)))` (within ±1 point tolerance)
8. No `NaN` or `Infinity` values in any numeric field
9. `wasted_tokens ≥ 0`
10. `total_tokens ≥ 0`

These invariants are enforced by automated tests that run against adversarial inputs including empty payloads, single-token contexts, 45,000+ token RAG dumps, and deeply nested tool loops.

### 5.4 Redundancy Detection Scope

ContextOps V0.1 detects redundancy using **lexical methods only**:

- N-gram overlap at scales 8, 12, and 16
- Jaccard similarity on word sets
- Exact string matching

It does **NOT** detect:

- Semantic similarity (same meaning, different words)
- Intent duplication (same goal, different phrasing)
- Cross-lingual repetition

### 5.5 Redundancy Classifications

The following classifications are used in V0.1 and their definitions will not change within the major version:

| Classification | Meaning | Penalized? |
|---|---|---|
| `REDUNDANT_CONTEXT` | High similarity between independent sources | ✅ Yes |
| `EXPECTED_OVERLAP` | Adjacent chunks from the same source | ⚠️ Discounted (20% of full penalty) |
| `BOILERPLATE` | Both items are boilerplate instructions | ❌ No |

No new classifications will be added in patch releases.

---

## 6. CLI Contract

### 6.1 Stable Commands

The following CLI commands are stable and will not be removed or have their behavior changed within a major version:

| Command | Purpose |
|---|---|
| `contextops inspect <file>` | Analyze a context payload and display results |
| `contextops check <file> --min-score N` | CI gate — exit code 1 if score < N |
| `contextops diff <file_a> <file_b>` | Compare two analysis snapshots |
| `contextops stability <dir>` | Run stability report across benchmark suite |

### 6.2 Stable Flags

| Flag | Available On | Purpose |
|---|---|---|
| `--json-output` | `inspect`, `check` | Output results as JSON |
| `--min-score N` | `check` | Set minimum passing score |
| `--model <name>` | `inspect`, `check` | Set target model for cost estimation |

### 6.3 Exit Codes

| Code | Meaning |
|---|---|
| `0` | Success (score meets threshold) |
| `1` | Failure (score below threshold, or analysis error) |

Exit code semantics will not change within a major version.

---

## 7. Performance Guarantees

ContextOps commits to the following performance bounds. These are tested in CI and violations are treated as bugs.

| Payload Size | Max Execution Time |
|---|---|
| ≤ 5,000 tokens | < 2 seconds |
| ≤ 20,000 tokens | < 5 seconds |
| ≤ 50,000 tokens | < 10 seconds |

### Memory

- Memory usage scales linearly with input size
- No unbounded dictionary or list growth without pruning
- The engine does not cache results between invocations

### Dependencies

ContextOps has **zero runtime dependencies** beyond the Python standard library and `click` for CLI. It will never require:

- GPU access
- Network connectivity
- External API keys
- Database connections

---

## 8. Input Contract

ContextOps accepts the following input formats. Support for these formats will not be removed within a major version:

| Format | Description |
|---|---|
| OpenAI message list | `[{"role": "user", "content": "..."}]` |
| Structured dict | `{"system": "...", "messages": [...], "chunks": [...]}` |
| Plain string | Treated as a system prompt |

### Input Resilience

The engine handles malformed inputs gracefully:

- Empty content strings → treated as 0 tokens, no crash
- Missing optional fields → defaults applied silently
- Whitespace-only content → treated as empty
- Negative token counts → clamped to 0

---

## 9. What Is NOT Guaranteed

The following aspects may change between minor versions without notice:

- **Human-readable text** in recommendations (the `fix` and `issue` strings)
- **Ordering of recommendations** (only guaranteed to be deterministic, not in any specific order)
- **Internal data structures** and private API surfaces
- **Exact wording** of CLI output in non-JSON mode
- **Estimated cost calculations** (`estimated_cost_usd`) — model pricing changes
- **Number of findings** reported (the engine may add more granular findings)

---

## 10. Breaking This Contract

If you discover behavior that violates any guarantee in this document, please file an issue with:

1. The exact input payload (as a JSON file)
2. The expected behavior (referencing this document)
3. The actual behavior (including full `--json-output`)
4. Your ContextOps version (`contextops --version`)

We treat stability contract violations as **P0 bugs** — they will be patched within the current minor version.

---

## 11. Future Roadmap (Deferred)

The following capabilities are planned for future **major** versions and will NOT appear in `0.1.x`:

- Intent classification (`informational`, `safety_constraint`, `grounding`)
- Semantic redundancy detection (embedding-based)
- Context lineage tracking and attribution graphs
- Safety-aware redundancy exemptions
- Configurable penalty curves

These features will ship behind explicit opt-in flags when introduced, and will not affect default scoring behavior until the next major version.

---

*This document is versioned alongside the codebase. Any changes to guarantees require a PR review and version bump.*
