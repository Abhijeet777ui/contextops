# Changelog

All notable changes to ContextOps will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/) as defined in [STABILITY.md](STABILITY.md).

## [0.2.0] — 2026-06-07

### Added

- **Signal Orthogonality Contract**: Added machine-enforced tests to guarantee strict isolation between penalty dimensions.
- **Concentration Signal (P_con)**: Replaced naive diversity logic with a rigorous 2-axis decomposition:
  - *Source Dominance ($P_{dom}$)*: Penalizes over-reliance on a single document.
  - *Entropy Imbalance ($P_{ent}$)*: Penalizes uneven distributions across multiple documents using Shannon Entropy.
  - Guards against penalizing "Gold Answer RAG" (single-chunk perfect retrieval).
- **Density Signal Upgrade**: Implemented true structural token waste measurement:
  - *Format Overhead ($d_S$)*: JSON/Markdown syntax vs payload character ratio.
  - *Whitespace Waste ($d_W$)*: Unnecessary indentation and spacing.
  - *Entropy Compression*: Normalized Shannon entropy to detect highly repetitive patterns without embeddings.

### Changed

- **API/Schema Breaking Change**: Renamed `diversity_penalty` to `concentration_penalty` in the JSON API output (`ScoreBreakdown`).
- `config_version` bumped to `"2.0"`.

## [0.1.0] — 2026-06-04

### Added

- **Core Engine** — Deterministic context scoring engine with 4-penalty model:
  - Redundancy Penalty (0–30): Multi-scale N-gram overlap detection (scales 8, 12, 16)
  - Density Penalty (0–30): Exponential waste curve `30 * (1 - exp(-k * waste))`
  - Structure Penalty (0–20): Threshold-based distribution imbalance detection
  - Diversity Penalty (0–20): Source concentration measurement
- **Redundancy Analyzer** — Hybrid Jaccard + N-gram scanner with 3 classifications:
  - `REDUNDANT_CONTEXT` — independent sources with high similarity (penalized)
  - `EXPECTED_OVERLAP` — adjacent chunks from same source (discounted 80%)
  - `BOILERPLATE` — repeated instructions (not penalized)
- **Structure Analyzer** — Detects retrieval dominance, system prompt bloat, memory explosion, and tool output sprawl
- **CLI Commands:**
  - `contextops inspect <file>` — Full analysis with rich terminal output
  - `contextops check <file> --min-score N` — CI gate with exit codes
  - `contextops demo` — Built-in demo context for first-run experience
  - `contextops stability <file>` — Deterministic stability report
  - `contextops diff <file_a> <file_b>` — Regression comparison between snapshots
- **Python API** — `inspect_context()`, `diff_contexts()`, `run_stability_report()`
- **Normalizer** — Accepts OpenAI message lists, structured dicts, and plain strings
- **Configurable thresholds** — Via JSON config files or CLI flags
- **JSON output mode** — `--json-output` for CI/CD pipeline integration
- **Chaos test suite** — 5 adversarial benchmarks validating global invariants
- **STABILITY.md** — Formal stability contract for CI adoption

### Performance

- Sub-second analysis for payloads up to 50,000 tokens
- Hash-bucket optimization for pairwise comparisons on large bundles (>50 items)
- N-gram fast-path skip when all items have unique content hashes
