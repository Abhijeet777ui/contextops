# ContextOps

**The deterministic context linter for LLM applications.**

[![PyPI version](https://img.shields.io/pypi/v/contextops.svg)](https://pypi.org/project/contextops/) 
[![Python](https://img.shields.io/pypi/pyversions/contextops.svg)](https://pypi.org/project/contextops/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://img.shields.io/badge/CI-stable-brightgreen.svg)](STABILITY.md)

ContextOps analyzes the context fed into your LLM and tells you what's broken — redundant chunks, wasted tokens, structural imbalance — with a **deterministic 0–100 score** and actionable fixes.

Think of it as **ESLint for your LLM prompts**.

---

## Why ContextOps?

Most LLM applications blindly stuff context into the prompt window. This leads to:

- 💸 **Wasted spend** — paying for redundant tokens that don't improve output
- 🔁 **Silent regressions** — a "small RAG change" floods the context with duplicates
- 🏗️ **Structural drift** — retrieval chunks slowly dominate the entire prompt
- 🎯 **No visibility** — teams have no way to measure context quality in CI

ContextOps gives you that visibility. It runs in your CI pipeline, scores every context payload, and fails the build if quality degrades.

---

## Quick Start

```bash
pip install contextops
```

### See it in action

```bash
# Run the built-in demo — instant "wow moment"
contextops demo
```

### Analyze your own context

```bash
# Full analysis with rich terminal output
contextops inspect context.json

# CI mode: fail if score drops below threshold
contextops check context.json --min-score 70

# Compare two snapshots for regressions
contextops diff before.json after.json

# JSON output for dashboards and automation
contextops inspect context.json --json-output
```

### Python API

```python
from contextops.api.inspect import inspect_context

result = inspect_context({
    "system": "You are a helpful assistant.",
    "chunks": [
        {"content": "Refund policy: 30 days...", "source": "docs/refund.md"},
        {"content": "Refund policy: within 30 days...", "source": "docs/refund.md"},
    ],
    "memory": ["User asked about refunds before."],
})

print(f"Score: {result.score}/100")
print(f"Wasted tokens: {result.token_breakdown.wasted_tokens}")
for rec in result.recommendations:
    print(f"  → {rec.fix}")
```

---

## What It Measures

ContextOps computes a **0–100 Context Score** from four independent penalty dimensions:

| Dimension | What It Detects | Max Penalty |
|---|---|---|
| **Redundancy** | Duplicate / near-duplicate chunks (N-gram + Jaccard) | 30 pts |
| **Density** | Wasted tokens from structural bloat | 30 pts |
| **Structure** | Imbalanced type distribution (e.g., retrieval > 70%) | 20 pts |
| **Concentration** | Source dominance or highly imbalanced chunk distribution | 20 pts |

```
Context Score = 100 - (Redundancy + Density + Structure + Concentration)
```

Every penalty maps to a **specific finding** with **token savings** and an **actionable fix**.

---

## CI / CD Integration

### GitHub Actions

```yaml
name: Context Quality Gate

on: [pull_request]

jobs:
  context-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - run: pip install contextops

      - name: Check context quality
        run: contextops check prompts/context.json --min-score 75
```

### Exit Codes

| Code | Meaning |
|---|---|
| `0` | Score meets threshold — build passes |
| `1` | Score below threshold — build fails |

### Regression Detection

```bash
# Save a baseline
contextops inspect prompts/v1.json --json-output > baseline.json

# After changes, compare
contextops diff baseline.json prompts/v2.json
```

---

## Context File Format

ContextOps accepts a JSON file with any combination of these keys:

```json
{
    "system": "Your system prompt here",
    "messages": [
        {"role": "user", "content": "User question"}
    ],
    "chunks": [
        {"content": "Retrieved chunk text", "source": "docs/page.md"}
    ],
    "memory": [
        "Previous conversation context"
    ],
    "tools": [
        {"name": "search_api", "output": "Tool response text"}
    ]
}
```

It also accepts raw OpenAI message lists:

```json
[
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is the refund policy?"}
]
```

---

## CLI Reference

| Command | Purpose |
|---|---|
| `contextops inspect <file>` | Analyze and display results |
| `contextops check <file> --min-score N` | CI gate with exit codes |
| `contextops demo` | Built-in demo context |
| `contextops stability <file>` | Deterministic stability report |
| `contextops diff <file_a> <file_b>` | Compare two snapshots |

### Flags

| Flag | Commands | Purpose |
|---|---|---|
| `--json-output` | inspect, check | Machine-readable JSON output |
| `--min-score N` | check | Minimum passing score (0–100) |
| `--model <name>` | inspect, check | Target model for cost estimation |
| `--explain` | inspect, check | Show detailed penalty reasoning |
| `--config <file>` | inspect, check | Custom threshold config file |

---

## Design Principles

1. **Deterministic** — Same input → same output. Always. No randomness, no embeddings, no LLM calls.
2. **Explainable** — Every penalty maps to a real issue with a token count and a fix.
3. **CI-native** — Designed for pipelines first. Exit codes, JSON output, threshold gating.
4. **Zero network** — Runs entirely offline. No API keys, no external services.

---

## Stability Contract

ContextOps ships with a formal [Stability Contract](STABILITY.md) that guarantees:

- **Scoring determinism** — same input always produces the same score
- **Schema stability** — JSON output fields never change within a major version
- **Performance bounds** — sub-second for payloads up to 50,000 tokens
- **Semantic versioning** — scoring formula changes require a major version bump

This contract exists so teams can trust ContextOps in production CI pipelines.

---

## Development

```bash
# Clone and install in dev mode
git clone https://github.com/Abhijeet777/contextops.git
cd contextops
pip install -e ".[dev]"

# Run tests
pytest

# Run chaos stress tests
pytest tests/test_chaos.py -v
```

---

## License

[MIT](LICENSE)
