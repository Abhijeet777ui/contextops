# ContextOps

**The deterministic context linter for LLM applications.**

[![PyPI version](https://img.shields.io/pypi/v/contextops.svg)](https://pypi.org/project/contextops/) 
[![Python](https://img.shields.io/pypi/pyversions/contextops.svg)](https://pypi.org/project/contextops/)
[![License: Sustainable Use](https://img.shields.io/badge/License-Sustainable_Use-blue.svg)](LICENSE)
[![CI](https://img.shields.io/badge/CI-stable-brightgreen.svg)](STABILITY.md)

ContextOps analyzes the context data you send to your LLM and tells you what is broken. It detects redundant information, wasted tokens, and structural imbalances, providing a deterministic 0-100 score and actionable fixes.

Think of it as a spell-checker for your AI prompts.

---

## Why ContextOps?

Most AI applications blindly stuff data into the prompt window. This leads to:

- Wasted spend: paying for redundant tokens that do not improve the AI's output.
- Silent regressions: a small change in search logic floods the prompt with duplicates.
- Structural drift: retrieved documents slowly take over the entire prompt, hiding the actual instructions.
- No visibility: developers have no way to measure context quality in automated testing.

ContextOps gives you this visibility. It runs in your testing pipeline, scores every context payload, and fails the build if quality drops.

---

## What is New in v0.3.3 (Production Ready)

ContextOps v0.3.3 introduces major architectural upgrades, graduating the engine from a basic linter to a robust CI signal system:

- **Archetype-Aware Scoring**: Optimize evaluation based on payload intent (`rag`, `agent`, `chatbot`, `toolchain`). ContextOps intelligently suppresses irrelevant structural findings (like tool sprawl for agents) without corrupting the universal 0-100 global score.
- **Opt-In Local Telemetry & Trends**: Track your context quality over time locally. Use `contextops telemetry trends` to instantly see your 30-day average score, wasted tokens, and most common failures.
- **GitHub Badges**: Show off your context quality! Run `contextops badge` to instantly generate a shields.io markdown badge for your project's README.
- **Deterministic Fast-Pass Redundancy**: 100% exact-match redundancy detection via rolling `xxhash` signatures. Filters out semantic paraphrases and only flags guaranteed token waste.
- **Log-Scale Density Scoring**: Advanced density calculations using logarithmic formulas to cleanly score whitespace, JSON structure, and markdown format overhead without false-positive penalty explosions.
- **Dual-Threshold Structure Confidence**: Structure penalties dynamically scale with the absolute token size of the payload. Prevents tiny, healthy contexts (< 200 tokens) from failing CI gates due to ratio imbalances.
- **The Roast Engine**: Enable the `--roast` flag on the CLI to get brutally honest, score-band commentary on your context quality.
- **ContextBench**: We have introduced ContextBench, a suite of 1,500 real-world enterprise context payloads to test and calibrate scoring algorithms.

---

## The ContextBench Leaderboard

The **[ContextBench Leaderboard](https://Abhijeet777ui.github.io/contextops/)** is the definitive benchmark for evaluating LLM context optimization architectures. It tests optimizers across 1,500 enterprise RAG payloads and scores them using a weighted formula: **50% Quality + 35% Compression + 15% Latency**.

To keep the competition honest, there is a hard **Quality Floor Gate**. If your optimizer destroys the context so badly that the ContextOps score drops below 78, you are instantly disqualified. You cannot win on compression alone.

### How to Submit
Submission is completely automated via GitHub Actions:
1. Write a Python function `def optimize_context(context: dict) -> dict:` in a public GitHub repository.
2. Open a new **Issue** in this repository titled `Submission: [Your Team Name]`.
3. Put the link to your public repository in the issue body.

Our automated harness will clone your code, run it against ContextBench, enforce the quality gates, generate a ContextOps Roast, and automatically push your score to the live leaderboard.

---

## Quick Start

```bash
pip install contextops
```

### See it in action

```bash
# Run the built-in demo to see a live analysis, complete with the Roast Engine
contextops demo
```

### Analyze your own context

```bash
# Full analysis with rich terminal output
contextops inspect context.json

# Add the roast flag for honest commentary
contextops inspect context.json --roast

# Use a specific archetype profile to suppress irrelevant warnings
contextops inspect context.json --profile rag

# CI mode: fail if the score drops below a threshold
contextops check context.json --min-score 70

# Compare two snapshots to find regressions
contextops diff before.json after.json

# View 30-day operational telemetry trends
contextops telemetry trends

# Generate a GitHub badge based on your last telemetry run
contextops badge
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
}, archetype="rag")

print(f"Score: {result.score}/100")
print(f"Wasted tokens: {result.token_breakdown.wasted_tokens}")
for rec in result.recommendations:
    print(f"  -> {rec.fix}")
```

---

## LangChain Integration

ContextOps plugs into any LangChain chain with a single line. We use explicit injection rather than hidden global patches, so you always know exactly where ContextOps is running.

```bash
pip install contextops langchain-core
```

```python
from contextops import ContextOps

# Attach to any chain
chain = chain.with_config({
    "callbacks": [ContextOps.auto()]
})

result = chain.invoke({"question": "What is the refund policy?"})
```

Output is printed automatically before the LLM execution begins. You can also configure it to warn or block execution if the score is too low:

```python
# Block execution entirely if context quality is below 70
ContextOps.auto(mode="block", min_score=70)
```

---

## What It Measures

ContextOps computes a 0-100 Context Score based on four independent penalties:

| Dimension | What It Detects | Max Penalty |
|---|---|---|
| **Redundancy** | Duplicate or near-duplicate chunks of text. | 30 pts |
| **Density** | Wasted tokens from structural bloat and formatting. | 30 pts |
| **Structure** | Imbalanced data distribution (e.g., retrieval taking up 80% of the prompt). | 20 pts |
| **Concentration** | Over-reliance on a single document or source. | 20 pts |

Every penalty maps to a specific finding with token savings and an actionable fix.

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

It also accepts raw OpenAI message lists directly.

---

## Stability Contract

ContextOps ships with a formal Stability Contract (STABILITY.md) that guarantees:

- Scoring determinism: the same input always produces the same score.
- Schema stability: JSON output fields never change within a major version.
- Semantic versioning: any changes to the scoring formula require a major version bump.

This ensures you can trust ContextOps in your production testing pipelines.

---

## License

Sustainable Use License (see LICENSE file for details).
