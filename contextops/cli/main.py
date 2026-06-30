"""
ContextOps CLI.

Commands:
  contextops inspect <file>    — Analyze a context file and display results
  contextops check <file>      — CI mode: fail if score below threshold
  contextops demo              — Run analysis on a built-in demo context

The CLI is a view layer. JSON is the source of truth.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from contextops.api.inspect import inspect_context
from contextops.api.stability import run_stability_report
from contextops.api.diff import diff_contexts
from contextops.cli.renderer import render, render_stability, render_diff
from contextops.core.config import ContextOpsConfig


def _build_config(
    config_file: str | None,
    retrieval_max_ratio: float | None,
    system_max_ratio: float | None,
    memory_max_ratio: float | None,
    tool_max_ratio: float | None,
    roast: bool = False,
) -> ContextOpsConfig:
    if config_file:
        config = ContextOpsConfig.from_file(config_file)
    else:
        config = ContextOpsConfig.default()

    # CLI overrides
    has_override = False
    if retrieval_max_ratio is not None:
        config.retrieval_max_ratio = retrieval_max_ratio
        has_override = True
    if system_max_ratio is not None:
        config.system_max_ratio = system_max_ratio
        has_override = True
    if memory_max_ratio is not None:
        config.memory_max_ratio = memory_max_ratio
        has_override = True
    if tool_max_ratio is not None:
        config.tool_max_ratio = tool_max_ratio
        has_override = True

    if has_override:
        config.mode = "custom"

    # CLI --roast flag always wins over config file
    if roast:
        config.roast_enabled = True

    return config


@click.group()
@click.version_option(version="0.1.0", prog_name="contextops")
def cli() -> None:
    """ContextOps — Context observability for LLM applications."""
    pass


@cli.command()
@click.argument("file", type=click.Path(exists=True), required=False)
@click.option("--json-output", is_flag=True, help="Output raw JSON instead of pretty format")
@click.option("--model", default="gpt-4o", help="Model for token encoding (default: gpt-4o)")
@click.option("--config", help="Path to JSON config file")
@click.option("--profile", type=click.Choice(["general", "rag", "agent", "chatbot", "toolchain"]), help="Archetype profile to use for scoring")
@click.option("--retrieval-max-ratio", type=float, help="Override retrieval max ratio")
@click.option("--system-max-ratio", type=float, help="Override system max ratio")
@click.option("--memory-max-ratio", type=float, help="Override memory max ratio")
@click.option("--tool-max-ratio", type=float, help="Override tool max ratio")
@click.option("--explain", is_flag=True, help="Show detailed reasoning for the penalties (Top Score Drivers)")
@click.option("--roast", is_flag=True, default=False, help="Enable roast mode: score-band commentary alongside analysis")
def inspect(
    file: str | None,
    json_output: bool,
    model: str,
    config: str | None,
    profile: str | None,
    retrieval_max_ratio: float | None,
    system_max_ratio: float | None,
    memory_max_ratio: float | None,
    tool_max_ratio: float | None,
    explain: bool,
    roast: bool,
) -> None:
    """Analyze a context file and display results.

    FILE should be a JSON file containing LLM context in any supported format:
    - OpenAI message list
    - Structured dict with system/messages/chunks/memory/tools keys
    """
    cfg = _build_config(
        config, retrieval_max_ratio, system_max_ratio, memory_max_ratio, tool_max_ratio, roast
    )

    raw_input: dict | list
    if file is None:
        # No file provided — run demo
        raw_input = _get_demo_context()
        click.echo(click.style("\n  [i]  No file provided - running demo context\n", fg="cyan"))
    else:
        raw_input = _load_file(file)

    result = inspect_context(raw_input, model=model, config=cfg, cli_profile=profile)
    output = render(result, use_json=json_output, explain=explain)
    _safe_echo(output)


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--min-score", type=int, required=True, help="Minimum passing score (0-100)")
@click.option("--model", default="gpt-4o", help="Model for token encoding (default: gpt-4o)")
@click.option("--json-output", is_flag=True, help="Output JSON instead of pretty format")
@click.option("--config", help="Path to JSON config file")
@click.option("--profile", type=click.Choice(["general", "rag", "agent", "chatbot", "toolchain"]), help="Archetype profile to use for scoring")
@click.option("--retrieval-max-ratio", type=float, help="Override retrieval max ratio")
@click.option("--system-max-ratio", type=float, help="Override system max ratio")
@click.option("--memory-max-ratio", type=float, help="Override memory max ratio")
@click.option("--tool-max-ratio", type=float, help="Override tool max ratio")
@click.option("--explain", is_flag=True, help="Show detailed reasoning for the penalties (Top Score Drivers)")
@click.option("--roast", is_flag=True, default=False, help="Enable roast mode: score-band commentary alongside analysis")
def check(
    file: str,
    min_score: int,
    model: str,
    json_output: bool,
    config: str | None,
    profile: str | None,
    retrieval_max_ratio: float | None,
    system_max_ratio: float | None,
    memory_max_ratio: float | None,
    tool_max_ratio: float | None,
    explain: bool,
    roast: bool,
) -> None:
    """CI mode: fail if context score is below threshold.

    Returns exit code 0 if score >= min_score, exit code 1 otherwise.
    Designed for CI/CD pipelines.

    Example:
        contextops check context.json --min-score 70
    """
    cfg = _build_config(
        config, retrieval_max_ratio, system_max_ratio, memory_max_ratio, tool_max_ratio, roast
    )

    raw_input = _load_file(file)
    result = inspect_context(raw_input, model=model, config=cfg, cli_profile=profile)

    if json_output:
        output = render(result, use_json=True, explain=explain)
    else:
        output = render(result, use_json=False, explain=explain)

    click.echo(output)

    if result.score >= min_score:
        click.echo(click.style(
            f"\n  PASS: score {result.score} >= {min_score}\n",
            fg="green", bold=True
        ))
        sys.exit(0)
    else:
        click.echo(click.style(
            f"\n  FAIL: score {result.score} < {min_score}\n",
            fg="red", bold=True
        ))
        sys.exit(1)


@cli.command()
@click.option("--json-output", is_flag=True, help="Output raw JSON")
@click.option("--roast", is_flag=True, default=True, help="Enable roast mode (default: on for demo)")
def demo(json_output: bool, roast: bool) -> None:
    """Run analysis on a built-in demo context that shows the wow moment."""
    raw_input = _get_demo_context()
    cfg = ContextOpsConfig.default()
    cfg.roast_enabled = roast
    result = inspect_context(raw_input, config=cfg)
    output = render(result, use_json=json_output, explain=False)
    _safe_echo(output)


@cli.command()
@click.argument("file", type=click.Path(exists=True), required=False)
def stability(file: str | None) -> None:
    """Run a deterministic stability report on the context scoring engine."""
    raw_input: dict | list
    if file is None:
        raw_input = _get_demo_context()
        click.echo(click.style("\n  [i]  No file provided - running demo context\n", fg="cyan"))
    else:
        raw_input = _load_file(file)

    report = run_stability_report(raw_input)
    output = render_stability(report)
    _safe_echo(output)


@cli.command()
@click.argument("file_a", type=click.Path(exists=True))
@click.argument("file_b", type=click.Path(exists=True))
def diff(file_a: str, file_b: str) -> None:
    """Compare two context analysis outputs to visualize changes."""
    raw_a = _load_file(file_a)
    raw_b = _load_file(file_b)
    
    result = diff_contexts(raw_a, raw_b)
    output = render_diff(result)
    _safe_echo(output)


# ── Helpers ─────────────────────────────────────────────────────────────


def _load_file(filepath: str) -> dict | list:
    """Load a JSON context file."""
    path = Path(filepath)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        click.echo(click.style(f"Error: Invalid JSON in {filepath}: {e}", fg="red"), err=True)
        sys.exit(2)
    except Exception as e:
        click.echo(click.style(f"Error reading {filepath}: {e}", fg="red"), err=True)
        sys.exit(2)
    return data


def _safe_echo(text: str) -> None:
    """Print text handling Windows console encoding issues."""
    try:
        click.echo(text)
    except UnicodeEncodeError:
        # Fallback: replace unencodable chars
        encoded = text.encode(sys.stdout.encoding or "utf-8", errors="replace")
        sys.stdout.buffer.write(encoded)
        sys.stdout.buffer.write(b"\n")
        sys.stdout.buffer.flush()


def _get_demo_context() -> dict:
    """
    Built-in demo context designed to trigger the "wow moment".

    This simulates a real-world RAG pipeline with:
    - A bloated system prompt
    - Redundant retrieval chunks (near-duplicates)
    - Memory that overlaps with retrieval
    - Low source diversity (all chunks from the same doc)
    """
    return {
        "system": (
            "You are a helpful AI assistant. You must always be polite and professional. "
            "You must always follow the user's instructions carefully. "
            "You must never generate harmful content. "
            "You must always provide accurate and helpful responses. "
            "Please ensure you follow all guidelines at all times. "
            "Remember to always be helpful and follow instructions. "
            "Important: always respond in a structured format. "
            "Note: ensure your responses are accurate and well-formatted. "
            "Guidelines: follow all rules and provide quality answers. "
            "Rules: be professional, be accurate, be helpful at all times."
        ),
        "messages": [
            {"role": "user", "content": "What is the refund policy for premium subscriptions?"},
        ],
        "chunks": [
            {
                "content": (
                    "Refund Policy: Premium subscriptions can be refunded within 30 days "
                    "of purchase. To request a refund, contact support@example.com with "
                    "your order number. Refunds are processed within 5-7 business days."
                ),
                "source": "support_docs/refund_policy.md",
            },
            {
                "content": (
                    "Premium Subscription Refund: You may request a refund for premium "
                    "subscriptions within 30 days of purchase. Send your order number to "
                    "support@example.com. Processing takes 5-7 business days."
                ),
                "source": "support_docs/refund_policy.md",
            },
            {
                "content": (
                    "Our refund policy allows premium subscription holders to get a full "
                    "refund within 30 days. Contact support@example.com with your order "
                    "number for processing. Expect 5-7 business days for completion."
                ),
                "source": "support_docs/refund_policy.md",
            },
            {
                "content": (
                    "How to get a refund: If you have a premium subscription, you can "
                    "request a refund within 30 days of purchase. Email support@example.com "
                    "with your order number. Refunds take 5-7 business days."
                ),
                "source": "support_docs/refund_policy.md",
            },
            {
                "content": (
                    "Pricing: Premium subscriptions cost $29.99/month or $299/year. "
                    "Enterprise plans start at $99/month. Contact sales for custom pricing."
                ),
                "source": "support_docs/pricing.md",
            },
        ],
        "memory": [
            (
                "User previously asked about premium subscription pricing on 2024-01-15. "
                "User was interested in the annual plan. Refund policy was mentioned briefly."
            ),
            (
                "User asked about the refund policy for premium subscriptions. "
                "Support documents indicate 30-day refund window. Contact support@example.com."
            ),
        ],
    }


# ── Telemetry CLI ──────────────────────────────────────────────────────────

@cli.group()
def telemetry() -> None:
    """Manage ContextOps local telemetry."""
    pass

@telemetry.command()
def status() -> None:
    """Show telemetry status and file location."""
    from contextops.core.telemetry import _IS_ENABLED, get_telemetry_path
    
    status_str = click.style("ENABLED", fg="green") if _IS_ENABLED else click.style("DISABLED", fg="yellow")
    click.echo(f"Telemetry is {status_str}")
    
    path = get_telemetry_path()
    click.echo(f"File path: {path}")
    
    if path.exists():
        size = path.stat().st_size
        click.echo(f"File size: {size} bytes")
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = sum(1 for _ in f)
            click.echo(f"Total events: {lines}")
        except Exception:
            pass
    else:
        click.echo("File does not exist yet.")

@telemetry.command()
@click.option("--limit", type=int, default=10, help="Number of recent events to show")
def log(limit: int) -> None:
    """Show recent telemetry events."""
    from contextops.core.telemetry import read_events
    events = read_events(limit)
    if not events:
        click.echo("No telemetry events found.")
        return
        
    for e in events:
        score = e.get("score", "?")
        delta = e.get("score_delta")
        delta_str = f"({delta:+d})" if delta is not None else ""
        arch = e.get("archetype", "general")
        ts = e.get("ts", "")
        findings = ", ".join(e.get("top_findings", []))
        
        click.echo(f"[{ts}] Score: {score} {delta_str} | Archetype: {arch} | Findings: {findings}")

@telemetry.command()
@click.option("--days", type=int, default=30, help="Number of days to analyze for trends")
def trends(days: int) -> None:
    """Show operational trends over a given time period."""
    from contextops.core.telemetry import get_trends
    
    t = get_trends(days)
    if not t or t.get("event_count", 0) == 0:
        click.echo("Not enough telemetry data to calculate trends.")
        return
        
    click.echo(f"Average score ({days} days): {t.get('avg_score', 0):.0f}")
    click.echo(f"Average wasted tokens: {t.get('avg_wasted', 0):.0f}")
    
    common = t.get("most_common_failure")
    if common:
        click.echo(f"\nMost common failure:\n{common}")
        
    trend_val = t.get("trend", 0)
    if trend_val > 0:
        trend_str = click.style(f"UP {trend_val:.0f}%", fg="green")
    elif trend_val < 0:
        trend_str = click.style(f"DOWN {abs(trend_val):.0f}%", fg="red")
    else:
        trend_str = "FLAT"
        
    click.echo(f"\nQuality trend:\n{trend_str}")


# ── Badge CLI ──────────────────────────────────────────────────────────────

@cli.command()
@click.option("--score", type=int, help="The score to generate a badge for (0-100). If omitted, uses your last telemetry score.")
def badge(score: int | None) -> None:
    """Generate a GitHub shields.io markdown badge for a ContextOps score."""
    if score is None:
        from contextops.core.telemetry import read_last_score
        last_score = read_last_score()
        if last_score is None:
            click.echo("Error: No previous telemetry score found. Please specify --score.", err=True)
            return
        score = last_score
        
    if score >= 90:
        color = "green"
    elif score >= 70:
        color = "yellow"
    else:
        color = "red"
        
    badge_md = f"[![ContextOps](https://img.shields.io/badge/ContextOps-{score}-{color})](https://github.com/contextops)"
    click.echo(badge_md)

if __name__ == "__main__":
    cli()
