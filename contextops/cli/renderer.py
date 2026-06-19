"""
CLI Renderer.

Pretty-prints AnalysisResult to the terminal.
All rendering is derived from JSON (AnalysisResult.to_dict()) — the CLI
is a view layer only. The JSON is the source of truth.
"""

from __future__ import annotations

import json

from contextops.core.models import AnalysisResult
from typing import Any


# ── ANSI color codes ────────────────────────────────────────────────────

class _Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"


C = _Colors


def _score_color(score: int) -> str:
    """Pick a color based on the score range."""
    if score >= 80:
        return C.GREEN
    elif score >= 60:
        return C.YELLOW
    elif score >= 40:
        return C.RED
    else:
        return C.BG_RED + C.WHITE


def _score_label(score: int) -> str:
    """Human-readable label for the score."""
    if score >= 90:
        return "EXCELLENT"
    elif score >= 80:
        return "GOOD"
    elif score >= 60:
        return "NEEDS WORK"
    elif score >= 40:
        return "POOR"
    else:
        return "CRITICAL"


def _bar(value: float, max_value: float, width: int = 20) -> str:
    """Render a simple horizontal bar chart."""
    filled = int((value / max(max_value, 1)) * width)
    filled = min(filled, width)
    return "#" * filled + "-" * (width - filled)


def render(result: AnalysisResult, use_json: bool = False, explain: bool = False) -> str:
    """
    Render an AnalysisResult for terminal display.

    Args:
        result: The analysis result to render.
        use_json: If True, output raw JSON instead of pretty formatting.

    Returns:
        A formatted string ready for print().
    """
    if use_json:
        return json.dumps(result.to_dict(), indent=2)

    lines: list[str] = []
    data = result.to_dict()
    score = data["score"]

    # ── Header ──────────────────────────────────────────────────────
    lines.append("")
    lines.append(f"{C.BOLD}{C.CYAN}+{'=' * 50}+{C.RESET}")
    lines.append(f"{C.BOLD}{C.CYAN}|          CONTEXTOPS -- Context Analysis           |{C.RESET}")
    lines.append(f"{C.BOLD}{C.CYAN}+{'=' * 50}+{C.RESET}")
    lines.append("")

    # ── Score ────────────────────────────────────────────────────────
    color = _score_color(score)
    label = _score_label(score)
    lines.append(f"  {C.BOLD}Context Score:{C.RESET}  {color}{C.BOLD} {score} / 100 {C.RESET}  {C.DIM}({label}){C.RESET}")
    lines.append("")

    # ── Roast (opt-in) ───────────────────────────────────────────────
    if result.roast is not None:
        lines.append(f"  {C.BOLD}{C.YELLOW}{'─' * 48}{C.RESET}")
        lines.append(f"  {C.BOLD}{C.YELLOW}  \"{result.roast.overall}\"{C.RESET}")
        if result.roast.dimensions:
            for dr in result.roast.dimensions:
                dim_color = C.RED if dr.severity == "high" else C.YELLOW
                lines.append(
                    f"  {dim_color}  [{dr.dimension.upper()}] {dr.roast}{C.RESET}"
                )
        lines.append(f"  {C.BOLD}{C.YELLOW}{'─' * 48}{C.RESET}")
        lines.append("")

    # ── Score Breakdown ─────────────────────────────────────────────
    bd = data["score_breakdown"]
    lines.append(f"  {C.BOLD}Score Breakdown:{C.RESET}")
    lines.append(f"  {C.DIM}{'-' * 48}{C.RESET}")

    penalties = [
        ("Redundancy", bd["redundancy_penalty"], 30, C.RED),
        ("Density", bd["density_penalty"], 30, C.YELLOW),
        ("Structure",  bd["structure_penalty"],  20, C.MAGENTA),
        ("Concentration",  bd["concentration_penalty"],  20, C.BLUE),
    ]
    for name, value, max_val, clr in penalties:
        bar = _bar(value, max_val)
        lines.append(
            f"  {clr}  {name:<12}{C.RESET} "
            f"-{value:>5.1f} / {max_val}  {C.DIM}{bar}{C.RESET}"
        )

    lines.append(f"  {C.DIM}{'-' * 48}{C.RESET}")
    lines.append(f"  {C.BOLD}  Total Penalty{C.RESET}  -{bd['total_penalty']:>5.1f} / 100")
    lines.append("")

    # ── Token Breakdown ─────────────────────────────────────────────
    tb = data["token_breakdown"]
    lines.append(f"  {C.BOLD}Token Breakdown:{C.RESET}")
    lines.append(f"  {C.DIM}{'-' * 48}{C.RESET}")
    if tb['wasted_tokens'] > 0:
        lines.append(f"  {C.DIM}Cost Model: ~15 penalty points per 1,000 wasted tokens (capped at 30){C.RESET}")
    lines.append(f"    Total tokens:    {C.BOLD}{tb['total_tokens']:,}{C.RESET}")
    lines.append(f"    Wasted tokens:   {C.RED}{tb['wasted_tokens']:,}{C.RESET}")
    lines.append(f"    Estimated cost:  ${tb['estimated_cost_usd']:.4f}")
    lines.append("")

    if tb["by_type"]:
        lines.append(f"    {C.DIM}By Type:{C.RESET}")
        for ctx_type, tokens in sorted(tb["by_type"].items(), key=lambda x: -x[1]):
            pct = (tokens / max(1, tb["total_tokens"])) * 100
            bar = _bar(tokens, tb["total_tokens"], width=15)
            lines.append(
                f"      {ctx_type:<12} {tokens:>6,} tokens  ({pct:>4.1f}%)  {C.DIM}{bar}{C.RESET}"
            )
        lines.append("")

    # ── Shadow Signals (Phase 2 Preview) ────────────────────────────
    if "density_signal" in data:
        ds = data["density_signal"]
        effect = data.get("density_effect", "shadow")
        if effect == "shadow":
            lines.append(f"  {C.BOLD}{C.CYAN}Shadow Metrics (Phase 2 Preview):{C.RESET}")
            lines.append(f"  {C.DIM}{'-' * 48}{C.RESET}")
            
            lines.append(f"    Format Overhead:     {ds['format_overhead']:.2f}")
            lines.append(f"    Whitespace Waste:    {ds['whitespace_waste']:.2f}")
            lines.append(f"    Entropy Compression: {ds['entropy_compression']:.2f}")
            lines.append(f"    {C.BOLD}Total Density Signal:{C.RESET} {C.YELLOW}{ds['total_density_signal']:.2f}{C.RESET}")
            lines.append(f"    {C.DIM}(This is a shadow metric and does not affect the score yet){C.RESET}")
            lines.append("")


    # ── Findings ────────────────────────────────────────────────────
    if not explain:
        redundancy_findings = data["findings"]["redundancy"]
        structure_findings = data["findings"]["structure"]

        if redundancy_findings or structure_findings:
            lines.append(f"  {C.BOLD}Findings:{C.RESET}")
            lines.append(f"  {C.DIM}{'-' * 48}{C.RESET}")

        for f in redundancy_findings:
            icon = "[~]" if f["classification"] == "expected_overlap" else "[!]"
            lines.append(
                f"    {icon}  {C.YELLOW}{f['classification'].upper()}{C.RESET}: "
                f"{f['detail']}"
            )
            lines.append(
                f"       {C.DIM}similarity: {f['similarity']:.0%} | "
                f"waste: {f['waste_tokens']:,} tokens{C.RESET}"
            )

        for f in structure_findings:
            sev_color = C.RED if f["severity"] in ("high", "critical") else C.YELLOW
            lines.append(
                f"    [S]  {sev_color}{f['issue']}{C.RESET}: "
                f"{f['type']} is {f['actual_ratio']:.0%} of context"
            )

        if redundancy_findings or structure_findings:
            lines.append("")

    # ── Recommendations ─────────────────────────────────────────────
    if not explain:
        recs = data["recommendations"]
        if recs:
            lines.append(f"  {C.BOLD}{C.GREEN}Recommendations:{C.RESET}")
            lines.append(f"  {C.DIM}{'-' * 48}{C.RESET}")

            for i, rec in enumerate(recs, 1):
                sev_color = C.RED if rec["severity"] in ("high", "critical") else C.YELLOW
                lines.append(
                    f"    {C.BOLD}{i}. {sev_color}{rec['issue']}{C.RESET}"
                )
                lines.append(
                    f"       {C.GREEN}Impact: {rec['impact']}{C.RESET} | "
                    f"Save: {rec['token_savings']:,} tokens"
                )
                lines.append(
                    f"       {C.CYAN}Fix: {rec['fix']}{C.RESET}"
                )
                lines.append("")

    # ── Explain Mode ────────────────────────────────────────────────
    if explain:
        # Deterministic sorting
        sorted_recs = sorted(
            result.recommendations,
            key=lambda r: (-r.impact_score, -r.token_savings, r.issue)
        )
        
        if sorted_recs:
            lines.append(f"  {C.BOLD}{C.MAGENTA}Top Score Drivers:{C.RESET}")
            lines.append(f"  {C.DIM}{'-' * 48}{C.RESET}")
            
            top_recs = sorted_recs[:3]
            for i, r in enumerate(top_recs, 1):
                lines.append(f"    {C.BOLD}{i}. {r.issue}{C.RESET}")
                lines.append(f"       Impact: {C.RED}-{round(r.impact_score, 1)}{C.RESET}")
                lines.append("")
                
            hidden_count = len(sorted_recs) - 3
            if hidden_count > 0:
                lines.append(f"    {C.DIM}+ {hidden_count} minor issues hidden{C.RESET}")
                lines.append("")
                
            # Potential Score Summary
            top_driver = sorted_recs[0]
            potential_score = min(100, score + round(top_driver.impact_score))
            
            lines.append(f"  {C.BOLD}{C.GREEN}Why This Score Is Not Higher{C.RESET}")
            lines.append(f"  {C.DIM}{'-' * 48}{C.RESET}")
            lines.append(f"    Potential Score: {C.GREEN}{C.BOLD}{potential_score}{C.RESET}")
            lines.append(f"    Current Score:   {color}{C.BOLD}{score}{C.RESET}")
            lines.append("")
            lines.append(f"    {C.BOLD}Largest Opportunity:{C.RESET}")
            lines.append(f"    {C.CYAN}{top_driver.fix}{C.RESET}")
            lines.append("")
            lines.append(f"    {C.BOLD}Expected Gain:{C.RESET}")
            lines.append(f"    {C.GREEN}+{round(top_driver.impact_score, 1)} points{C.RESET}")
            if top_driver.token_savings > 0:
                lines.append(f"    {C.GREEN}{top_driver.token_savings:,} tokens saved{C.RESET}")
            lines.append("")

    # ── Footer ──────────────────────────────────────────────────────
    lines.append(f"  {C.DIM}contextops v{data['metadata'].get('version', '0.1.0')} "
                 f"| {data['metadata'].get('item_count', 0)} items analyzed{C.RESET}")
    lines.append("")
    lines.append(f"  {C.YELLOW}{C.BOLD}LIMITATION:{C.RESET}")
    lines.append(f"  {C.DIM}This tool measures structural density, not semantic usefulness.{C.RESET}")
    lines.append(f"  {C.DIM}A high score does not guarantee the LLM has the right facts to answer.{C.RESET}")
    lines.append("")

    return "\n".join(lines)


def render_stability(report: Any) -> str:
    """
    Render a StabilityReport for terminal display.
    Expects report of type contextops.api.stability.StabilityReport.
    """
    lines: list[str] = []

    lines.append("")
    lines.append(f"{C.BOLD}{C.CYAN}+{'=' * 50}+{C.RESET}")
    lines.append(f"{C.BOLD}{C.CYAN}|        CONTEXTOPS -- Stability Report             |{C.RESET}")
    lines.append(f"{C.BOLD}{C.CYAN}+{'=' * 50}+{C.RESET}")
    lines.append("")

    lines.append(f"  {C.BOLD}Base Score:{C.RESET} {report.base_score}")
    lines.append(f"  {C.BOLD}Base Tokens:{C.RESET} {report.base_tokens:,}")
    lines.append(f"  {C.BOLD}Base Waste Tokens:{C.RESET} {report.base_waste_tokens:,}")
    lines.append("")

    for inv in report.invariants:
        lines.append(f"  {C.BOLD}{inv.name}{C.RESET}")
        
        if inv.passed:
            lines.append(f"  {C.GREEN}PASS{C.RESET}")
        else:
            lines.append(f"  {C.RED}FAIL{C.RESET}")

        for key, value in inv.diagnostic_info.items():
            lines.append(f"    {C.DIM}{key}: {value}{C.RESET}")
            
        lines.append("")

    score = report.score_percentage
    color = _score_color(score)
    passed_count = sum(1 for inv in report.invariants if inv.passed)
    total_count = len(report.invariants)

    if score >= 90:
        confidence = "High"
    elif score >= 70:
        confidence = "Medium"
    else:
        confidence = "Low"

    lines.append(f"  {C.DIM}{'-' * 48}{C.RESET}")
    lines.append(f"  {C.BOLD}Stability Score:{C.RESET}")
    lines.append(f"  {color}{C.BOLD}{score}%{C.RESET}  {C.DIM}({passed_count}/{total_count} passed){C.RESET}")
    lines.append(f"  {C.BOLD}Confidence: {confidence}{C.RESET}")
    lines.append("")

    return "\n".join(lines)


def render_diff(diff: Any) -> str:
    """
    Render a ContextDiffResult for terminal display.
    Expects diff of type contextops.api.diff.ContextDiffResult.
    """
    lines: list[str] = []

    lines.append("")
    lines.append(f"{C.BOLD}{C.CYAN}+{'=' * 50}+{C.RESET}")
    lines.append(f"{C.BOLD}{C.CYAN}|        CONTEXTOPS -- Context Diff                 |{C.RESET}")
    lines.append(f"{C.BOLD}{C.CYAN}+{'=' * 50}+{C.RESET}")
    lines.append("")

    # ── Score Section ───────────────────────────────────────────────
    score_a = diff.result_a.score
    score_b = diff.result_b.score
    score_color = C.GREEN if diff.score_delta > 0 else (C.RED if diff.score_delta < 0 else C.RESET)
    score_sign = "+" if diff.score_delta > 0 else ""
    lines.append(f"  {C.BOLD}Score:{C.RESET} {score_a} -> {score_b} "
                 f"({score_color}{score_sign}{diff.score_delta}{C.RESET})")
    lines.append("")

    # ── Tokens / Cost ───────────────────────────────────────────────
    lines.append(f"  {C.BOLD}Tokens / Cost{C.RESET}")
    tok_a = diff.result_a.token_breakdown.total_tokens
    tok_b = diff.result_b.token_breakdown.total_tokens
    tok_color = C.GREEN if diff.token_delta < 0 else (C.RED if diff.token_delta > 0 else C.RESET)
    tok_sign = "+" if diff.token_delta > 0 else ""
    lines.append(f"  Tokens: {tok_a:,} -> {tok_b:,} "
                 f"({tok_color}{tok_sign}{diff.token_delta:,}{C.RESET})")
                 
    cost_a = diff.result_a.token_breakdown.estimated_cost_usd
    cost_b = diff.result_b.token_breakdown.estimated_cost_usd
    cost_color = C.GREEN if diff.cost_delta < 0 else (C.RED if diff.cost_delta > 0 else C.RESET)
    cost_sign = "+" if diff.cost_delta > 0 else ""
    lines.append(f"  Cost: ${cost_a:.4f} -> ${cost_b:.4f} "
                 f"({cost_color}{cost_sign}${abs(diff.cost_delta):.4f}{C.RESET})")
    lines.append("")

    # ── Structure Changes ───────────────────────────────────────────
    lines.append(f"  {C.BOLD}Structure Changes (Penalties){C.RESET}")
    for key, delta in diff.structure_delta.items():
        if abs(delta) < 0.01:
            continue
        color = C.GREEN if delta < 0 else C.RED
        sign = "+" if delta > 0 else ""
        lines.append(f"  {key.capitalize()}: {color}{sign}{delta:.2f}{C.RESET}")
    
    if all(abs(v) < 0.01 for v in diff.structure_delta.values()):
        lines.append(f"  {C.DIM}No significant structure changes.{C.RESET}")
    lines.append("")

    # ── Recommendation Lifecycle ────────────────────────────────────
    lines.append(f"  {C.BOLD}Recommendation Lifecycle{C.RESET}")
    
    if diff.resolved_recommendations:
        lines.append(f"  {C.GREEN}Resolved:{C.RESET}")
        for r in diff.resolved_recommendations:
            lines.append(f"    {C.GREEN}[-] {r.issue}{C.RESET}")
            
    if diff.new_recommendations:
        lines.append(f"  {C.RED}New:{C.RESET}")
        for r in diff.new_recommendations:
            lines.append(f"    {C.RED}[+] {r.issue}{C.RESET}")
            
    if diff.persisting_recommendations:
        lines.append(f"  {C.DIM}Persisting:{C.RESET}")
        for r in diff.persisting_recommendations:
            lines.append(f"    {C.DIM}[~] {r.issue}{C.RESET}")
            
    if not (diff.resolved_recommendations or diff.new_recommendations or diff.persisting_recommendations):
        lines.append(f"  {C.DIM}No recommendations.{C.RESET}")
        
    lines.append("")

    # ── Net Impact Summary ──────────────────────────────────────────
    lines.append(f"  {C.DIM}{'-' * 48}{C.RESET}")
    lines.append(f"  {C.BOLD}Net Impact Summary:{C.RESET}")
    
    # Generate impact bullets
    if diff.token_delta < 0:
        lines.append(f"  {C.GREEN}\u2714 Reduced token usage ({diff.token_delta:,}){C.RESET}")
    elif diff.token_delta > 0:
        lines.append(f"  {C.RED}\u2716 Increased token usage (+{diff.token_delta:,}){C.RESET}")
        
    if diff.score_delta > 0:
        lines.append(f"  {C.GREEN}\u2714 Improved score (+{diff.score_delta}){C.RESET}")
    elif diff.score_delta < 0:
        lines.append(f"  {C.RED}\u2716 Score degraded ({diff.score_delta}){C.RESET}")
        
    for key, delta in diff.structure_delta.items():
        if delta > 0.05:  # significant penalty increase
            lines.append(f"  {C.RED}\u2716 {key.replace('_', ' ')} penalty increased (+{delta:.2f}){C.RESET}")
        elif delta < -0.05:  # significant penalty decrease
            lines.append(f"  {C.GREEN}\u2714 {key.replace('_', ' ')} penalty decreased ({delta:.2f}){C.RESET}")

    lines.append("")
    
    if diff.net_impact == "IMPROVEMENT":
        lines.append(f"  Overall: {C.BG_GREEN}{C.WHITE}{C.BOLD} IMPROVEMENT {C.RESET}")
    elif diff.net_impact == "DEGRADATION":
        lines.append(f"  Overall: {C.BG_RED}{C.WHITE}{C.BOLD} DEGRADATION {C.RESET}")
    else:
        lines.append(f"  Overall: {C.BG_YELLOW}{C.WHITE}{C.BOLD} NEUTRAL {C.RESET}")
        
    lines.append("")

    return "\n".join(lines)
