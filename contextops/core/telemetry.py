"""
Telemetry tracking for ContextOps.
Writes local JSONL files for score trend analysis.
"""

import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from contextops.core.models import AnalysisResult

# Use CONTEXTOPS_TELEMETRY=1 or TELEMETRY_ENABLED=1 to enable.
_IS_ENABLED = os.environ.get("CONTEXTOPS_TELEMETRY", os.environ.get("TELEMETRY_ENABLED", "0")) == "1"


def get_telemetry_path() -> Path:
    """Resolve the path to the telemetry JSONL file."""
    if os.name == "nt":
        base_dir = Path(os.environ.get("USERPROFILE", "~")).expanduser()
    else:
        base_dir = Path("~").expanduser()
    
    contextops_dir = base_dir / ".contextops"
    return contextops_dir / "telemetry.jsonl"


def read_last_score() -> Optional[int]:
    """Read the last recorded score from the telemetry file."""
    telemetry_file = get_telemetry_path()
    if not telemetry_file.exists():
        return None
        
    try:
        with open(telemetry_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if not lines:
                return None
            last_line = lines[-1].strip()
            if not last_line:
                return None
            data = json.loads(last_line)
            return data.get("score")
    except Exception:
        return None


def record_event(result: AnalysisResult) -> None:
    """
    Record a telemetry event for the given AnalysisResult.
    Silently fails (with a RuntimeWarning) to avoid breaking CI.
    """
    if not _IS_ENABLED:
        return

    try:
        telemetry_file = get_telemetry_path()
        telemetry_file.parent.mkdir(parents=True, exist_ok=True)

        previous_score = read_last_score()
        
        # Calculate score delta
        score_delta = None
        if previous_score is not None:
            score_delta = result.score - previous_score

        # Extract top 3 findings by impact
        # Structure findings usually don't have impact_score explicitly mapped in models unless it's a Recommendation,
        # but the AnalysisResult has a `score_breakdown` and `structure_findings`.
        # For "top_findings", we'll just gather the issue names from the breakdown or findings.
        top_findings = []
        if result.redundancy_findings:
            top_findings.append("high_redundancy")
        
        for sf in result.structure_findings:
            if sf.issue:
                top_findings.append(sf.issue)
                
        # If no explicit structure/redundancy findings, but density is low:
        if result.score_breakdown.density_penalty > 0 and "low_density" not in top_findings:
            top_findings.append("low_density")

        # Limit to top 3 for brevity
        top_findings = top_findings[:3]

        event_data: Dict[str, Any] = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "score": result.score,
            "score_delta": score_delta,
            "archetype": result.archetype_resolved,
            "total_tokens": result.token_breakdown.total_tokens,
            "wasted_tokens": result.token_breakdown.wasted_tokens,
            "ci_status": "PASS" if result.score > 0 else "FAIL", # Simplistic CI status
            "version": "0.3.0", # TODO: dynamic version
            "top_findings": top_findings,
        }

        with open(telemetry_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event_data) + "\n")

    except Exception as e:
        warnings.warn(f"ContextOps telemetry write failed: {e}", RuntimeWarning)


def read_events(limit: int = 10) -> List[Dict[str, Any]]:
    """Read the last N events from the telemetry file."""
    telemetry_file = get_telemetry_path()
    if not telemetry_file.exists():
        return []
        
    try:
        events = []
        with open(telemetry_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events[-limit:]
    except Exception:
        return []

def get_trends(days: int = 30) -> Dict[str, Any]:
    """Compute basic trends over the last N days."""
    telemetry_file = get_telemetry_path()
    if not telemetry_file.exists():
        return {}
        
    now = datetime.utcnow()
    valid_events = []
    
    try:
        with open(telemetry_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                ts_str = event.get("ts", "").replace("Z", "")
                try:
                    ts = datetime.fromisoformat(ts_str)
                    if (now - ts).days <= days:
                        valid_events.append(event)
                except ValueError:
                    pass
    except Exception:
        pass
        
    if not valid_events:
        return {}
        
    scores = [e["score"] for e in valid_events if "score" in e]
    wasted = [e["wasted_tokens"] for e in valid_events if "wasted_tokens" in e]
    
    findings_count: Dict[str, int] = {}
    for e in valid_events:
        for finding in e.get("top_findings", []):
            findings_count[finding] = findings_count.get(finding, 0) + 1
            
    most_common = max(findings_count.items(), key=lambda x: x[1])[0] if findings_count else None
    
    avg_score = sum(scores) / len(scores) if scores else 0
    avg_wasted = sum(wasted) / len(wasted) if wasted else 0
    
    # Simple trend logic: compare first half of period to second half
    half_idx = len(scores) // 2
    if half_idx > 0 and len(scores) >= 4:
        older_avg = sum(scores[:half_idx]) / half_idx
        newer_avg = sum(scores[half_idx:]) / (len(scores) - half_idx)
        trend = newer_avg - older_avg
    else:
        # Fallback to total score delta if we don't have enough events
        # We can just sum deltas
        deltas = [e["score_delta"] for e in valid_events if e.get("score_delta") is not None]
        trend = sum(deltas) if deltas else 0.0

    return {
        "avg_score": avg_score,
        "avg_wasted": avg_wasted,
        "most_common_failure": most_common,
        "trend": trend,
        "event_count": len(valid_events)
    }
