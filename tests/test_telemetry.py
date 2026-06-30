import json
import os
import warnings
from pathlib import Path
from unittest.mock import patch, mock_open

import pytest

from contextops.core.models import AnalysisResult, ScoreBreakdown, TokenBreakdown
from contextops.core.telemetry import record_event, read_events


@pytest.fixture
def mock_telemetry_file(tmp_path):
    telemetry_file = tmp_path / ".contextops" / "telemetry.jsonl"
    with patch("contextops.core.telemetry.get_telemetry_path", return_value=telemetry_file):
        yield telemetry_file


@pytest.fixture
def dummy_result():
    return AnalysisResult(
        score=85,
        score_breakdown=ScoreBreakdown(
            redundancy_penalty=0,
            density_penalty=15,
            structure_penalty=0,
            concentration_penalty=0,
        ),
        token_breakdown=TokenBreakdown(
            total_tokens=1000,
            wasted_tokens=100,
            by_type={"chunk": 0, "system": 0, "message": 0, "tool": 0, "memory": 0},
        ),
        density_signal=0.5,
        redundancy_findings=[],
        structure_findings=[],
        recommendations=[],
        archetype_resolved="general",
        archetype_source="default",
    )


def test_telemetry_disabled_by_default(dummy_result, mock_telemetry_file):
    # Without environment variables, it shouldn't write anything.
    with patch("contextops.core.telemetry._IS_ENABLED", False):
        record_event(dummy_result)
        assert not mock_telemetry_file.exists()


def test_telemetry_writes_event_when_enabled(dummy_result, mock_telemetry_file):
    with patch("contextops.core.telemetry._IS_ENABLED", True):
        record_event(dummy_result)
        
        assert mock_telemetry_file.exists()
        with open(mock_telemetry_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            
            assert data["score"] == 85
            assert data["score_delta"] is None
            assert data["archetype"] == "general"
            assert data["total_tokens"] == 1000
            assert "low_density" in data["top_findings"]


def test_telemetry_calculates_delta(dummy_result, mock_telemetry_file):
    with patch("contextops.core.telemetry._IS_ENABLED", True):
        # Write first event (85)
        record_event(dummy_result)
        
        # Write second event (90)
        dummy_result.score = 90
        record_event(dummy_result)
        
        with open(mock_telemetry_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 2
            
            event2 = json.loads(lines[1])
            assert event2["score"] == 90
            assert event2["score_delta"] == 5


def test_telemetry_read_events(dummy_result, mock_telemetry_file):
    with patch("contextops.core.telemetry._IS_ENABLED", True):
        for i in range(5):
            dummy_result.score = 80 + i
            record_event(dummy_result)
            
        events = read_events(limit=3)
        assert len(events) == 3
        assert events[0]["score"] == 82
        assert events[1]["score"] == 83
        assert events[2]["score"] == 84


def test_telemetry_write_failure_warns_silently(dummy_result):
    with patch("contextops.core.telemetry._IS_ENABLED", True):
        with patch("contextops.core.telemetry.get_telemetry_path", side_effect=PermissionError("Denied")):
            with pytest.warns(RuntimeWarning, match="ContextOps telemetry write failed: Denied"):
                record_event(dummy_result)
