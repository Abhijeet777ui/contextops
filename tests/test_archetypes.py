"""
Tests for Phase 6: Archetype-Aware Scoring.

Coverage:
  - Resolution source trace
  - Payload override
  - rag archetype suppresses retrieval dominance
  - agent archetype suppresses tool sprawl
  - general archetype raises findings normally
  - archetype mismatch advisory
"""

import pytest

from contextops.api.inspect import inspect_context
from contextops.core.config import ContextOpsConfig
from contextops.core.models import FindingSeverity


def test_rag_suppresses_retrieval_dominance():
    # ~80% retrieval (between generic 70% and rag 95%)
    payload = {
        "system": "sys " * 25,
        "chunks": [{"content": "chunk " * 100}],
    }
    # With 'rag' profile
    result = inspect_context(payload, cli_profile="rag")
    issues = [f.issue for f in result.structure_findings]
    assert "Retrieval dominance" not in issues
    assert result.archetype_resolved == "rag"
    assert result.archetype_source == "cli"


def test_agent_suppresses_tool_sprawl():
    # ~80% tool (between generic 60% and agent 90%)
    payload = {
        "system": "sys " * 25,
        "tools": [{"content": "tool " * 100}],
    }
    # With 'agent' profile
    result = inspect_context(payload, cli_profile="agent")
    issues = [f.issue for f in result.structure_findings]
    assert "Tool output sprawl" not in issues
    assert result.archetype_resolved == "agent"


def test_general_still_fires():
    # ~80% retrieval under general profile should trigger finding
    payload_rag = {
        "system": "sys " * 25,
        "chunks": [{"content": "chunk " * 100}],
    }
    result_rag = inspect_context(payload_rag, cli_profile="general")
    issues_rag = [f.issue for f in result_rag.structure_findings]
    assert "Retrieval dominance" in issues_rag
    assert result_rag.archetype_resolved == "general"

    # ~80% tool under general profile should trigger finding
    payload_agent = {
        "system": "sys " * 25,
        "tools": [{"content": "tool " * 100}],
    }
    result_agent = inspect_context(payload_agent, cli_profile="general")
    issues_agent = [f.issue for f in result_agent.structure_findings]
    assert "Tool output sprawl" in issues_agent


def test_payload_archetype_override():
    # ~80% retrieval
    payload = {
        "archetype": "rag",
        "system": "sys " * 25,
        "chunks": [{"content": "chunk " * 100}],
    }
    result = inspect_context(payload)
    issues = [f.issue for f in result.structure_findings]
    assert "Retrieval dominance" not in issues
    assert result.archetype_resolved == "rag"
    assert result.archetype_source == "payload"


def test_archetype_mismatch_advisory():
    # 'rag' profile but heavy tool usage (over 20%)
    payload = {
        "system": "sys",
        "tools": [{"content": "tool " * 40}],
    }
    result = inspect_context(payload, cli_profile="rag")
    # Finding should be suppressed/not there as a structure finding, but the advisory recommendation should exist
    advisories = [
        r for r in result.recommendations 
        if r.issue == "archetype_mismatch"
    ]
    assert len(advisories) == 1
    assert advisories[0].severity == FindingSeverity.LOW
    assert advisories[0].impact_score == 0.0
    assert "RAG archetype declared but tool output exceeds 20%" in advisories[0].fix


def test_resolution_source_trace():
    payload = {
        "archetype": "rag", # payload
        "system": "sys",
    }
    # 1. Payload wins if no CLI or API
    res1 = inspect_context(payload)
    assert res1.archetype_resolved == "rag"
    assert res1.archetype_source == "payload"
    assert res1.to_dict()["archetype"]["source"] == "payload"

    # 2. API wins over payload
    res2 = inspect_context(payload, archetype="agent")
    assert res2.archetype_resolved == "agent"
    assert res2.archetype_source == "api"

    # 3. CLI wins over all
    res3 = inspect_context(payload, archetype="agent", cli_profile="chatbot")
    assert res3.archetype_resolved == "chatbot"
    assert res3.archetype_source == "cli"

    # 4. Config wins if payload has none and no CLI/API
    cfg = ContextOpsConfig(context_profile="toolchain")
    res4 = inspect_context({"system": "sys"}, config=cfg)
    assert res4.archetype_resolved == "toolchain"
    assert res4.archetype_source == "config"

    # 5. Default wins if nothing specified
    res5 = inspect_context({"system": "sys"})
    assert res5.archetype_resolved == "general"
    assert res5.archetype_source == "default"
