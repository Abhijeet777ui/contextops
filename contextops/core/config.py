"""
Configuration models for ContextOps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ContextOpsConfig:
    """
    Configuration for ContextOps analyzers.
    
    Thresholds represent the maximum allowed ratio of context for a given type.
    """
    retrieval_max_ratio: float = 0.70
    system_max_ratio: float = 0.50
    memory_max_ratio: float = 0.50
    tool_max_ratio: float = 0.60
    
    # Opt-in flag to enable LSH/MinHash semantic paraphrase detection
    strict_semantic: bool = False

    # Roast mode — off by default (enterprise-safe).
    # When enabled, CLI and JSON output include score-band commentary.
    # Explicitly excluded from the determinism contract (random per-run).
    roast_enabled: bool = False
    
    # "strict" means default thresholds are used (standardized score).
    # "custom" means user has overridden thresholds.
    mode: str = "strict"
    version: str = "1.0"

    @classmethod
    def default(cls) -> ContextOpsConfig:
        return cls()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextOpsConfig:
        config = cls()
        has_custom = False
        
        if "retrieval_max_ratio" in data:
            config.retrieval_max_ratio = float(data["retrieval_max_ratio"])
            has_custom = True
        if "system_max_ratio" in data:
            config.system_max_ratio = float(data["system_max_ratio"])
            has_custom = True
        if "memory_max_ratio" in data:
            config.memory_max_ratio = float(data["memory_max_ratio"])
            has_custom = True
        if "tool_max_ratio" in data:
            config.tool_max_ratio = float(data["tool_max_ratio"])
            has_custom = True
            
        if "strict_semantic" in data:
            config.strict_semantic = bool(data["strict_semantic"])
            has_custom = True

        if "roast_enabled" in data:
            config.roast_enabled = bool(data["roast_enabled"])
            # Note: roast_enabled does not set has_custom — it's a UI preference,
            # not a scoring threshold change.
            
        if has_custom:
            config.mode = "custom"
            
        return config

    @classmethod
    def from_file(cls, path: str | Path) -> ContextOpsConfig:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
