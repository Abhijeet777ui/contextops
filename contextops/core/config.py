"""
Configuration models for ContextOps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_PROFILE_PRESETS = {
    "generic": {
        "retrieval_max_ratio": 0.70,
        "system_max_ratio": 0.50,
        "memory_max_ratio": 0.50,
        "tool_max_ratio": 0.60,
        "concentration_weight": 1.0,
    },
    "rag": {
        "retrieval_max_ratio": 0.95,
        "system_max_ratio": 0.40,
        "memory_max_ratio": 0.20,
        "tool_max_ratio": 0.30,
        "concentration_weight": 0.3,
    },
    "agent": {
        "retrieval_max_ratio": 0.50,
        "system_max_ratio": 0.40,
        "memory_max_ratio": 0.40,
        "tool_max_ratio": 0.90,
        "concentration_weight": 0.7,
    },
    "chatbot": {
        "retrieval_max_ratio": 0.40,
        "system_max_ratio": 0.50,
        "memory_max_ratio": 0.85,
        "tool_max_ratio": 0.30,
        "concentration_weight": 0.6,
    },
    "toolchain": {
        "retrieval_max_ratio": 0.50,
        "system_max_ratio": 0.40,
        "memory_max_ratio": 0.30,
        "tool_max_ratio": 0.95,
        "concentration_weight": 0.5,
    },
}

@dataclass
class ContextOpsConfig:
    """
    Configuration for ContextOps analyzers.
    
    Thresholds represent the maximum allowed ratio of context for a given type.
    """
    context_profile: str = "generic"
    
    retrieval_max_ratio: float = 0.70
    system_max_ratio: float = 0.50
    memory_max_ratio: float = 0.50
    tool_max_ratio: float = 0.60
    concentration_weight: float = 1.0

    # RS threshold configuration (Phase 0 Bug 4 fix — no longer hardcoded in redundancy.py).
    # rs_minimum: findings below this RS score are silently ignored (not noise).
    # rs_advisory_minimum: findings between advisory and minimum are reported as
    #   informational-only with zero penalty contribution.
    # Default of 0.35 is the calibrated value for general technical corpora.
    # For high-overlap domains (legal, API specs), tune down to ~0.28.
    rs_minimum: float = 0.35
    rs_advisory_minimum: float = 0.25
    
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
    def default(cls, profile: str = "generic") -> ContextOpsConfig:
        config = cls(context_profile=profile)
        if profile in _PROFILE_PRESETS:
            preset = _PROFILE_PRESETS[profile]
            for k, v in preset.items():
                setattr(config, k, v)
        return config

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextOpsConfig:
        profile = data.get("context_profile", "generic")
        config = cls.default(profile=profile)
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
        if "concentration_weight" in data:
            config.concentration_weight = float(data["concentration_weight"])
            has_custom = True
        if "rs_minimum" in data:
            config.rs_minimum = float(data["rs_minimum"])
            has_custom = True
        if "rs_advisory_minimum" in data:
            config.rs_advisory_minimum = float(data["rs_advisory_minimum"])
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

    def with_profile(self, profile: str) -> "ContextOpsConfig":
        """
        Return a new ContextOpsConfig with the given profile's preset thresholds
        applied on top of this config's settings.

        The original config is never mutated. This is the immutable resolution
        pattern used by the engine to apply an archetype without side effects.

        Args:
            profile: One of 'general', 'rag', 'agent', 'chatbot', 'toolchain'.
                     Unknown profiles fall back to 'general' silently.

        Returns:
            A fresh ContextOpsConfig with preset thresholds merged in.
        """
        import copy
        new_config = copy.copy(self)
        
        if self.context_profile == profile:
            return new_config
            
        preset = _PROFILE_PRESETS.get(profile, _PROFILE_PRESETS["generic"])
        for k, v in preset.items():
            setattr(new_config, k, v)
        new_config.context_profile = profile
        return new_config
