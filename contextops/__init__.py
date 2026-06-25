"""ContextOps — Context observability for LLM applications."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from contextops.core.config import ContextOpsConfig

if TYPE_CHECKING:
    from contextops.integrations.langchain.callback import ContextOpsCallbackHandler

__version__ = "0.3.0"


class ContextOps:
    """
    ContextOps public entry point.

    This is a static factory — not a global patcher.
    It returns an explicit, inspectable callback handler you attach yourself.

    Canonical usage (the ONLY recommended pattern):

        from contextops import ContextOps

        chain = chain.with_config({
            "callbacks": [ContextOps.auto()]
        })

    That's it. No magic. No global side-effects. Full debuggability.
    """

    @staticmethod
    def auto(
        mode: str = "log",
        min_score: int = 70,
        profile: str = "generic",
        debug: bool = False,
        config: Optional[ContextOpsConfig] = None,
    ) -> "ContextOpsCallbackHandler":
        """
        Create a ready-to-use ContextOps callback handler.

        This is Phase 1 of the distribution layer: an explicit, opt-in wrapper.
        No global monkey-patching. No hidden injection.

        Args:
            mode:      Reaction mode — "log" (default), "warn", or "block".
                       - "log":   prints the score report to stdout.
                       - "warn":  emits a Python warning if score < min_score.
                       - "block": raises ContextOpsScoreError if score < min_score.
            min_score: Score threshold for "warn" / "block" modes (default: 70).
            profile:   Architecture profile preset to use (default: "generic").
                       Options: "generic", "rag", "agent", "chatbot", "toolchain".
            debug:     If True, prints the full context reconstruction payload
                       before the score. Useful for verifying capture is correct.
            config:    Optional ContextOpsConfig to override internal thresholds.
                       If provided, `profile` is ignored.

        Returns:
            ContextOpsCallbackHandler — a LangChain BaseCallbackHandler instance.

        Example::

            # Basic — just see the score
            chain.with_config({"callbacks": [ContextOps.auto()]})

            # Debug mode — see exactly what was captured
            chain.with_config({"callbacks": [ContextOps.auto(debug=True)]})

            # Block mode — fail hard on bad context quality
            chain.with_config({
                "callbacks": [ContextOps.auto(mode="block", min_score=75)]
            })
        """
        from contextops.integrations.langchain.callback import ContextOpsCallbackHandler

        if config is None:
            config = ContextOpsConfig.default(profile=profile)

        return ContextOpsCallbackHandler(
            mode=mode,
            min_score=min_score,
            debug=debug,
            config=config,
        )

    @staticmethod
    def langchain_config(
        mode: str = "log",
        min_score: int = 70,
        profile: str = "generic",
        debug: bool = False,
        config: Optional[ContextOpsConfig] = None,
    ) -> dict:
        """
        Return a ready-to-spread LangChain config dict.

        This is the most ergonomic pattern for inline chain configuration.

        Example::

            chain = chain.with_config(ContextOps.langchain_config(debug=True))

        Equivalent to::

            chain = chain.with_config({
                "callbacks": [ContextOps.auto(debug=True)]
            })
        """
        return {
            "callbacks": [
                ContextOps.auto(
                    mode=mode,
                    min_score=min_score,
                    profile=profile,
                    debug=debug,
                    config=config,
                )
            ]
        }

    @staticmethod
    def version() -> str:
        """Return the current ContextOps version."""
        return __version__


# ── Convenience re-exports ──────────────────────────────────────────────────
# These allow: from contextops import ContextOpsCallbackHandler
# without requiring users to know the internal module structure.

from contextops.integrations.langchain.callback import (  # noqa: E402
    ContextOpsCallbackHandler,
    ContextOpsScoreError,
)

__all__ = [
    "ContextOps",
    "ContextOpsCallbackHandler",
    "ContextOpsScoreError",
    "__version__",
]
