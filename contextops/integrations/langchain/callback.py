"""
LangChain Callback Handler for ContextOps.

This is a stateful callback handler that captures retrieval and tool events
in a temporal buffer (keyed by run_id), and then reconstructs the final 
context payload precisely at `on_chat_model_start` (the execution gate).

It provides a "zero-architecture-change" way to get pre-execution 
context observability.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from contextops.api.inspect import inspect_context
from contextops.core.config import ContextOpsConfig
from contextops.core.models import ContextBundle, ContextItem, ContextType

# Graceful optional imports for zero-friction installation
try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage, ToolMessage
    from langchain_core.documents import Document
    LANGCHAIN_INSTALLED = True
except ImportError:
    # Dummy stubs if LangChain is not installed
    class BaseCallbackHandler:  # type: ignore
        pass
    BaseMessage = object  # type: ignore
    SystemMessage = object  # type: ignore
    AIMessage = object  # type: ignore
    HumanMessage = object  # type: ignore
    ToolMessage = object  # type: ignore
    Document = object  # type: ignore
    LANGCHAIN_INSTALLED = False


logger = logging.getLogger(__name__)


class ContextOpsScoreError(Exception):
    """Raised when a context score falls below the blocking threshold."""
    pass


class ContextOpsCallbackHandler(BaseCallbackHandler):
    """
    ContextOps LangChain Callback Handler.
    
    Acts as a runtime diagnostic signal inside LLM pipelines.
    Buffers events by run_id, merging them into a final ContextBundle
    just before LLM execution, and emitting a highly actionable developer warning.
    """

    def __init__(
        self,
        mode: str = "log",
        min_score: int = 70,
        debug: bool = False,
        config: Optional[ContextOpsConfig] = None
    ) -> None:
        """
        Args:
            mode: "log", "warn", or "block".
            min_score: Threshold below which a warning or block is triggered.
            debug: If True, prints verbose truth-verification context payloads.
            config: Optional ContextOpsConfig to override thresholds.
        """
        if not LANGCHAIN_INSTALLED:
            raise ImportError(
                "LangChain is not installed. Please install it with `pip install langchain-core` "
                "to use the ContextOpsCallbackHandler."
            )
            
        self.mode = mode
        self.min_score = min_score
        self.debug = debug
        self.config = config or ContextOpsConfig.default()

        # Temporal event buffer keyed by root run_id
        # { run_id: { "retrieval": [...], "tools": [...] } }
        self._run_state: Dict[UUID, Dict[str, List[Any]]] = {}
        
        # Parent tracking to resolve deep LCEL hierarchies
        # { run_id: parent_run_id }
        self._run_tree: Dict[UUID, UUID] = {}
        
        # Last trace payload for export
        self.last_trace: Optional[Dict[str, Any]] = None

        # Pre-run buffer for manually injected chunks (custom retriever pattern)
        # Chunks added here are merged into the next on_chat_model_start call.
        self._pending_chunks: List[Dict[str, str]] = []
        self._pending_tools: List[Dict[str, str]] = []

    def _get_root_run_id(self, run_id: UUID) -> UUID:
        """Walk up the execution tree to find the root run_id."""
        current = run_id
        while current in self._run_tree and self._run_tree[current] is not None:
            current = self._run_tree[current]
        return current

    def _get_or_create_state(self, run_id: UUID) -> Dict[str, List[Any]]:
        root_id = self._get_root_run_id(run_id)
        if root_id not in self._run_state:
            self._run_state[root_id] = {
                "retrieval": [],
                "tools": [],
            }
        return self._run_state[root_id]

    def inject_chunks(self, documents: List[Any]) -> None:
        """
        Manually inject retrieved documents into the pending buffer.

        Use this when wrapping a custom retriever that does NOT go through
        LangChain's standard retriever interface. Injected chunks are merged
        into the next on_chat_model_start call automatically.

        Args:
            documents: List of LangChain Document objects or dicts with
                       'content' and optionally 'source' keys.

        Example::

            handler = ContextOps.auto(debug=True)
            docs = my_retriever.get_docs(query)
            handler.inject_chunks(docs)          # <-- inject here
            chain.with_config({"callbacks": [handler]}).invoke(...)
        """
        for doc in documents:
            if hasattr(doc, "page_content"):
                # LangChain Document object
                source = doc.metadata.get("source", "unknown") if hasattr(doc, "metadata") else "unknown"
                self._pending_chunks.append({"content": doc.page_content, "source": source})
            elif isinstance(doc, dict):
                self._pending_chunks.append({
                    "content": doc.get("content", str(doc)),
                    "source": doc.get("source", "unknown"),
                })
            else:
                self._pending_chunks.append({"content": str(doc), "source": "unknown"})

    def inject_tool_output(self, output: str, name: str = "tool") -> None:
        """
        Manually inject a tool output into the pending buffer.

        Same pattern as inject_chunks() but for tool outputs.
        """
        self._pending_tools.append({"content": str(output), "source": name})

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Track the execution tree hierarchy."""
        if parent_run_id:
            self._run_tree[run_id] = parent_run_id

    # ── Hook 1: Retriever End (Capture RAG chunks) ───────────────────────────────────

    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id:
            self._run_tree[run_id] = parent_run_id

    def on_retriever_end(
        self,
        documents: List[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Silently stash retrieved documents into the temporal buffer."""
        # Attach to the root run_id by walking up the tree
        target_run_id = self._get_root_run_id(run_id)
        state = self._get_or_create_state(target_run_id)
        
        for doc in documents:
            source = doc.metadata.get("source", "unknown_source")
            state["retrieval"].append({
                "content": doc.page_content,
                "source": source
            })

    # ── Hook 2: Tool End (Capture Tool outputs) ──────────────────────────────────────

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id:
            self._run_tree[run_id] = parent_run_id

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Silently stash tool outputs into the temporal buffer."""
        target_run_id = self._get_root_run_id(run_id)
        state = self._get_or_create_state(target_run_id)
        
        state["tools"].append({
            "content": str(output),
            "source": name or "tool"
        })

    # ── Hook 3: Chat Model Start (The Execution Gate) ────────────────────────────────

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Intercept the final prompt exact moments before it goes to the LLM.
        Merge stashed retrieval/tool context with the prompt, score it, and emit.
        """
        if parent_run_id:
            self._run_tree[run_id] = parent_run_id
            
        target_run_id = self._get_root_run_id(run_id)
        state = self._run_state.get(target_run_id, {"retrieval": [], "tools": []})

        # Merge any manually pre-injected chunks (custom retriever pattern)
        if self._pending_chunks:
            state = dict(state)  # shallow copy — don't mutate the stored state
            state["retrieval"] = list(state["retrieval"]) + self._pending_chunks
            self._pending_chunks = []  # consume the buffer
        if self._pending_tools:
            state = dict(state)
            state["tools"] = list(state["tools"]) + self._pending_tools
            self._pending_tools = []
        
        # We only look at the first generation's messages for the score (typical RAG)
        if not messages or not messages[0]:
            return
            
        flat_messages = messages[0]
        
        # 1. Context Reconstruction (ContextReducer logic)
        raw_input = self._reconstruct_context(flat_messages, state)
        
        # Stash trace for future replay/export
        self.last_trace = raw_input
        
        # 2. Run Analysis
        result = inspect_context(raw_input, config=self.config)
        
        if self.debug:
            self._emit_debug(raw_input, result)
            
        # 3. Emit 3-line dopamine reaction
        self._emit_reaction(result)
        
        # 4. Enforce block mode if applicable
        if self.mode == "block" and result.score < self.min_score:
            raise ContextOpsScoreError(
                f"ContextOps blocked execution: Score {result.score} is below min_score {self.min_score}"
            )

    def _reconstruct_context(self, messages: List[BaseMessage], state: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        ContextReducer: Merges the temporal buffer with the flat messages into
        our structured format.
        """
        payload: Dict[str, Any] = {
            "system": "",
            "messages": [],
            "chunks": state["retrieval"],  # Attach the stashed retrieval context
            "tools": state["tools"],       # Attach the stashed tool context
        }
        
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            
            if isinstance(msg, SystemMessage):
                # If multiple system messages exist, concatenate them
                payload["system"] += content + "\n"
            elif isinstance(msg, ToolMessage):
                # Tool messages that are directly in the prompt
                payload["tools"].append({"content": content, "source": msg.name or "tool"})
            else:
                # Human, AI, etc.
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                payload["messages"].append({"role": role, "content": content})
                
        return payload

    # ── Architecture classification ──────────────────────────────────────────

    @staticmethod
    def _architecture_label(score: int, structure_penalty: float) -> str:
        """
        Classify the context architecture into a developer-facing state.

        This is the key insight from Phase 2: ContextOps is not just a token
        optimizer — it is a context architecture debugger.
        """
        if structure_penalty >= 10:
            return "ARCHITECTURE FAILURE"
        elif score < 60:
            return "CRITICAL"
        elif score < 75:
            return "DEGRADED"
        elif score < 90:
            return "NEEDS WORK"
        else:
            return "HEALTHY"

    @staticmethod
    def _mini_bar(penalty: float, max_penalty: float, width: int = 6) -> str:
        """Render a tiny block-character bar for the penalty axes row."""
        if max_penalty == 0:
            filled = 0
        else:
            filled = round((penalty / max_penalty) * width)
        filled = min(filled, width)
        return "\u2593" * filled + "\u2591" * (width - filled)

    def _emit_reaction(self, result: Any) -> None:
        """
        Render the high-impact, 5-layer developer reaction format.

        Layer 1  — Score + Architecture label  (the hook)
        Layer 2  — Diagnosis                   (what ContextOps actually found)
        Layer 3  — Cost at scale               (why the dev should care)
        Layer 4  — Penalty axes mini-bars       (which dimension is killing the score)
        Layer 5  — Fix                         (exactly what to do)
        """
        top_rec = result.recommendations[0] if result.recommendations else None
        sb = result.score_breakdown
        tb = result.token_breakdown
        score = result.score

        arch_label = self._architecture_label(score, sb.structure_penalty)

        # Score icon
        if score >= 90:
            score_icon = "\u2705"
        elif score >= 75:
            score_icon = "\u26a0\ufe0f"
        else:
            score_icon = "\u274c"

        # Cost strings
        cost_per_req = tb.estimated_cost_usd
        cost_per_1k = cost_per_req * 1000
        cost_str = f"${cost_per_req:.4f}/req  |  ${cost_per_1k:.2f}/1k requests"

        # Top issue label
        top_issue = top_rec.issue if top_rec else "No major issues detected"

        print("\n" + "\u2500" * 62)

        # Layer 1: Score + Architecture label
        print(f"{score_icon}  ContextOps Score: {score}/100  [{arch_label}]")

        # Layer 2: Diagnosis (the architecture insight)
        print(f"   Diagnosis: {top_issue}")

        # Layer 3: Cost at scale
        print(f"\n   Cost:    {cost_str}")
        if tb.wasted_tokens > 0:
            wasted_cost_per_1k = (tb.wasted_tokens / max(1, tb.total_tokens)) * cost_per_1k
            print(f"   Wasted:  ~{tb.wasted_tokens} tokens  |  ~${wasted_cost_per_1k:.2f}/1k requests wasted")

        # Layer 4: Penalty axis mini-bars
        r_bar = self._mini_bar(sb.redundancy_penalty, 30)
        d_bar = self._mini_bar(sb.density_penalty, 30)
        s_bar = self._mini_bar(sb.structure_penalty, 20)
        c_bar = self._mini_bar(sb.concentration_penalty, 20)
        print(
            f"\n   Axes:  "
            f"Redundancy {r_bar}  "
            f"Density {d_bar}  "
            f"Structure {s_bar}  "
            f"Concentration {c_bar}"
        )

        # Layer 5: Actionable fix
        if top_rec:
            print(f"\n   Fix:  {top_rec.fix}")

        print("\u2500" * 62 + "\n")

    def _emit_debug(self, raw_input: Dict[str, Any], result: Any) -> None:
        """Verbose truth-verification layer for ContextOps developers."""
        
        chunks = raw_input.get("chunks", [])
        tools = raw_input.get("tools", [])
        
        if len(chunks) > 0 and len(tools) > 0:
            capture_mode = "FULL"
        elif len(chunks) > 0 or len(tools) > 0:
            capture_mode = "PARTIAL"
        else:
            capture_mode = "MESSAGE-ONLY"
            
        print("\n" + "═" * 60)
        print(f"🔍 CONTEXTOPS DEBUG MODE (TRUTH LAYER) - {capture_mode}")
        print("═" * 60)
        
        # System prompt size
        sys_prompt = raw_input.get("system", "")
        print(f"System Prompt: {len(sys_prompt)} chars")
        
        # Retrieval count
        print(f"Captured Retriever Chunks: {len(chunks)}")
        
        # Tools count
        print(f"Captured Tool Outputs: {len(tools)}")
        
        # Token Breakdown Classification
        print("\nToken Classification Breakdown:")
        for ctx_type, tokens in result.token_breakdown.by_type.items():
            print(f"  - {ctx_type.upper()}: {tokens} tokens")
            
        print("═" * 60 + "\n")

    def export_trace(self, filepath: str) -> None:
        """
        Export the last reconstructed context to JSON for replay testing.
        This creates the foundation for the dataset / benchmark moat.
        """
        if not self.last_trace:
            raise ValueError("No trace available. The handler has not intercepted a chain yet.")
            
        import json
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.last_trace, f, indent=2)
        logger.info(f"ContextOps trace exported to {filepath}")
