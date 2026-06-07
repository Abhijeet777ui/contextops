"""
Context Normalizer.

Converts raw LLM inputs into the canonical ContextBundle format.
Supports:
  - OpenAI-style message lists
  - Raw dict lists with type/content
  - Single string inputs (treated as system prompt)
"""

from __future__ import annotations

from contextops.core.models import ContextBundle, ContextItem, ContextType


# Maps OpenAI message roles to our ContextType enum
_ROLE_MAP: dict[str, ContextType] = {
    "system": ContextType.SYSTEM,
    "user": ContextType.MESSAGE,
    "assistant": ContextType.MESSAGE,
    "tool": ContextType.TOOL,
    "function": ContextType.TOOL,
}


def normalize(raw_input: str | list[dict] | dict) -> ContextBundle:
    """
    Normalize any supported raw input into a ContextBundle.

    Args:
        raw_input: One of:
            - A plain string (treated as a system prompt)
            - A list of OpenAI-style message dicts
            - A dict with explicit 'messages', 'chunks', 'memory', 'system' keys

    Returns:
        A ContextBundle with all items normalized.

    Raises:
        ValueError: If the input format is not recognized.
    """
    if isinstance(raw_input, str):
        return _normalize_string(raw_input)
    elif isinstance(raw_input, list):
        return _normalize_message_list(raw_input)
    elif isinstance(raw_input, dict):
        # Unwrap benchmark-style "input" wrapper if present
        if "input" in raw_input and isinstance(raw_input["input"], dict):
            raw_input = raw_input["input"]
        return _normalize_structured_dict(raw_input)
    else:
        raise ValueError(
            f"Unsupported input type: {type(raw_input).__name__}. "
            "Expected str, list[dict], or dict."
        )


def _normalize_string(text: str) -> ContextBundle:
    """Treat a single string as a system prompt."""
    item = ContextItem(
        type=ContextType.SYSTEM,
        content=text,
        source="raw_string",
    )
    return ContextBundle(items=[item])


def _normalize_message_list(messages: list[dict]) -> ContextBundle:
    """
    Normalize an OpenAI-style message list.

    Each dict should have at least 'role' and 'content' keys.
    """
    items: list[ContextItem] = []

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"Message at index {i} is not a dict: {type(msg).__name__}")

        role = msg.get("role", "user")
        content = msg.get("content", "")
        context_type = _ROLE_MAP.get(role, ContextType.MESSAGE)

        # Extract source hint if available
        source = msg.get("name") or msg.get("source") or f"message_{i}"

        item = ContextItem(
            type=context_type,
            content=content if content else "",
            source=source,
            metadata={"role": role, "index": i},
        )
        items.append(item)

    return ContextBundle(items=items)


def _normalize_structured_dict(data: dict) -> ContextBundle:
    """
    Normalize a structured dict with explicit context sections.

    Expected keys (all optional):
        - system: str
        - messages: list[dict] with role/content
        - chunks / retrieval: list[str | dict]
        - memory: list[str | dict]
        - tools: list[str | dict]
    """
    items: list[ContextItem] = []

    # System prompt
    if "system" in data:
        items.append(ContextItem(
            type=ContextType.SYSTEM,
            content=data["system"],
            source="system_prompt",
        ))

    # Chat messages
    for i, msg in enumerate(data.get("messages", [])):
        if isinstance(msg, str):
            items.append(ContextItem(
                type=ContextType.MESSAGE,
                content=msg,
                source=f"message_{i}",
            ))
        elif isinstance(msg, dict):
            role = msg.get("role", "user")
            items.append(ContextItem(
                type=_ROLE_MAP.get(role, ContextType.MESSAGE),
                content=msg.get("content", ""),
                source=msg.get("source", f"message_{i}"),
                metadata={"role": role, "index": i},
            ))

    # Retrieval chunks (key can be 'chunks' or 'retrieval')
    chunks = data.get("chunks", data.get("retrieval", []))
    for i, chunk in enumerate(chunks):
        if isinstance(chunk, str):
            items.append(ContextItem(
                type=ContextType.RETRIEVAL,
                content=chunk,
                source=f"chunk_{i}",
            ))
        elif isinstance(chunk, dict):
            items.append(ContextItem(
                type=ContextType.RETRIEVAL,
                content=chunk.get("content", ""),
                source=chunk.get("source", f"chunk_{i}"),
                metadata={k: v for k, v in chunk.items() if k not in ("content", "source")},
            ))

    # Memory
    for i, mem in enumerate(data.get("memory", [])):
        if isinstance(mem, str):
            items.append(ContextItem(
                type=ContextType.MEMORY,
                content=mem,
                source=f"memory_{i}",
            ))
        elif isinstance(mem, dict):
            items.append(ContextItem(
                type=ContextType.MEMORY,
                content=mem.get("content", ""),
                source=mem.get("source", f"memory_{i}"),
                metadata={k: v for k, v in mem.items() if k not in ("content", "source")},
            ))

    # Tool outputs
    for i, tool in enumerate(data.get("tools", [])):
        if isinstance(tool, str):
            items.append(ContextItem(
                type=ContextType.TOOL,
                content=tool,
                source=f"tool_{i}",
            ))
        elif isinstance(tool, dict):
            # Tool outputs may use 'content' or 'output' as the text key
            tool_content = tool.get("content", "") or tool.get("output", "")
            items.append(ContextItem(
                type=ContextType.TOOL,
                content=tool_content,
                source=tool.get("source", tool.get("name", f"tool_{i}")),
                metadata={k: v for k, v in tool.items() if k not in ("content", "source", "output", "name")},
            ))

    return ContextBundle(items=items)
