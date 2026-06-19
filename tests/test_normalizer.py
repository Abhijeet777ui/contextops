"""
Tests for the Context Normalizer.

The normalizer is the single entry point that converts raw user input into
ContextBundle. If it breaks, every downstream analyzer produces garbage.

Coverage:
  - String input path
  - OpenAI message list path
  - Structured dict path (with all sub-keys)
  - Benchmark-style "input" wrapper unwrapping
  - Missing / None / empty fields
  - Malformed types and bad data
  - Unicode and special characters
  - Source extraction fallbacks
  - "retrieval" vs "chunks" key aliasing
  - Tool output "content" vs "output" key aliasing
"""

import pytest

from contextops.core.normalizer import (
    normalize,
    _normalize_string,
    _normalize_message_list,
    _normalize_structured_dict,
)
from contextops.core.models import ContextBundle, ContextItem, ContextType


# ═══════════════════════════════════════════════════════════════════════════
# STRING INPUT PATH
# ═══════════════════════════════════════════════════════════════════════════

class TestStringInput:
    """normalize("some string") must become a single SYSTEM item."""

    def test_basic_string(self):
        bundle = normalize("You are a helpful assistant.")
        assert len(bundle.items) == 1
        assert bundle.items[0].type == ContextType.SYSTEM
        assert bundle.items[0].content == "You are a helpful assistant."
        assert bundle.items[0].source == "raw_string"

    def test_empty_string(self):
        """Empty string is valid — normalizer must not crash."""
        bundle = normalize("")
        assert len(bundle.items) == 1
        assert bundle.items[0].content == ""
        assert bundle.items[0].type == ContextType.SYSTEM

    def test_whitespace_only_string(self):
        bundle = normalize("   \n\t  ")
        assert len(bundle.items) == 1
        assert bundle.items[0].content == "   \n\t  "

    def test_unicode_string(self):
        """Unicode must pass through without corruption."""
        text = "あなたは親切なアシスタントです。Ça marche? Ñoño — 🎯"
        bundle = normalize(text)
        assert bundle.items[0].content == text

    def test_very_long_string(self):
        """10K character string must not crash or truncate."""
        text = "word " * 10000
        bundle = normalize(text)
        assert len(bundle.items[0].content) == len(text)

    def test_string_with_special_chars(self):
        """SQL, HTML, JSON-like strings must not be misinterpreted."""
        text = '<script>alert("xss")</script> SELECT * FROM users; {"key": "val"}'
        bundle = normalize(text)
        assert bundle.items[0].content == text


# ═══════════════════════════════════════════════════════════════════════════
# MESSAGE LIST INPUT PATH
# ═══════════════════════════════════════════════════════════════════════════

class TestMessageListInput:
    """normalize([{role, content}, ...]) — OpenAI-style message lists."""

    def test_basic_message_list(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        bundle = normalize(messages)
        assert len(bundle.items) == 3
        assert bundle.items[0].type == ContextType.SYSTEM
        assert bundle.items[1].type == ContextType.MESSAGE
        assert bundle.items[2].type == ContextType.MESSAGE

    def test_tool_and_function_roles(self):
        """Both 'tool' and 'function' roles must map to ContextType.TOOL."""
        messages = [
            {"role": "tool", "content": "result from tool"},
            {"role": "function", "content": "result from function"},
        ]
        bundle = normalize(messages)
        assert bundle.items[0].type == ContextType.TOOL
        assert bundle.items[1].type == ContextType.TOOL

    def test_unknown_role_defaults_to_message(self):
        """An unrecognized role must default to MESSAGE, not crash."""
        messages = [{"role": "banana", "content": "some text"}]
        bundle = normalize(messages)
        assert bundle.items[0].type == ContextType.MESSAGE

    def test_missing_role_defaults_to_user(self):
        """Dict without 'role' key must default to 'user' → MESSAGE."""
        messages = [{"content": "some text"}]
        bundle = normalize(messages)
        assert bundle.items[0].type == ContextType.MESSAGE
        assert bundle.items[0].metadata["role"] == "user"

    def test_missing_content_defaults_to_empty(self):
        """Dict without 'content' key must default to empty string."""
        messages = [{"role": "user"}]
        bundle = normalize(messages)
        assert bundle.items[0].content == ""

    def test_none_content_treated_as_empty(self):
        """content=None must become empty string, not crash."""
        messages = [{"role": "assistant", "content": None}]
        bundle = normalize(messages)
        assert bundle.items[0].content == ""

    def test_empty_list(self):
        """Empty message list must produce empty bundle."""
        bundle = normalize([])
        assert len(bundle.items) == 0

    def test_source_from_name_field(self):
        """'name' field in message dict must be used as source."""
        messages = [{"role": "tool", "content": "data", "name": "search_api"}]
        bundle = normalize(messages)
        assert bundle.items[0].source == "search_api"

    def test_source_from_source_field(self):
        """'source' field must work as source fallback."""
        messages = [{"role": "user", "content": "hi", "source": "chat_input"}]
        bundle = normalize(messages)
        assert bundle.items[0].source == "chat_input"

    def test_source_fallback_to_index(self):
        """If no name/source, default to 'message_{i}'."""
        messages = [{"role": "user", "content": "hello"}]
        bundle = normalize(messages)
        assert bundle.items[0].source == "message_0"

    def test_non_dict_in_list_raises(self):
        """A non-dict element in the list must raise ValueError."""
        with pytest.raises(ValueError, match="not a dict"):
            normalize(["just a string", "another string"])

    def test_mixed_valid_messages(self):
        """Large diverse message list must produce correct count and types."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "tool", "content": "tool_out", "name": "calc"},
        ]
        bundle = normalize(messages)
        assert len(bundle.items) == 6
        types = [item.type for item in bundle.items]
        assert types.count(ContextType.SYSTEM) == 1
        assert types.count(ContextType.MESSAGE) == 4
        assert types.count(ContextType.TOOL) == 1

    def test_metadata_preserved(self):
        """Each message must get role and index in metadata."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
        ]
        bundle = normalize(messages)
        assert bundle.items[0].metadata["role"] == "user"
        assert bundle.items[0].metadata["index"] == 0
        assert bundle.items[1].metadata["index"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# STRUCTURED DICT INPUT PATH
# ═══════════════════════════════════════════════════════════════════════════

class TestStructuredDictInput:
    """normalize({system, messages, chunks, memory, tools}) — the structured API."""

    def test_system_prompt_only(self):
        bundle = normalize({"system": "You are helpful."})
        assert len(bundle.items) == 1
        assert bundle.items[0].type == ContextType.SYSTEM
        assert bundle.items[0].source == "system_prompt"

    def test_empty_system_prompt(self):
        """Empty system string must still produce a SYSTEM item."""
        bundle = normalize({"system": ""})
        assert len(bundle.items) == 1
        assert bundle.items[0].content == ""

    def test_chunks_as_strings(self):
        """chunks: ["str1", "str2"] must become RETRIEVAL items."""
        bundle = normalize({"chunks": ["chunk one", "chunk two"]})
        assert len(bundle.items) == 2
        assert all(item.type == ContextType.RETRIEVAL for item in bundle.items)
        assert bundle.items[0].source == "chunk_0"
        assert bundle.items[1].source == "chunk_1"

    def test_chunks_as_dicts(self):
        """chunks: [{content, source}] must extract content and source."""
        bundle = normalize({
            "chunks": [
                {"content": "data here", "source": "doc.pdf"},
                {"content": "more data", "source": "wiki.md"},
            ]
        })
        assert len(bundle.items) == 2
        assert bundle.items[0].content == "data here"
        assert bundle.items[0].source == "doc.pdf"

    def test_chunks_dict_extra_metadata(self):
        """Extra keys in chunk dicts must be captured in metadata."""
        bundle = normalize({
            "chunks": [
                {"content": "x", "source": "y", "score": 0.95, "page": 3}
            ]
        })
        assert bundle.items[0].metadata["score"] == 0.95
        assert bundle.items[0].metadata["page"] == 3
        # content and source must NOT be in metadata
        assert "content" not in bundle.items[0].metadata
        assert "source" not in bundle.items[0].metadata

    def test_retrieval_key_alias(self):
        """'retrieval' key must work as an alias for 'chunks'."""
        bundle = normalize({"retrieval": ["data from retrieval key"]})
        assert len(bundle.items) == 1
        assert bundle.items[0].type == ContextType.RETRIEVAL

    def test_chunks_takes_priority_over_retrieval(self):
        """If both 'chunks' and 'retrieval' are present, 'chunks' wins."""
        bundle = normalize({
            "chunks": ["from chunks"],
            "retrieval": ["from retrieval"],
        })
        assert len(bundle.items) == 1
        assert bundle.items[0].content == "from chunks"

    def test_messages_as_strings(self):
        """messages: ["str"] must become MESSAGE items."""
        bundle = normalize({"messages": ["Hello there"]})
        assert len(bundle.items) == 1
        assert bundle.items[0].type == ContextType.MESSAGE

    def test_messages_as_dicts(self):
        """messages: [{role, content}] must map roles correctly."""
        bundle = normalize({
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        })
        assert len(bundle.items) == 2
        assert all(item.type == ContextType.MESSAGE for item in bundle.items)

    def test_memory_as_strings(self):
        bundle = normalize({"memory": ["user likes cats", "previous session summary"]})
        assert len(bundle.items) == 2
        assert all(item.type == ContextType.MEMORY for item in bundle.items)
        assert bundle.items[0].source == "memory_0"

    def test_memory_as_dicts(self):
        bundle = normalize({
            "memory": [{"content": "mem data", "source": "session_1"}]
        })
        assert bundle.items[0].type == ContextType.MEMORY
        assert bundle.items[0].source == "session_1"

    def test_tools_as_strings(self):
        bundle = normalize({"tools": ["tool output text"]})
        assert len(bundle.items) == 1
        assert bundle.items[0].type == ContextType.TOOL

    def test_tools_content_key(self):
        """Tool dicts with 'content' key must extract correctly."""
        bundle = normalize({
            "tools": [{"content": "search results", "name": "web_search"}]
        })
        assert bundle.items[0].content == "search results"
        assert bundle.items[0].source == "web_search"

    def test_tools_output_key_alias(self):
        """Tool dicts with 'output' key (instead of 'content') must work."""
        bundle = normalize({
            "tools": [{"output": "calc result", "name": "calculator"}]
        })
        assert bundle.items[0].content == "calc result"

    def test_tools_prefer_content_over_output(self):
        """If both 'content' and 'output' exist, 'content' wins."""
        bundle = normalize({
            "tools": [{"content": "from content", "output": "from output"}]
        })
        assert bundle.items[0].content == "from content"

    def test_full_structured_dict(self):
        """All sections populated — verify correct count and type ordering."""
        bundle = normalize({
            "system": "system prompt",
            "messages": [
                {"role": "user", "content": "question"},
            ],
            "chunks": [
                {"content": "chunk data", "source": "doc.md"},
            ],
            "memory": ["old memory"],
            "tools": [{"output": "tool result", "name": "api"}],
        })
        # system(1) + messages(1) + chunks(1) + memory(1) + tools(1) = 5
        assert len(bundle.items) == 5
        types = [item.type for item in bundle.items]
        assert ContextType.SYSTEM in types
        assert ContextType.MESSAGE in types
        assert ContextType.RETRIEVAL in types
        assert ContextType.MEMORY in types
        assert ContextType.TOOL in types

    def test_completely_empty_dict(self):
        """An empty dict {} must produce an empty bundle, not crash."""
        bundle = normalize({})
        assert len(bundle.items) == 0

    def test_dict_with_all_empty_sections(self):
        """All sections present but empty must produce only the system item."""
        bundle = normalize({
            "system": "",
            "messages": [],
            "chunks": [],
            "memory": [],
            "tools": [],
        })
        # Only system item (even with empty content)
        assert len(bundle.items) == 1
        assert bundle.items[0].type == ContextType.SYSTEM

    def test_chunk_with_missing_content(self):
        """Chunk dict without 'content' key must default to empty string."""
        bundle = normalize({"chunks": [{"source": "mystery.md"}]})
        assert bundle.items[0].content == ""

    def test_chunk_with_missing_source(self):
        """Chunk dict without 'source' must get auto-generated source."""
        bundle = normalize({"chunks": [{"content": "data"}]})
        assert bundle.items[0].source == "chunk_0"


# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARK-STYLE WRAPPER UNWRAPPING
# ═══════════════════════════════════════════════════════════════════════════

class TestBenchmarkWrapper:
    """Dicts with an outer "input" key must be auto-unwrapped."""

    def test_input_wrapper_unwrapped(self):
        """{"input": {system, chunks}} must unwrap to the inner dict."""
        raw = {
            "name": "benchmark_case",
            "expected_contract": {"min_score": 90},
            "input": {
                "system": "sys prompt",
                "chunks": [{"content": "data", "source": "doc.md"}],
            }
        }
        bundle = normalize(raw)
        # Should process the inner "input" dict, ignoring name/expected_contract
        types = [item.type for item in bundle.items]
        assert ContextType.SYSTEM in types
        assert ContextType.RETRIEVAL in types

    def test_input_wrapper_with_non_dict_input_ignored(self):
        """If "input" is a string (not dict), don't unwrap — treat as normal key."""
        raw = {"input": "not a dict", "system": "real system"}
        bundle = normalize(raw)
        # Should process as structured dict with "system" key
        assert any(item.type == ContextType.SYSTEM for item in bundle.items)


# ═══════════════════════════════════════════════════════════════════════════
# UNSUPPORTED / INVALID INPUT TYPES
# ═══════════════════════════════════════════════════════════════════════════

class TestInvalidInputs:
    """normalize() must raise clear errors for unsupported types."""

    def test_int_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unsupported input type"):
            normalize(42)

    def test_float_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unsupported input type"):
            normalize(3.14)

    def test_none_raises_valueerror(self):
        with pytest.raises((ValueError, TypeError)):
            normalize(None)

    def test_bool_raises_valueerror(self):
        with pytest.raises((ValueError, TypeError)):
            normalize(True)

    def test_tuple_raises(self):
        with pytest.raises((ValueError, TypeError)):
            normalize(("system", "hello"))

    def test_set_raises(self):
        with pytest.raises((ValueError, TypeError)):
            normalize({"hello", "world"})


# ═══════════════════════════════════════════════════════════════════════════
# UNICODE / ENCODING EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════

class TestUnicodeEdgeCases:
    """Ensure unicode passes through all normalization paths correctly."""

    def test_chinese_characters_in_chunks(self):
        bundle = normalize({"chunks": ["这是一个中文文档的检索结果"]})
        assert bundle.items[0].content == "这是一个中文文档的检索结果"

    def test_emoji_in_messages(self):
        msgs = [{"role": "user", "content": "Rate this: 🌟🌟🌟🌟🌟"}]
        bundle = normalize(msgs)
        assert "🌟" in bundle.items[0].content

    def test_arabic_rtl_text(self):
        bundle = normalize("مرحبا بالعالم")
        assert bundle.items[0].content == "مرحبا بالعالم"

    def test_mixed_scripts(self):
        text = "Hello こんにちは Привет مرحبا 🎯"
        bundle = normalize(text)
        assert bundle.items[0].content == text

    def test_null_bytes_in_content(self):
        """Null bytes must not crash the normalizer."""
        bundle = normalize({"chunks": ["data\x00with\x00nulls"]})
        assert "\x00" in bundle.items[0].content

    def test_newlines_and_tabs_preserved(self):
        text = "line1\nline2\tindented\r\nwindows"
        bundle = normalize(text)
        assert bundle.items[0].content == text
