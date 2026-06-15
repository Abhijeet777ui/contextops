"""Generate chaos benchmark JSON files for Phase 2 stress validation."""
import json
from pathlib import Path


def create_massive_rag_dump():
    """150 retrieval chunks of ~150 tokens each = 20,000+ tokens."""
    chunks = []
    for i in range(150):
        content = f"Chunk {i}: " + " ".join([f"word{j}" for j in range(150)])
        chunks.append({"content": content, "source": f"doc_{i}.md"})

    return {
        "name": "chaos_massive_rag_dump",
        "expected_contract": {
            "max_score": 80,
            "must_flag": ["Retrieval dominance"],
        },
        "input": {
            "system": "You are an AI.",
            "messages": [{"role": "user", "content": "Query"}],
            "chunks": chunks,
            "memory": [],
            "tools": [],
        },
    }


def create_agent_trace_loop():
    """Agent hitting the same tool error 15 times."""
    tools = []
    for _ in range(15):
        tools.append(
            {
                "name": "execute_sql",
                "output": "Error: column 'user_id' does not exist in table 'orders'.",
            }
        )

    return {
        "name": "chaos_agent_trace_loop",
        "expected_contract": {
            "must_flag": ["Redundant context"],
        },
        "input": {
            "system": "You are an agent.",
            "messages": [{"role": "user", "content": "Find orders"}],
            "chunks": [],
            "memory": [],
            "tools": tools,
        },
    }


def create_system_prompt_explosion():
    """System prompt duplicated aggressively."""
    system = "DO NOT HARM. " * 500

    return {
        "name": "chaos_system_prompt_explosion",
        "expected_contract": {
            "must_flag": ["System prompt bloat"],
        },
        "input": {
            "system": system,
            "messages": [{"role": "user", "content": "Hello"}],
            "chunks": [{"content": "Here is some context.", "source": "info.txt"}],
            "memory": [],
            "tools": [],
        },
    }


def create_empty_and_corrupt():
    """Empty items and missing fields."""
    return {
        "name": "chaos_empty_and_corrupt",
        "expected_contract": {
            "min_score": 60,
            "must_flag": [],
        },
        "input": {
            "system": "",
            "messages": [],
            "chunks": [
                {"content": "", "source": "empty.md"},
                {"content": "    ", "source": "spaces.md"},
            ],
            "memory": [],
            "tools": [],
        },
    }


def create_micro_context():
    """Exactly 1 short message (~5 tokens)."""
    return {
        "name": "chaos_micro_context",
        "expected_contract": {
            "min_score": 90,
            "must_flag": [],
        },
        "input": {
            "system": "Sys",
            "messages": [{"role": "user", "content": "Hi"}],
            "chunks": [],
            "memory": [],
            "tools": [],
        },
    }


def generate_all():
    out_dir = Path("benchmarks/chaos")
    out_dir.mkdir(parents=True, exist_ok=True)

    generators = [
        create_massive_rag_dump,
        create_agent_trace_loop,
        create_system_prompt_explosion,
        create_empty_and_corrupt,
        create_micro_context,
    ]

    for gen in generators:
        data = gen()
        file_path = out_dir / f"{data['name']}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Generated {file_path}")


if __name__ == "__main__":
    generate_all()
