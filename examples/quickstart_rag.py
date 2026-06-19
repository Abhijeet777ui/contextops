"""
ContextOps Quickstart — LangChain Integration Demo
===================================================

This is the canonical 30-second demo. It shows ContextOps detecting real
context architecture problems in a typical RAG pipeline.

Requirements:
    pip install contextops langchain-core

No API keys needed. Uses FakeListChatModel as the LLM.

Run:
    python examples/quickstart_rag.py
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

from contextops import ContextOps, ContextOpsCallbackHandler


# ── Fake retriever that injects redundant chunks ─────────────────────────────

def make_retriever(handler: ContextOpsCallbackHandler):
    """
    Returns a RunnableLambda that simulates a poorly-tuned RAG retriever.

    Injects chunks directly into the ContextOps handler so they are
    captured and scored — this is the correct pattern for manual retriever
    wrapping when not using LangChain's built-in retriever interface.
    """
    from langchain_core.runnables import RunnableLambda

    def retrieve(input_data):
        docs = [
            Document(
                page_content="Refund Policy: Customers may return items within 30 days of purchase for a full refund.",
                metadata={"source": "policy/refund.md"},
            ),
            Document(
                page_content="Refund Policy: Items may be returned within 30 days of purchase. Full refund guaranteed.",
                metadata={"source": "policy/refund.md"},  # Near-duplicate!
            ),
            Document(
                page_content="Our company was founded in 2010 and operates globally across 40 countries.",
                metadata={"source": "about/company.md"},  # Noise — unrelated to query
            ),
        ]

        # ✅ Clean public API — inject directly into the handler's pending buffer
        handler.inject_chunks(docs)

        return docs

    return RunnableLambda(retrieve)



# ── Build chain (deferred so we can pass the handler to the retriever) ────────

SYSTEM_PROMPT = (
    "You are a customer service assistant. "
    + "Always be polite. Always follow the rules. Always check the policy. " * 40
    + "\n\nRetrieved Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

llm = FakeListChatModel(responses=["Based on our policy, you can return items within 30 days."])


def build_chain(handler: ContextOpsCallbackHandler):
    """Build the chain with the handler pre-wired into the retriever."""
    retriever = make_retriever(handler)
    return (
        RunnablePassthrough.assign(context=retriever)
        | prompt
        | llm
    )


# ── Run ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  ContextOps — Quickstart RAG Demo")
    print("  Two lines to attach. Zero configuration. Full visibility.")
    print("=" * 62)
    print()
    print("  Simulated issues in this pipeline:")
    print("  - Bloated system prompt (repeated rules, 600+ tokens)")
    print("  - Near-duplicate retrieval chunks from the same source")
    print("  - Irrelevant noise chunk injected by the retriever")
    print()
    print("  Watch ContextOps catch all three automatically:")
    print()

    # ── THE INTEGRATION ───────────────────────────────────────────────
    handler = ContextOps.auto(debug=True)       # Line 1: create handler
    chain = build_chain(handler)                # pass to retriever
    chain_with_ops = chain.with_config(         # Line 2: attach to chain
        {"callbacks": [handler]}
    )
    # ─────────────────────────────────────────────────────────────────

    chat_history = [
        HumanMessage(content="Hi, I have a question about returns."),
        AIMessage(content="Of course! I'd be happy to help with that."),
    ]

    chain_with_ops.invoke({
        "question": "Can I return a product I bought 3 weeks ago?",
        "chat_history": chat_history,
    })

    print()
    print("=" * 62)
    print("  That's it. Two lines. Add ContextOps to any LangChain chain.")
    print()
    print("  Next steps:")
    print("  - Use mode='block' to enforce a minimum score in CI")
    print("  - Use ContextOps.auto(debug=True) to inspect captures")
    print("  - Run: contextops check context.json --min-score 75")
    print("=" * 62)
    print()


if __name__ == "__main__":
    main()
