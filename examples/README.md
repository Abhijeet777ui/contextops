# ContextOps Examples

This directory contains standalone, zero-setup examples demonstrating how to integrate ContextOps into your AI applications.

## 🚀 The 30-Second Quickstart

If you only run one file, run this:

```bash
python examples/quickstart_rag.py
```

This is the canonical integration demo. It uses `FakeListChatModel` (no API keys required) to simulate a messy RAG pipeline with three real-world architecture failures:
1. A bloated, repetitive system prompt
2. Duplicate context chunks returned by the retriever
3. Irrelevant noise injected into the context

It demonstrates the explicit, two-line `ContextOps.auto()` integration pattern and shows exactly what ContextOps outputs to the terminal before the LLM executes.

## Pattern Reference

If you are looking for specific integration patterns, `quickstart_rag.py` demonstrates:

- **Canonical Injection:** `chain.with_config({"callbacks": [ContextOps.auto()]})`
- **Custom Retrievers:** How to use `handler.inject_chunks(docs)` when you aren't using standard LangChain retrievers.
- **Debug Mode:** `ContextOps.auto(debug=True)` to print the internal truth layer and see exactly what tokens ContextOps captured.
