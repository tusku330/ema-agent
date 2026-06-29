## Project Overview

ema-agent is the reusable **core agent component** for the e-mongolia.mn / e-business.mn
support chatbot, published and consumed as a GitHub dependency (pyproject package).
It provides a generic RAG router-workflow engine (LlamaIndex Workflows) with hybrid
FAISS + BM25 retrieval and optional cross-encoder reranking.

It does NOT contain the backend entry point or the front-end. Those live in separate repos:
- `ema-chatbot` — backend entry point that consumes this package
- `ema-chatbot-front` — custom Chainlit front-end

## Tech Stack

- Framework: LlamaIndex (Workflows)
- Language: Python (requires-python >= 3.10)
- Retrieval: FAISS (vector) + BM25 (lexical), bge-m3 embeddings, bge-reranker-v2-m3 rerank
- LLM: OpenAI (gpt-4o-mini default); Gemini/Ollama scaffolded but commented out
- Data validation: pydantic
- Packaging: setuptools, pyproject.toml

## Coding Conventions

- snake_case for functions, variables, and modules; PascalCase for classes
- Type-hint public functions and class signatures
- Prefer `async def` for I/O-bound code (retrieval, LLM calls)
- Use module-level docstrings explaining intent (see existing modules for the style)
- Prefer early returns over nested conditionals

## Never Do This

- Never install a new package without asking me first
- Never rewrite a module I did not ask you to touch
- Never add placeholder comments like "# TODO: implement this"
- Never use emojis in code comments
- Never wrap everything in a try/except without telling me
- Never suggest switching the framework, vector store, or LLM provider

## File Structure

src/ema_agent/              # The installable package (src layout)
  __init__.py              # The ONLY public surface: curated re-exports + __all__
  py.typed                 # PEP 561 marker — ships type hints to consumers
  workflow.py              # RouterWorkflow, events, pydantic schemas (control flow)
  prompts.py               # Prompt templates + canned strings (override here)
  agent_starter.py         # Topic enum (taxonomy) + append_session_to_history
  indexing.py              # embedding config + FAISS + document builders
  retrieval/               # hybrid retrieval subpackage
    hybrid.py              # HybridRetriever (vector + BM25 fusion)
    rerank.py              # cross-encoder rerank (lazy-loaded; [rerank] extra)
    mongolian.py           # MN stemmer + stopwords ("language pack")
examples/                  # NOT shipped — runnable demo + project-specific glue
  faq_cli.py               # interactive CLI demo
  faq_dataset.py           # Excel/Topic glue for the e-Mongolia dataset
  llm_config.py            # OpenAI/Gemini/Ollama config (app concern, not library)
tests/                     # pytest tests (run against the installed package)

Library code stays generic in src/ema_agent/. Anything tied to THIS project's
dataset, taxonomy values, or LLM/provider config belongs in examples/ (or in the
consuming ema-chatbot repo), never in the package.

Install for local dev: `pip install -e .[rerank,examples,dev]`.
Do not create new top-level folders without asking.

## Current Goals

bet