"""ema-agent: reusable RAG router-workflow engine.

Public API. Only the light, side-effect-free modules are re-exported here.
`store_index_faq` is intentionally NOT imported: it loads an Excel dataset and
builds the FAISS index at import time, so importing it must stay an explicit,
opt-in choice (`from ema_agent.store_index_faq import retrieve_documents`).
"""

from .indexing import (
    configure_embeddings,
    documents_from_excel,
    documents_from_json,
    load_or_build_index,
    new_faiss_store,
)
from .retriever import HybridRetriever, rerank

from ema_agent.agent_starter import Topic, append_session_to_history
from ema_agent.workflow import (
    RouterWorkflow,
    RetrieverFn,
    RouteDecision,
    AnswerEvaluation,
    DecomposedQueries,
    StreamEvent,
    ChatEvent,
    ClarifyEvent,
    RetrieveEvent,
    ComplexEvent,
    SubQueryEvent,
    RetrievalResultEvent,
    AnswerEvent,
)

__version__ = "0.1.0"

__all__ = [
    "Topic",
    "append_session_to_history",
    "RouterWorkflow",
    "RetrieverFn",
    "RouteDecision",
    "AnswerEvaluation",
    "DecomposedQueries",
    "StreamEvent",
    "ChatEvent",
    "ClarifyEvent",
    "RetrieveEvent",
    "ComplexEvent",
    "SubQueryEvent",
    "RetrievalResultEvent",
    "AnswerEvent",
    # indexing
    "configure_embeddings",
    "new_faiss_store",
    "load_or_build_index",
    "documents_from_excel",
    "documents_from_json",
    # retrieval
    "HybridRetriever",
    "rerank",
]
