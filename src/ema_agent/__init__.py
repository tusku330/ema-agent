"""ema-agent: reusable RAG router-workflow engine.

Public API. Only the light, side-effect-free building blocks are re-exported here.
Project-specific dataset glue (Excel loading + import-time FAISS build) is NOT part
of the library — see ``examples/faq_dataset.py`` for how a consuming app wires its
own data into ``indexing`` + ``retrieval``.
"""

from .indexing import (
    configure_embeddings,
    documents_from_excel,
    documents_from_json,
    load_or_build_index,
    new_faiss_store,
)
from .retrieval import HybridRetriever, rerank

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
