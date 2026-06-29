"""Reusable hybrid (vector + BM25) retriever with optional cross-encoder rerank.

Generic over a node list + a vector index, so it can be reused across projects
unchanged. The dataset loading, taxonomy validation, and answer-map formatting
that are project-specific stay in the app layer (see ``examples/faq_dataset.py``).

Typical use::

    from ema_agent.indexing import configure_embeddings, load_or_build_index
    from ema_agent.retrieval import HybridRetriever

    dim = configure_embeddings()
    index = load_or_build_index(nodes, "./storage/my_index", dim)
    retriever = HybridRetriever(nodes, index)          # filter_key defaults to "topic"
    hits = await retriever.aretrieve("my query", filter_value="tax", use_reranker=True)
"""

from __future__ import annotations

from typing import Callable, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.llms import LLM, MockLLM
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.retrievers.bm25 import BM25Retriever

from ema_agent.retrieval.mongolian import MN_STOPWORDS, mn_stemmer
from ema_agent.retrieval.rerank import DEFAULT_RERANKER_MODEL, rerank


class HybridRetriever:
    """Hybrid vector + BM25 retriever with optional metadata pre/post-filter.

    Generic over the node list and vector index. ``filter_key`` names the
    metadata field used to scope retrieval (default ``"topic"``); pass
    ``filter_value=None`` at query time to search everything.
    """

    def __init__(
        self,
        nodes: list[TextNode],
        vector_index: VectorStoreIndex,
        *,
        filter_key: str = "topic",
        stemmer: Callable[[list[str]], list[str]] = mn_stemmer,
        stopwords: Optional[list[str]] = None,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        llm: Optional[LLM] = None,
    ) -> None:
        self._nodes = nodes
        self._vector_index = vector_index
        self._filter_key = filter_key
        self._stemmer = stemmer
        self._stopwords = MN_STOPWORDS if stopwords is None else stopwords
        self._reranker_model = reranker_model
        # num_queries=1 means QueryFusionRetriever never actually calls an LLM,
        # but its constructor still resolves Settings.llm (default: OpenAI) unless
        # one is passed. MockLLM keeps pure vector+BM25 fusion free of any API key
        # — pass a real LLM only if you raise num_queries for query expansion.
        self._llm = llm if llm is not None else MockLLM()

    def build_retriever(
        self, filter_value: Optional[str], similarity_top_k: int
    ) -> QueryFusionRetriever:
        """Construct a fused BM25 + vector retriever, pre-filtering BM25 by metadata."""
        # BM25: pre-filter nodes by ``filter_key`` so keyword scores stay in-domain.
        nodes = self._nodes
        if filter_value:
            nodes = [
                n for n in self._nodes
                if n.metadata.get(self._filter_key) == filter_value
            ]

        # Vector: FAISS has no metadata filter support — use wider top_k and
        # post-filter after fusion.
        vector = self._vector_index.as_retriever(similarity_top_k=similarity_top_k * 3)

        # If the filter matches no nodes, BM25 has an empty corpus (from_defaults
        # would raise). Fall back to vector-only — the post-filter still applies.
        if not nodes:
            return QueryFusionRetriever(
                retrievers=[vector],
                similarity_top_k=similarity_top_k * 3,
                num_queries=1,
                mode="reciprocal_rerank",
                use_async=True,
                llm=self._llm,
            )

        # bm25s raises when k > corpus size; clamp to the filtered node count.
        bm25 = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=min(similarity_top_k, len(nodes)),
            stemmer=self._stemmer,
            language=self._stopwords,
        )

        return QueryFusionRetriever(
            retrievers=[vector, bm25],
            similarity_top_k=similarity_top_k * 3,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=True,
            llm=self._llm,
        )

    async def aretrieve(
        self,
        query: str,
        filter_value: Optional[str] = None,
        use_reranker: bool = False,
        top_n: int = 4,
    ) -> list[NodeWithScore]:
        """Retrieve ``top_n`` nodes for ``query``, scoped to ``filter_value``.

        Set ``use_reranker=True`` to re-score the fused candidates with the
        cross-encoder before truncating to ``top_n``. When reranking, the fusion
        candidate pool is widened so the cross-encoder has more to choose from
        (rerank precision improves with recall).
        """
        # Wider candidate pool when reranking — the cross-encoder reorders it.
        base_k = 8 if use_reranker else 4
        retriever = self.build_retriever(filter_value, similarity_top_k=base_k)
        raw_nodes = await retriever.aretrieve(query)

        # Post-filter: drop vector-side results outside the filter.
        if filter_value:
            raw_nodes = [
                n for n in raw_nodes
                if n.node.metadata.get(self._filter_key) == filter_value
            ]

        if use_reranker:
            raw_nodes = await rerank(query, raw_nodes, top_n, self._reranker_model)
        else:
            raw_nodes = raw_nodes[:top_n]

        return raw_nodes
