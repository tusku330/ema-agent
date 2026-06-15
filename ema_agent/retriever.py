"""Reusable hybrid (vector + BM25) retriever with optional cross-encoder rerank.

Ported from emon-chatbot's ``store_index_faq.py``. The dataset loading, Topic-enum
validation, and answer-map formatting that were project-specific stay in the app
layer (e.g. ``store_index_faq.py``); this module is generic over a node list + a
vector index, so it can be reused across projects unchanged.

Typical use::

    from ema_agent.indexing import configure_embeddings, load_or_build_index
    from ema_agent.retriever import HybridRetriever

    dim = configure_embeddings()
    index = load_or_build_index(nodes, "./storage/my_index", dim)
    retriever = HybridRetriever(nodes, index)          # filter_key defaults to "topic"
    hits = await retriever.aretrieve("my query", filter_value="tax", use_reranker=True)
"""

from __future__ import annotations

import asyncio
from typing import Callable, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.llms import LLM, MockLLM
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.retrievers.bm25 import BM25Retriever


# --- Mongolian-aware BM25 tokenization --------------------------------------
# bm25s's default pipeline tokenizes Cyrillic fine (the `(?u)` token pattern
# treats Cyrillic as word chars) but then applies the *English* Snowball
# stemmer, which is a no-op on Mongolian. Mongolian is agglutinative: татвар,
# татвараа, татварын, татвартай share one stem but the default tokenizer emits
# four distinct tokens, so the lexical half of the hybrid retriever barely
# fires for inflected queries. The custom stemmer below strips the most common
# nominal case/number/possessive suffixes so inflected forms collapse to a
# shared stem at BOTH index and query time (bm25s reuses self.stemmer for the
# query, so the two stay consistent).
#
# This is a deliberately conservative, dependency-free heuristic stemmer — not a
# full morphological analyzer. It strips at most ONE suffix and only when a stem
# of at least ``MN_MIN_STEM`` characters remains, to avoid over-stemming short
# words into collisions. Pass your own ``stemmer`` / ``stopwords`` to
# ``HybridRetriever`` for a different language.

MN_MIN_STEM = 3

# Suffixes ordered LONGEST-FIRST so the longest valid match wins (e.g. "ийн"
# before "н"). Grouped by grammatical role for readability. Verb morphology is
# intentionally excluded — it's far riskier for over-stemming than nominal cases.
MN_SUFFIXES = [
    # plural / collective
    "нуудаа", "нүүдээ", "чуудаа", "чүүдээ",
    "нууд", "нүүд", "чууд", "чүүд", "ууд", "үүд", "нар", "нэр",
    # ablative / instrumental / comitative (3-char vowel-harmony variants)
    "аас", "ээс", "оос", "өөс",
    "аар", "ээр", "оор", "өөр",
    "тай", "тэй", "той",
    "руу", "рүү", "луу", "лүү",
    # genitive
    "гийн", "ийн", "ний", "ын", "ий",
    # accusative
    "ийг", "ыг",
    # dative-locative
    "нд", "ад", "эд", "од", "өд",
    # reflexive possessive
    "аа", "ээ", "оо", "өө",
    # short single-char case markers (last — only strip if a real stem remains)
    "г", "д", "т", "н",
]


def mn_stem_word(token: str) -> str:
    """Strip at most one common Mongolian nominal suffix, longest match first."""
    for suf in MN_SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= MN_MIN_STEM:
            return token[: -len(suf)]
    return token


def mn_stemmer(tokens: list[str]) -> list[str]:
    """bm25s stemmer contract: list[str] -> list[str]."""
    return [mn_stem_word(t) for t in tokens]


# A small Mongolian stopword set (particles/copulas that carry no retrieval
# signal). Passed as ``language=`` because BM25Retriever forwards it straight to
# bm25s.tokenize's ``stopwords`` arg, which accepts a custom list.
MN_STOPWORDS = [
    "ба", "буюу", "болон", "нь", "юм", "вэ", "бэ", "уу", "үү",
    "энэ", "тэр", "гэж", "гэх", "байна", "бол", "мөн", "тухай",
]


# --- Cross-encoder reranker (low-resource friendly, e.g. Mongolian) ---------
# QueryFusionRetriever only returns an RRF *rank* score, and both of its
# retrievers are bi-encoders/lexical (query and passage encoded separately).
# A cross-encoder jointly encodes (query, passage) in a single pass, so it
# models token-level interaction and gives a true relevance score. This is
# where most of the precision-at-top win comes from for a low-resource,
# morphologically rich language like Mongolian, where bge-m3 recall is decent
# but top-k ordering is noisy. bge-reranker-v2-m3 is the same family as the
# bge-m3 embeddings, is explicitly multilingual, and ships with the
# already-installed FlagEmbedding package (no new dependency).
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

_reranker = None
_reranker_name: Optional[str] = None


def _get_reranker(model_name: str = DEFAULT_RERANKER_MODEL):
    """Lazily load (and cache) the cross-encoder — the model is heavy, so only
    pay for it when reranking is actually requested."""
    global _reranker, _reranker_name
    if _reranker is None or _reranker_name != model_name:
        from FlagEmbedding import FlagReranker

        _reranker = FlagReranker(model_name, use_fp16=True)
        _reranker_name = model_name
    return _reranker


async def rerank(
    query: str,
    nodes: list[NodeWithScore],
    top_n: int,
    model_name: str = DEFAULT_RERANKER_MODEL,
) -> list[NodeWithScore]:
    """Re-score fused candidates with the cross-encoder and keep the best ``top_n``.

    Scores are sigmoid-normalised to 0-1 so they're comparable to fusion scores
    when eyeballing results side by side.
    """
    if not nodes:
        return nodes
    reranker = _get_reranker(model_name)
    pairs = [(query, n.node.get_content()) for n in nodes]
    scores = await asyncio.to_thread(reranker.compute_score, pairs, normalize=True)
    if not isinstance(scores, list):
        scores = [scores]
    for n, s in zip(nodes, scores):
        n.score = float(s)
    return sorted(nodes, key=lambda n: n.score, reverse=True)[:top_n]


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
