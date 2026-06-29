"""Cross-encoder reranker (low-resource friendly, e.g. Mongolian).

QueryFusionRetriever only returns an RRF *rank* score, and both of its retrievers
are bi-encoders/lexical (query and passage encoded separately). A cross-encoder
jointly encodes (query, passage) in a single pass, so it models token-level
interaction and gives a true relevance score. This is where most of the
precision-at-top win comes from for a low-resource, morphologically rich language
like Mongolian, where bge-m3 recall is decent but top-k ordering is noisy.
bge-reranker-v2-m3 is the same family as the bge-m3 embeddings, is explicitly
multilingual, and ships with the FlagEmbedding package.

FlagEmbedding is an optional dependency (``pip install ema-agent[rerank]``). It is
imported lazily inside ``_get_reranker`` so the rest of the package works without
it — you only pay for the (heavy) model when reranking is actually requested.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from llama_index.core.schema import NodeWithScore

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
