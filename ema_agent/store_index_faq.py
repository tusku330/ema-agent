from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import asyncio
import sys

load_dotenv()
sys.path.append(str(Path(__file__).parent.parent))

from ema_agent.store_index_init import FaissVectorStore, _new_faiss_store, embed_dim
from ema_agent.agent_starter import Topic

DATA_PATH = "./data/datasets_20260613-2.xlsx"
STORAGE_PATH = "./storage/faq_hybrid_2"


def _load_nodes() -> tuple[list[TextNode], dict[str, str]]:
    """
    Indexes question + answer together so keyword-rich answer terms
    ('e-tax', 'ТБ-01 маягт', 'НӨАТ босго') are searchable.
    store_index_1 only indexed the question, missing these signals.
    """
    faq_df = pd.read_excel(DATA_PATH, sheet_name="faq")
    doc_df = pd.read_excel(DATA_PATH, sheet_name="document")
    answer_map = dict(zip(doc_df["document_id"], doc_df["text"].fillna("")))

    nodes = []
    for _, row in faq_df.iterrows():
        doc_id = str(row.get("document_id", "")).strip()
        question = str(row.get("question", "")).strip()
        answer = str(answer_map.get(doc_id, "")).strip()
        topic = str(row.get("topic", "")).strip()

        nodes.append(
            TextNode(
                text=f"Асуулт: {question}\nХариулт: {answer}",
                metadata={
                    "document_id": doc_id,
                    "topic": topic,
                    "question": question,
                    "source": "FAQ",
                },
            )
        )

    return nodes, answer_map


def _build_vector_index(nodes: list[TextNode]) -> VectorStoreIndex:
    try:
        vs = FaissVectorStore.from_persist_dir(STORAGE_PATH)
        sc = StorageContext.from_defaults(vector_store=vs, persist_dir=STORAGE_PATH)
        return load_index_from_storage(storage_context=sc)
    except Exception:
        pass

    vs = _new_faiss_store(embed_dim)
    sc = StorageContext.from_defaults(vector_store=vs)
    index = VectorStoreIndex(nodes, storage_context=sc)
    index.storage_context.persist(persist_dir=STORAGE_PATH)
    return index


_nodes, _answer_map = _load_nodes()


def _validate_topics(nodes: list[TextNode]) -> None:
    """Fail loudly at import if the dataset uses a topic not in the `Topic` enum.

    Retrieval filters by exact string equality (`metadata["topic"] == topic`),
    and the router can only emit `Topic` members. So any topic present in the
    data but absent from the enum is unreachable — it would silently return zero
    nodes -> empty context -> "info not found", with no error. Catch the drift
    here instead, with the offending value named.
    """
    valid = {t.value for t in Topic}
    data_topics = {n.metadata["topic"] for n in nodes if n.metadata.get("topic")}
    unknown = data_topics - valid
    assert not unknown, (
        f"Dataset has topic(s) not in Topic enum: {sorted(unknown)}. "
        f"Add them to component.agent_starter.Topic or fix the data."
    )


_validate_topics(_nodes)
_vector_index = _build_vector_index(_nodes)


# --- Cross-encoder reranker (low-resource friendly, e.g. Mongolian) ---------
# QueryFusionRetriever only returns an RRF *rank* score, and both of its
# retrievers are bi-encoders/lexical (query and passage encoded separately).
# A cross-encoder jointly encodes (query, passage) in a single pass, so it
# models token-level interaction and gives a true relevance score. This is
# where most of the precision-at-top win comes from for a low-resource,
# morphologically rich language like Mongolian, where bge-m3 recall is decent
# but top-k ordering is noisy. bge-reranker-v2-m3 is the same family as the
# bge-m3 embeddings already used, is explicitly multilingual, and ships with
# the already-installed FlagEmbedding package (no new dependency).
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_reranker = None


def _get_reranker():
    """Lazily load the cross-encoder — the model is heavy, so only pay for it
    when reranking is actually requested."""
    global _reranker
    if _reranker is None:
        from FlagEmbedding import FlagReranker

        _reranker = FlagReranker(RERANKER_MODEL, use_fp16=True)
    return _reranker


async def _rerank(query: str, nodes: list, top_n: int) -> list:
    """Re-score fused candidates with the cross-encoder and keep the best top_n.
    Scores are sigmoid-normalised to 0-1 so they're comparable to fusion scores
    when eyeballing results side by side."""

    if not nodes:
        return nodes
    reranker = _get_reranker()
    pairs = [(query, n.node.get_content()) for n in nodes]
    scores = await asyncio.to_thread(reranker.compute_score, pairs, normalize=True)
    if not isinstance(scores, list):
        scores = [scores]
    for n, s in zip(nodes, scores):
        n.score = float(s)
    return sorted(nodes, key=lambda n: n.score, reverse=True)[:top_n]


# --- Mongolian-aware BM25 tokenization (P1 fix) -----------------------------
# bm25s's default pipeline tokenizes Cyrillic fine (the `(?u)` token pattern
# treats Cyrillic as word chars) but then applies the *English* Snowball
# stemmer, which is a no-op on Mongolian. Mongolian is agglutinative: татвар,
# татвараа, татварын, татвартай share one stem but the default tokenizer emits
# four distinct tokens, so the lexical half of the hybrid retriever barely
# fires for inflected queries. We pass a custom stemmer callable that strips the
# most common nominal case/number/possessive suffixes so inflected forms collapse
# to a shared stem at BOTH index and query time (bm25s reuses self.stemmer for
# the query, so the two stay consistent).
#
# This is a deliberately conservative, dependency-free heuristic stemmer — not a
# full morphological analyzer. It strips at most ONE suffix and only when a
# stem of at least _MN_MIN_STEM characters remains, to avoid over-stemming short
# words into collisions.

_MN_MIN_STEM = 3

# Suffixes ordered LONGEST-FIRST so the longest valid match wins (e.g. "ийн"
# before "н"). Grouped by grammatical role for readability. Verb morphology is
# intentionally excluded — it's far riskier for over-stemming than nominal cases.
_MN_SUFFIXES = [
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


def _mn_stem_word(token: str) -> str:
    """Strip at most one common Mongolian nominal suffix, longest match first."""
    for suf in _MN_SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= _MN_MIN_STEM:
            return token[: -len(suf)]
    return token


def _mn_stemmer(tokens: list[str]) -> list[str]:
    """bm25s stemmer contract: list[str] -> list[str]."""
    return [_mn_stem_word(t) for t in tokens]


# A small Mongolian stopword set (particles/copulas that carry no retrieval
# signal). Passed as `language=` because BM25Retriever forwards it straight to
# bm25s.tokenize's `stopwords` arg, which accepts a custom list.
_MN_STOPWORDS = [
    "ба", "буюу", "болон", "нь", "юм", "вэ", "бэ", "уу", "үү",
    "энэ", "тэр", "гэж", "гэх", "байна", "бол", "мөн", "тухай",
]


def getOutput(doc_id: str) -> str | None:
    return _answer_map.get(doc_id)


def _topic_value(topic: Topic | str | None) -> str | None:
    """Accept Topic enum, plain string, or None; return the string value or None."""
    if isinstance(topic, Topic):
        topic = topic.value
    return None if not topic or topic == Topic.NONE.value else topic


def _get_retriever(topic: Topic | str | None, similarity_top_k: int) -> QueryFusionRetriever:
    # BM25: pre-filter nodes by topic so keyword scores stay within the domain
    topic = _topic_value(topic)
    nodes = _nodes
    if topic:
        nodes = [n for n in _nodes if n.metadata.get("topic") == topic]

    # Mongolian-aware BM25: custom suffix-stripping stemmer + MN stopwords so
    # inflected forms (татвар/татвараа/татварын) collapse to a shared stem.
    bm25 = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=similarity_top_k,
        stemmer=_mn_stemmer,
        language=_MN_STOPWORDS,
    )

    # Vector: FAISS has no metadata filter support — use wider top_k
    # and post-filter after fusion
    vector = _vector_index.as_retriever(similarity_top_k=similarity_top_k * 3)

    return QueryFusionRetriever(
        retrievers=[vector, bm25],
        similarity_top_k=similarity_top_k * 3,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=True,
        llm=None,
    )

async def retrieve_documents(
    query: str,
    topic: Topic | str | None,
    use_reranker: bool = False,
    top_n: int = 4,
) -> str:
    """Drop-in replacement for store_index_1.retrieve_documents.

    Set use_reranker=True to re-score the fused candidates with the
    bge-reranker-v2-m3 cross-encoder before truncating to top_n. When
    reranking, the fusion candidate pool is widened so the cross-encoder has
    more to choose from (rerank precision improves with recall).
    """
    topic = _topic_value(topic)
    # Wider candidate pool when reranking — the cross-encoder reorders it.
    base_k = 8 if use_reranker else 4
    retriever = _get_retriever(topic, similarity_top_k=base_k)
    raw_nodes = await retriever.aretrieve(query)

    # Post-filter: drop vector-side results outside the topic
    if topic:
        raw_nodes = [
            n for n in raw_nodes
            if n.node.metadata.get("topic") == topic
        ]

    if use_reranker:
        raw_nodes = await _rerank(query, raw_nodes, top_n)
    else:
        raw_nodes = raw_nodes[:top_n]

    response = ""
    for nws in raw_nodes:
        doc_id = nws.node.metadata.get("document_id")
        answer = getOutput(doc_id)
        response += f"score: {nws.score}, text: {answer}\n"

    return response


async def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    test_cases = [
        # keyword in answer, not in question — store_index_1 would miss this
        ("e-tax систем дээр яаж тайлан илгээх вэ?", Topic.TAX),
        # exact question phrasing — both should work
        ("Тайлангаа яаж өгөх вэ?", Topic.TAX),
        # keyword 'ТБ-01' is only in the answer
        ("ТБ-01 маягт гэж юу вэ?", Topic.TAX),
        # cross-topic: NONE means search all
        ("Байгуулагын тоон гарын үсэг авах", Topic.NONE),
        ("Нийгмийн даатгалд яаж бүртгүүлэх вэ?", Topic.INSURANCE),
    ]

    for query, topic in test_cases:
        print(f"\n{'='*60}")
        print(f"Query: {query}  [topic={topic.value}]")

        print("\n  -- fusion only (RRF rank score) --")
        baseline = await retrieve_documents(query, topic, use_reranker=False)
        for line in baseline.strip().split("\n"):
            print("   ", line[:120])

        print("\n  -- + cross-encoder rerank (bge-reranker-v2-m3, 0-1 score) --")
        reranked = await retrieve_documents(query, topic, use_reranker=True)
        for line in reranked.strip().split("\n"):
            print("   ", line[:120])


if __name__ == "__main__":
    asyncio.run(main())
