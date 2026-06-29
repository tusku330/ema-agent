"""App-specific FAQ retrieval glue for the emon-chatbot dataset.

The reusable mechanics now live in two generic modules so they can be dropped
into another project unchanged:

  * ``ema_agent.indexing``  — embedding config + FAISS load/build helpers.
  * ``ema_agent.retriever`` — ``HybridRetriever`` (vector + BM25 fusion + rerank).

What remains here is everything tied to *this* project: the Excel schema, the
``Topic`` enum taxonomy + validation, the answer-map lookup, and the import-time
index build. ``retrieve_documents`` / ``getOutput`` keep their original
signatures, so existing callers
(``from ema_agent.store_index_faq import retrieve_documents``) are unaffected.
"""

from dotenv import load_dotenv
from llama_index.core.schema import TextNode
import pandas as pd
import asyncio

load_dotenv()

from ema_agent.indexing import configure_embeddings, load_or_build_index
from ema_agent.retrieval import HybridRetriever
from ema_agent.agent_starter import Topic

DATA_PATH = "./data/datasets_20260613-2.xlsx"
STORAGE_PATH = "./storage/faq_hybrid"


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


_nodes, _answer_map = _load_nodes()
_validate_topics(_nodes)

_dim = configure_embeddings()
_vector_index = load_or_build_index(_nodes, STORAGE_PATH, _dim)
_retriever = HybridRetriever(_nodes, _vector_index, filter_key="topic")


def getOutput(doc_id: str) -> str | None:
    return _answer_map.get(doc_id)


def _topic_value(topic: Topic | str | None) -> str | None:
    """Accept Topic enum, plain string, or None; return the string value or None."""
    if isinstance(topic, Topic):
        topic = topic.value
    return None if not topic or topic == Topic.NONE.value else topic


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
    raw_nodes = await _retriever.aretrieve(
        query,
        filter_value=_topic_value(topic),
        use_reranker=use_reranker,
        top_n=top_n,
    )

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
