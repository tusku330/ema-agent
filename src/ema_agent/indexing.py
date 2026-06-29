"""Embedding configuration + FAISS index helpers.

Ported from emon-chatbot's ``component/store_index_init.py`` and the load/build
block of ``store_index_faq.py``. The import-time side effect (setting
``Settings.embed_model`` on import) is replaced by the explicit
:func:`configure_embeddings` call so consuming apps control when the (heavy)
embedding model loads.
"""

from __future__ import annotations

import json
from typing import Optional

import faiss
import pandas as pd
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

DEFAULT_EMBED_MODEL = "BAAI/bge-m3"   # 1024-dim
DEFAULT_EMBED_DIM = 1024


def configure_embeddings(
    model_name: str = DEFAULT_EMBED_MODEL,
    dim: int = DEFAULT_EMBED_DIM,
) -> int:
    """Set the global LlamaIndex embedding model. Returns the embedding dim.

    Call once at app startup before building or loading an index.
    """
    Settings.embed_model = HuggingFaceEmbedding(model_name=model_name)
    return dim


def new_faiss_store(dim: int = DEFAULT_EMBED_DIM) -> FaissVectorStore:
    """A fresh flat-L2 FAISS vector store of the given dimension."""
    return FaissVectorStore(faiss_index=faiss.IndexFlatL2(dim))


def load_or_build_index(
    nodes: list[TextNode],
    storage_path: str,
    dim: int = DEFAULT_EMBED_DIM,
) -> VectorStoreIndex:
    """Load a persisted FAISS index from ``storage_path``, or build + persist one.

    The index is cached on disk after the first build, so subsequent runs skip
    re-embedding. ``configure_embeddings`` must have been called first.
    """
    try:
        vs = FaissVectorStore.from_persist_dir(storage_path)
        sc = StorageContext.from_defaults(vector_store=vs, persist_dir=storage_path)
        return load_index_from_storage(storage_context=sc)
    except Exception:
        pass

    vs = new_faiss_store(dim)
    sc = StorageContext.from_defaults(vector_store=vs)
    index = VectorStoreIndex(nodes, storage_context=sc)
    index.storage_context.persist(persist_dir=storage_path)
    return index


# ── Optional document builders ────────────────────────────────────────────────

def documents_from_excel(
    path: str,
    sheet_name: str,
    text_columns: list[str],
    metadata_columns: Optional[list[str]] = None,
) -> list[Document]:
    """Build ``Document`` objects from an Excel sheet.

    ``text_columns`` are concatenated into the document text; ``metadata_columns``
    (default: all remaining columns) are stored as metadata.
    """
    df = pd.read_excel(path, sheet_name=sheet_name)
    meta_cols = metadata_columns or [c for c in df.columns if c not in text_columns]

    docs: list[Document] = []
    for _, row in df.iterrows():
        text = "\n".join(str(row.get(c, "")).strip() for c in text_columns).strip()
        metadata = {c: row.get(c) for c in meta_cols}
        docs.append(Document(text=text, metadata=metadata))
    return docs


def documents_from_json(
    path: str,
    text_key: str,
    metadata_keys: Optional[list[str]] = None,
) -> list[Document]:
    """Build ``Document`` objects from a JSON array of records."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    docs: list[Document] = []
    for record in data:
        text = str(record.get(text_key, "")).strip()
        if metadata_keys is None:
            metadata = {k: v for k, v in record.items() if k != text_key}
        else:
            metadata = {k: record.get(k) for k in metadata_keys}
        docs.append(Document(text=text, metadata=metadata))
    return docs
