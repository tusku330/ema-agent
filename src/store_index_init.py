from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
import logging
import sys

Settings.embed_model = HuggingFaceEmbedding(
    # model_name="gmunkhtur/paraphrase-mongolian-minilm-mn_v2"      #384
    # model_name="sentence-transformers/distiluse-base-multilingual-cased-v2"  # 512
    # model_name="sentence-transformers/LaBSE",  # storage/services1      d = 768
    # model_name="intfloat/multilingual-e5-base",  # storage/services1      d = 768
    # model_name="intfloat/multilingual-e5-large-instruct",  # d = 1024
    model_name="BAAI/bge-m3",  # d = 1024
)
# Settings.llm = llm3

from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.node_parser import SentenceSplitter

# create the sentence window node parser w/ default settings
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

embed_dim = 1024

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def _new_faiss_store(dim=embed_dim):
    return FaissVectorStore(faiss_index=faiss.IndexFlatL2(dim))


import pandas as pd
import json
from llama_index.core import Document


def make_documents_from_excel(path: str, doc_type: str):
    df = pd.read_excel(path, sheet_name=doc_type)

    docs = []


    # if doc_type == "faq":
    #     answer_df = pd.read_excel(path, sheet_name="document")
    #     answer_map = dict(zip(answer_df["document_id"], answer_df["text"]))

    for _, row in df.iterrows():
        if doc_type == "faq":
            # answer = answer_map.get(row.get("document_id"), "")
            # text = f"Асуулт: {row.get('question')}\nХариулт: {answer}"

            text = f"""
Асуулт: {row.get('question')}
            """

            docs.append(
                Document(
                    text=text.strip(),
                    metadata={
                        "question": row.get("question"),
                        "topic": row.get("topic"),
                        "document_id": row.get("document_id"),
                        "source": "FAQ",
                    },
                )
            )
        elif doc_type == "document":
            text = f"""
Хариулт: {row.get('text')}
            """
            docs.append(
                Document(
                    text=text.strip(),
                    metadata={
                        "answer": row.get("text"),
                        "document_id": row.get("document_id"),
                        "source": "FAQ",
                    },
                )
            )
    return docs


def make_documents_from_json(path: str, type: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for s in data:
        if type == "service_request2":
            docs.append(
                Document(
                    text=s.get("service_name", ""),
                    metadata={
                        "service_name": s.get("service_name"),
                        "service_code": s.get("service_code"),
                        "service_type": s.get("service_type"),
                    },
                )
            )

    return docs
