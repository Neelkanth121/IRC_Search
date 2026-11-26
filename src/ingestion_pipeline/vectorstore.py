import hashlib
import os
from pathlib import Path
from typing import Any, List

import chromadb
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_PERSIST_DIR = str(BASE_DIR / "data" / "vector_store")


def _clean_metadata(raw_meta: dict) -> dict:
    """Keep metadata values compatible with ChromaDB."""
    clean = {}

    for key, value in raw_meta.items():
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif value is not None:
            clean[key] = str(value)

    return clean


class VectorStore:
    """Manages document embeddings in a ChromaDB vector store."""

    def __init__(
        self,
        collection_name: str = "pdf_documents",
        persist_directory: str = DEFAULT_PERSIST_DIR,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB client and collection."""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Document embeddings for RAG"},
            )

            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """Add or update document chunks and their embeddings."""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        print(f"Preparing {len(documents)} documents for vector store...")

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        # Prepare one record for each chunk.
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            metadata = _clean_metadata(dict(doc.metadata))
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)

            source = str(metadata.get("source", ""))
            page = str(metadata.get("page", metadata.get("chunk_number", "")))
            content_hash = hashlib.sha256(
                doc.page_content.encode("utf-8")
            ).hexdigest()

            doc_id = hashlib.sha256(
                f"{source}|{page}|{content_hash}".encode("utf-8")
            ).hexdigest()

            ids.append(doc_id)
            metadatas.append(metadata)
            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        # ChromaDB requires IDs to be unique within each request.
        unique_records = {}
        for doc_id, metadata, text, embedding in zip(
            ids, metadatas, documents_text, embeddings_list
        ):
            unique_records.setdefault(doc_id, (metadata, text, embedding))

        removed = len(ids) - len(unique_records)
        ids = list(unique_records.keys())
        metadatas = [record[0] for record in unique_records.values()]
        documents_text = [record[1] for record in unique_records.values()]
        embeddings_list = [record[2] for record in unique_records.values()]

        print(f"Removed {removed} duplicate chunks")

        # Upsert inserts new IDs and updates IDs already in the collection.
        batch_size = 500
        total = len(ids)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)

            try:
                self.collection.upsert(
                    ids=ids[start:end],
                    embeddings=embeddings_list[start:end],
                    metadatas=metadatas[start:end],
                    documents=documents_text[start:end],
                )
                print(f"Added batch {start // batch_size + 1}: {end - start} docs")
            except Exception as e:
                print(f"Error adding batch {start}-{end}: {e}")
                raise

        print(f"Successfully processed {total} documents")
        print(f"Total documents in collection: {self.collection.count()}")