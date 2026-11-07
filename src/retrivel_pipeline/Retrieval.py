## To Retrieve relevant document chunks from ChromaDB based on a user query, we define the RetrievalEngine class. This class uses an embedding model to convert the query into a vector and then searches for similar vectors in the ChromaDB collection.
from typing import List, Dict, Any
import numpy as np
from src.ingestion_pipeline.vectorstore import VectorStore
from src.ingestion_pipeline.embedding import EmbeddingPipeline


class RetrievalEngine:
    """Retrieves relevant document chunks from ChromaDB."""

    def __init__(self, store: VectorStore | None = None, embedder: EmbeddingPipeline | None = None, top_k: int = 6):
        self.store = store if store is not None else VectorStore()
        self.embedder = embedder if embedder is not None else EmbeddingPipeline()
        self.top_k = top_k

    def _embed_query(self, query: str) -> np.ndarray:
        """Convert a user query into an embedding vector."""

        query_embedding = self.embedder.model.encode(
            [query],
            normalize_embeddings=True
        )

        return query_embedding[0]

    def retrieve(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        """Retrieve the most relevant document chunks for a query."""

        k = top_k if top_k is not None else self.top_k

        # Generate an embedding for the user's query
        query_embedding = self._embed_query(query)

        # Search for similar embeddings in ChromaDB
        results = self.store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved_documents = []

        # Extract the results for the single query
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]

        # Combine each document with its metadata and distance
        for doc_id, content, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            retrieved_documents.append({
                "id": doc_id,
                "content": content,
                "metadata": metadata,
                "distance": distance
            })

        return retrieved_documents
