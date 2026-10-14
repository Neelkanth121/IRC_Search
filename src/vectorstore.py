import os
from pathlib import Path
import numpy as np
import chromadb
import uuid
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PERSIST_DIR = str(BASE_DIR / "data" / "vector_store")

def _clean_metadata(raw_meta: dict) -> dict:
    """
    Return a copy of `raw_meta` containing only JSON‑compatible values
    (str, int, float, bool).  `None` entries are dropped, and any
    other type is cast to its string representation.
    """
    clean: dict = {}
    for k, v in raw_meta.items():
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif v is None:
            # skip the key – ChromaDB can store the rest
            continue
        else:
            # fallback: store a readable string
            clean[k] = str(v)
    return clean

class VectorStore:
    """Manages document embeddings in a ChromaDB vector store"""
    
    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = DEFAULT_PERSIST_DIR):
        """
        Initialize the vector store
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create persistent ChromaDB client
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG"}
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
            
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the vector store
        
        Args:
            documents: List of LangChain documents
            embeddings: Corresponding embeddings for the documents
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        print(f"Adding {len(documents)} documents to vector store...")
        
        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            
            # Prepare metadata
            metadata = _clean_metadata(dict(doc.metadata))
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            
            # Document content
            documents_text.append(doc.page_content)
            
            # Embedding
            embeddings_list.append(embedding.tolist())
        
        # Add to collection
        BATCH_SIZE = 500          # you can tune 300‑1000; 500 works well
        total = len(ids)
        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            try:
                self.collection.add(
                    ids=ids[start:end],
                    embeddings=embeddings_list[start:end],
                    metadatas=metadatas[start:end],
                    documents=documents_text[start:end],
                )
                print(f"Added batch {start // BATCH_SIZE + 1}: {end - start} docs")
            except Exception as e:
                print(f"Error adding batch {start}-{end}: {e}")
                raise
        # -----------------------------------------
        print(f"Successfully added {total} documents to vector store")
        print(f"Total documents in collection: {self.collection.count()}")

    