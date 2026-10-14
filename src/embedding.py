from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_loader import load_all_documents

class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 500, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"Initialized EmbeddingPipeline with model: {model_name}, chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}")

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks

    def embed_chunks(self, chunks: list[Document]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"Generating embeddings for {len(texts)} chunks")
        embeddings = self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings


if __name__ == "__main__":
    # Load documents
    docs = load_all_documents("data")
    print(f"Total documents loaded: {len(docs)}")

    # Initialize EmbeddingPipeline
    embedding_pipeline = EmbeddingPipeline()

    # Chunk documents
    chunks = embedding_pipeline.chunk_documents(docs)
    print(f"Total chunks created: {len(chunks)}")

    # Generate embeddings
    embeddings = embedding_pipeline.embed_chunks(chunks)
    print(f"Embeddings shape: {embeddings.shape}")
    