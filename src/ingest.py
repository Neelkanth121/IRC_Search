## Ingest script for loading documents, generating embeddings, and storing them in ChromaDB.
import sys
from pathlib import Path

# Add the project root to PYTHONPATH when running the script directly
ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.vectorstore import VectorStore


def main(data_dir: str = "data"):
    # 1. Load documents
    print(f"\n=== Loading documents from '{data_dir}' ===")

    docs = load_all_documents(data_dir)

    if not docs:
        print("No documents found. Check the directory and ensure PDFs exist.")
        return

    # 2. Split documents into chunks
    print("\n=== Chunking documents ===")

    embedder = EmbeddingPipeline()
    chunks = embedder.chunk_documents(docs)

    print(f"Created {len(chunks)} chunks")

    # 3. Generate embeddings
    print("\n=== Generating embeddings ===")

    embeddings = embedder.embed_chunks(chunks)

    print(f"Embeddings shape: {embeddings.shape}")

    # 4. Store documents and embeddings in ChromaDB
    print("\n=== Saving to ChromaDB vector store ===")

    store = VectorStore()
    store.add_documents(chunks, embeddings)

    # 5. Display ingestion summary
    print("\nIngestion finished successfully.")
    print(f"Total documents in collection: {store.collection.count()}")


if __name__ == "__main__":
    # Accept an optional data directory from the command line
    dir_arg = sys.argv[1] if len(sys.argv) > 1 else "data"
    main(dir_arg)