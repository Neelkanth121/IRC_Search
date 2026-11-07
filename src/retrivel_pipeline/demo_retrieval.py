# src/demo_retrieval.py
from src.retrivel_pipeline.Retrieval import RetrievalEngine

if __name__ == "__main__":
    engine = RetrievalEngine(top_k=5)   # adjust top_k as you wish

    while True:
        q = input("\nEnter a question about IRC standards (or 'quit'): ").strip()
        if q.lower() in {"quit", "exit"}:
            break

        results = engine.retrieve(q)
        print("\n--- Top matches ---")
        for i, r in enumerate(results, start=1):
            meta = r["metadata"]
            src = meta.get("source") or meta.get("file_path") or "unknown file"
            print(f"\n{i}. Score: {r['distance']:.4f}")
            print(f"   Source : {src}")
            # Show a one‑line preview of the chunk (first 200 chars)
            preview = r["content"][:200].replace("\n", " ")
            print(f"   Chunk  : {preview}")
