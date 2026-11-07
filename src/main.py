# src/main.py
# To run this file "python -m src.main"

from src.retrivel_pipeline.generation import answer_question
from src.retrivel_pipeline.Retrieval import RetrievalEngine


retriever = RetrievalEngine()  


def interactive_loop(default_model: str = "openai/gpt-oss-120b", top_k: int = 5) -> None:
    print("--- IRC RAG Assistant (type 'quit' or 'exit' to stop) ---")
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in {"quit", "exit"}:
            print("Good‑bye!")
            break
        if not question:
            print("Please enter a non‑empty question.")
            continue
        answer = answer_question(
            question=question,
            retriever=retriever,
            top_k=top_k,
            llm_model=default_model,
        )
        print(f"\nAnswer:\n{answer}\n")


if __name__ == "__main__":
    interactive_loop()

