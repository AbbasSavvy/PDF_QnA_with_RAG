import sys
from src.embedder import embed_chunks
from src.vector_store import get_client, query_chunks
from src.llm import get_answer
from config import COLLECTION_NAME, TOP_K, HYBRID_ALPHA


def query(question):
    print(f"\nQuestion: {question}\n")

    # Step 1 — Embed the question
    print("--- Embedding question ---")
    question_vector = embed_chunks([{"text": question}])[0]

    # Step 2 — Retrieve chunks via hybrid search
    print(f"--- Hybrid search (alpha={HYBRID_ALPHA}) ---")
    client = get_client()
    chunks = query_chunks(client, question, question_vector, TOP_K, HYBRID_ALPHA)
    client.close()

    # Step 3 — Show retrieved chunks
    print(f"Retrieved {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}] Page {chunk['page']} | score={chunk['score']:.4f} — {chunk['text'][:80].strip()}...")

    # Step 4 — Get answer from Ollama
    print("\n--- Querying Ollama ---")
    answer = get_answer(chunks, question)

    print(f"\n{'='*60}")
    print("Answer:")
    print(f"{'='*60}")
    print(answer)
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query.py \"your question here\"")
        sys.exit(1)
    query(sys.argv[1])