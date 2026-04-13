import sys
from src.embedder import embed_chunks
from src.vector_store import get_client, query_chunks
from src.llm import get_answer
from config import COLLECTION_NAME, TOP_K, HYBRID_ALPHA


def query(question, verbose=False, confidential=True):
    if verbose:
        print(f"\nQuestion: {question}\n")
        print(f"Mode: {'confidential (Ollama)' if confidential else 'standard (Groq)'}")
        print("--- Embedding question ---")

    question_vector = embed_chunks([{"text": question}])[0]

    if verbose:
        print(f"--- Hybrid search (alpha={HYBRID_ALPHA}) ---")

    client = get_client()
    chunks = query_chunks(client, question, question_vector, TOP_K, HYBRID_ALPHA)
    client.close()

    if verbose:
        print(f"Retrieved {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks, 1):
            print(f"  [{i}] Page {chunk['page']} | score={chunk['score']:.4f} — {chunk['text'][:80].strip()}...")
        print("\n--- Querying LLM ---")

    answer = get_answer(chunks, question, confidential=confidential)

    pages = sorted(set(chunk["page"] for chunk in chunks))
    pages_str = ", ".join(f"Page {p}" for p in pages)

    print(f"\n{'=' * 60}")
    print(f"Answer: ({'Ollama' if confidential else 'Groq'})")
    print(f"{'=' * 60}")
    print(answer)
    print(f"Sources: {pages_str}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Usage: python query.py \"your question here\" [--verbose] [--no-confidential]")
        sys.exit(1)
    verbose = "--verbose" in args
    confidential = "--no-confidential" not in args
    question_args = [a for a in args if a not in ("--verbose", "--no-confidential")]
    question = " ".join(question_args)
    query(question, verbose=verbose, confidential=confidential)