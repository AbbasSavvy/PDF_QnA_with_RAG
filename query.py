import sys
import os
import warnings

# Suppress warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from src.embedder import embed_chunks
from src.vector_store import get_client, query_chunks
from src.llm import get_answer
from config import COLLECTION_NAME, TOP_K, HYBRID_ALPHA


def query(question, verbose=False):
    if verbose:
        print(f"\nQuestion: {question}\n")

    # Step 1 — Embed the question
    if verbose:
        print("--- Embedding question ---")
    question_vector = embed_chunks([{"text": question}])[0]

    # Step 2 — Retrieve chunks via hybrid search
    if verbose:
        print(f"--- Hybrid search (alpha={HYBRID_ALPHA}) ---")
    client = get_client()
    chunks = query_chunks(client, question, question_vector, TOP_K, HYBRID_ALPHA)
    client.close()

    # Step 3 — Show retrieved chunks
    if verbose:
        print(f"Retrieved {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks, 1):
            print(f"  [{i}] Page {chunk['page']} | score={chunk['score']:.4f} — {chunk['text'][:80].strip()}...")

    # Step 4 — Get answer from Ollama
    if verbose:
        print("\n--- Querying Ollama ---")
    answer = get_answer(chunks, question)

    # Step 5 — Source attribution
    pages = sorted(set(chunk["page"] for chunk in chunks))
    pages_str = ", ".join(f"Page {p}" for p in pages)

    print(f"\n{'='*60}")
    print("Answer:")
    print(f"{'='*60}")
    print(answer)
    print(f"\nSources: {pages_str}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        print("Usage: python query.py \"your question here\" [--verbose]")
        sys.exit(1)

    verbose = "--verbose" in args
    question_args = [a for a in args if a != "--verbose"]
    question = " ".join(question_args)

    query(question, verbose=verbose)