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
from config import TOP_K, MAX_HISTORY_TURNS, HYBRID_ALPHA


def retrieve_chunks(question, verbose=False):
    question_vector = embed_chunks([{"text": question}])[0]

    client = get_client()
    chunks = query_chunks(client, question, question_vector, TOP_K, HYBRID_ALPHA)
    client.close()

    if verbose:
        print(f"  Retrieved {len(chunks)} chunks | scores: {[round(c['score'], 4) for c in chunks]}")

    return chunks


def chat(verbose=False):
    print("\n=== PDF Q&A Chat ===")
    print("Type your question and press Enter. Type 'exit' or 'quit' to end.")
    if verbose:
        print("(verbose mode on)\n")
    else:
        print()

    history = []

    while True:
        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() in ("exit", "quit"):
            print("Ending chat session.")
            break

        # Retrieve chunks
        chunks = retrieve_chunks(question, verbose=verbose)

        # Get answer with history
        answer = get_answer(chunks, question, history)

        # Source attribution
        pages = sorted(set(chunk["page"] for chunk in chunks))
        pages_str = ", ".join(f"Page {p}" for p in pages)

        print(f"\nAssistant: {answer}")
        print(f"Sources: {pages_str}\n")

        # Append turn and trim to MAX_HISTORY_TURNS
        history.append((question, answer))
        history = history[-MAX_HISTORY_TURNS:]


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv
    chat(verbose=verbose)