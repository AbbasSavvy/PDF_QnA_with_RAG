import sys
import json
from src.embedder import embed_chunks
from src.vector_store import get_client, query_chunks
from src.llm import get_answer
from config import TOP_K, MAX_HISTORY_TURNS, HYBRID_ALPHA, METADATA_PATH


def load_metadata():
    try:
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: No metadata file found at {METADATA_PATH}. Run ingest.py first.")
        return {}


def retrieve_chunks(question, verbose=False):
    question_vector = embed_chunks([{"text": question}])[0]

    client = get_client()
    chunks = query_chunks(client, question, question_vector, TOP_K, HYBRID_ALPHA)
    client.close()

    if verbose:
        print(f"  Retrieved {len(chunks)} chunks | scores: {[round(c['score'], 4) for c in chunks]}\n")

    return chunks


def chat(verbose=False, confidential=True):
    print("\n=== PDF Q&A Chat ===")
    print(f"Mode: {'confidential (Ollama)' if confidential else 'standard (Groq)'}")
    print("Type your question and press Enter. Type 'exit' or 'quit' to end.")
    if verbose:
        print("(verbose mode on)\n")
    else:
        print()

    doc_metadata = load_metadata()

    history = []

    while True:
        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() in ("exit", "quit"):
            print("Ending chat session.")
            break

        chunks = retrieve_chunks(question, verbose=verbose)
        answer = get_answer(chunks, question, history, confidential=confidential, doc_metadata=doc_metadata)

        pages = sorted(set(chunk["page"] for chunk in chunks))
        pages_str = ", ".join(f"Page {p}" for p in pages)

        print(f"Assistant: {answer}")
        print(f"Sources: {pages_str}\n")

        history.append((question, answer))
        history = history[-MAX_HISTORY_TURNS:]


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv
    confidential = "--no-confidential" not in sys.argv
    chat(verbose=verbose, confidential=confidential)