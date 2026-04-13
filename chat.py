import sys
from src.embedder import embed_chunks
from src.vector_store import get_client, query_chunks
from src.llm import get_answer
from config import TOP_K, MAX_HISTORY_TURNS, HYBRID_ALPHA


def retrieve_chunks(question):
    question_vector = embed_chunks([{"text": question}])[0]

    client = get_client()
    chunks = query_chunks(client, question, question_vector, TOP_K, HYBRID_ALPHA)
    client.close()

    return chunks


def chat():
    print("\n=== PDF Q&A Chat ===")
    print("Type your question and press Enter. Type 'exit' or 'quit' to end.\n")

    history = []

    while True:
        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() in ("exit", "quit"):
            print("Ending chat session.")
            break

        chunks = retrieve_chunks(question)
        print(f"  Retrieved {len(chunks)} chunks | scores: {[round(c['score'], 4) for c in chunks]}\n")

        answer = get_answer(chunks, question, history)

        pages = sorted(set(chunk["page"] for chunk in chunks))
        pages_str = ", ".join(f"Page {p}" for p in pages)

        print(f"Assistant: {answer}")
        print(f"Sources: {pages_str}\n")

        history.append((question, answer))
        history = history[-MAX_HISTORY_TURNS:]


if __name__ == "__main__":
    chat()