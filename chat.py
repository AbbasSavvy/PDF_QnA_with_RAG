import sys
from src.embedder import embed_chunks
from src.vector_store import get_client, query_chunks
from src.llm import get_answer
from config import COLLECTION_NAME, TOP_K, MAX_HISTORY_TURNS, HYBRID_ALPHA
import weaviate.classes as wvc


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

        # Retrieve chunks
        chunks = retrieve_chunks(question)
        print(f"  Retrieved {len(chunks)} chunks | scores: {[round(c['score'], 4) for c in chunks]}\n")

        # Get answer with history
        answer = get_answer(chunks, question, history)

        print(f"Assistant: {answer}\n")

        # Append turn and trim to MAX_HISTORY_TURNS
        history.append((question, answer))
        history = history[-MAX_HISTORY_TURNS:]


if __name__ == "__main__":
    chat()