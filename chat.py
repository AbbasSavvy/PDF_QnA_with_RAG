import sys
from src.embedder import embed_chunks
from src.vector_store import get_client
from src.llm import get_answer
from config import COLLECTION_NAME, TOP_K, CERTAINTY_THRESHOLD, MAX_HISTORY_TURNS
import weaviate.classes as wvc


def retrieve_chunks(question):
    question_vector = embed_chunks([{"text": question}])[0]

    client = get_client()
    collection = client.collections.get(COLLECTION_NAME)

    results = collection.query.near_vector(
        near_vector=question_vector.tolist(),
        limit=TOP_K,
        return_properties=["text", "source", "page"],
        return_metadata=wvc.query.MetadataQuery(certainty=True, distance=True)
    )

    chunks = []
    for obj in results.objects:
        certainty = obj.metadata.certainty
        if certainty >= CERTAINTY_THRESHOLD:
            chunks.append({
                "text": obj.properties["text"],
                "source": obj.properties["source"],
                "page": obj.properties["page"],
                "certainty": certainty,
                "distance": obj.metadata.distance
            })

    if len(chunks) < 3:
        print(f"  (threshold filtered to {len(chunks)} chunks — falling back to all results)")
        chunks = []
        for obj in results.objects:
            chunks.append({
                "text": obj.properties["text"],
                "source": obj.properties["source"],
                "page": obj.properties["page"],
                "certainty": obj.metadata.certainty,
                "distance": obj.metadata.distance
            })

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
        print(f"  Retrieved {len(chunks)} chunks\n")

        # Get answer with history
        answer = get_answer(chunks, question, history)

        print(f"Assistant: {answer}\n")

        # Append turn and trim to MAX_HISTORY_TURNS
        history.append((question, answer))
        history = history[-MAX_HISTORY_TURNS:]


if __name__ == "__main__":
    chat()