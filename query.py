import sys
from src.embedder import embed_chunks
from src.vector_store import get_client
from src.llm import get_answer
from config import COLLECTION_NAME, TOP_K, CERTAINTY_THRESHOLD
import weaviate.classes as wvc


def query(question):
    print(f"\nQuestion: {question}\n")

    # Step 1 — Embed the question
    print("--- Embedding question ---")
    question_vector = embed_chunks([{"text": question}])[0]

    # Step 2 — Retrieve top-K chunks from Weaviate
    print(f"--- Retrieving top {TOP_K} chunks from Weaviate ---")
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
        distance = obj.metadata.distance
        if certainty >= CERTAINTY_THRESHOLD:
            chunks.append({
                "text": obj.properties["text"],
                "source": obj.properties["source"],
                "page": obj.properties["page"],
                "certainty": certainty,
                "distance": distance
            })

    client.close()

    # Step 3 — Show retrieved chunks
    print(f"Retrieved {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}] Page {chunk['page']} | certainty={chunk['certainty']:.3f} — {chunk['text'][:80].strip()}...")

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
