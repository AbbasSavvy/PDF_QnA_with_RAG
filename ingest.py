import sys
from src.loader import load_pdf
from src.chunker import chunk_pages
from src.embedder import embed_chunks
from src.vector_store import get_client, setup_collection, store_chunks

def ingest(pdf_path):
    print(f"\n--- Loading PDF ---")
    pages = load_pdf(pdf_path)
    print(f"Loaded {len(pages)} pages")

    print(f"\n--- Chunking ---")
    chunks = chunk_pages(pages)
    print(f"Created {len(chunks)} chunks")

    print(f"\n--- Embedding ---")
    vectors = embed_chunks(chunks)

    print(f"\n--- Storing in Weaviate ---")
    client = get_client()
    setup_collection(client)
    store_chunks(client, chunks, vectors)
    client.close()

    print(f"\n Ingestion complete!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_pdf>")
        sys.exit(1)
    ingest(sys.argv[1])