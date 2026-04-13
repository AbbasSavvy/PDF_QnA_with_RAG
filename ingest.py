import sys
import json
import os
from src.loader import load_pdf
from src.chunker import chunk_pages
from src.embedder import embed_chunks
from src.vector_store import get_client, setup_collection, store_chunks
from config import METADATA_PATH

def ingest(pdf_path):
    print(f"\n--- Loading PDF ---")
    pages, metadata = load_pdf(pdf_path)
    print(f"Loaded {len(pages)} pages")
    print(f"Metadata: {metadata}")

    print(f"\n--- Saving metadata ---")
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved to {METADATA_PATH}")

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

    print(f"\nIngestion complete!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_pdf>")
        sys.exit(1)
    ingest(sys.argv[1])