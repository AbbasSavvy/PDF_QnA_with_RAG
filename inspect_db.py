import weaviate
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import WEAVIATE_HOST, WEAVIATE_PORT, COLLECTION_NAME

client = weaviate.connect_to_local(host=WEAVIATE_HOST, port=WEAVIATE_PORT)
collection = client.collections.get(COLLECTION_NAME)

# --- Total count ---
response = collection.aggregate.over_all(total_count=True)
print(f"{'='*60}")
print(f"Collection: {COLLECTION_NAME}")
print(f"Total chunks: {response.total_count}")
print(f"{'='*60}\n")

# --- Page-by-page breakdown ---
all_objects = collection.query.fetch_objects(limit=200)

page_counts = {}
for obj in all_objects.objects:
    page = obj.properties['page']
    page_counts[page] = page_counts.get(page, 0) + 1

print("Chunks per page:")
for page in sorted(page_counts.keys()):
    bar = '█' * page_counts[page]
    print(f"  Page {page:>2}: {bar} ({page_counts[page]})")

print()

# --- Preview 3 chunks ---
print(f"{'='*60}")
print("Chunk previews (first 3):")
print(f"{'='*60}\n")

for i, obj in enumerate(all_objects.objects[:3], 1):
    props = obj.properties
    print(f"[Chunk {i}] Page {props['page']} | {props['source']}")
    print(f"  {props['text'][:150].strip()}...")
    print()

# --- Vector preview (3 chunks) ---
print(f"{'='*60}")
print("Vector preview (first 3 chunks):")
print(f"{'='*60}\n")

vector_results = collection.query.fetch_objects(
    limit=3,
    include_vector=True
)

for i, obj in enumerate(vector_results.objects, 1):
    props = obj.properties
    vector = obj.vector['default']
    print(f"[Chunk {i}] Page {props['page']}")
    print(f"  Text: {props['text'][:80].strip()}...")
    print(f"  Vector dimensions: {len(vector)}")
    print(f"  First 8 values: {[round(v, 4) for v in vector[:8]]}")
    print(f"  Last  8 values: {[round(v, 4) for v in vector[-8:]]}")
    print()

client.close()