import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import weaviate
import weaviate.classes as wvc
from config import WEAVIATE_HOST, WEAVIATE_PORT, COLLECTION_NAME


def get_client():
    client = weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_PORT
    )
    return client


def setup_collection(client):
    if client.collections.exists(COLLECTION_NAME):
        client.collections.delete(COLLECTION_NAME)

    client.collections.create(
        name=COLLECTION_NAME,
        vector_config=wvc.config.Configure.Vectors.self_provided(),
        properties=[
            wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="source", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="page", data_type=wvc.config.DataType.INT),
        ]
    )
    print(f"Collection '{COLLECTION_NAME}' created")


def store_chunks(client, chunks, vectors):
    collection = client.collections.get(COLLECTION_NAME)

    with collection.batch.dynamic() as batch:
        for chunk, vector in zip(chunks, vectors):
            batch.add_object(
                properties={
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "page": chunk["page"]
                },
                vector=vector.tolist()
            )
    print(f"Stored {len(chunks)} chunks in Weaviate")

def query_chunks(client, question, question_vector, top_k, alpha):
    collection = client.collections.get(COLLECTION_NAME)

    results = collection.query.hybrid(
        query=question,
        vector=question_vector.tolist(),
        alpha=alpha,
        limit=top_k,
        return_properties=["text", "source", "page"],
        return_metadata=wvc.query.MetadataQuery(score=True)
    )

    chunks = []
    for obj in results.objects:
        chunks.append({
            "text": obj.properties["text"],
            "source": obj.properties["source"],
            "page": obj.properties["page"],
            "score": obj.metadata.score
        })

    return chunks
