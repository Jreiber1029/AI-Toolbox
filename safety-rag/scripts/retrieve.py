import sys
from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


PROJECT_DIR = Path("/home/jasonreiber/safety-rag")
STORAGE_DIR = PROJECT_DIR / "storage"
COLLECTION_NAME = "safety_rules"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def main():
      if len(sys.argv) < 2:
          raise SystemExit(
              'Usage: python retrieve.py "description to search for"'
          )

      query = " ".join(sys.argv[1:])

      embed_model = FastEmbedEmbedding(
          model_name=EMBEDDING_MODEL,
      )

      qdrant_client = QdrantClient(
          path=str(STORAGE_DIR),
      )

      vector_store = QdrantVectorStore(
          client=qdrant_client,
          collection_name=COLLECTION_NAME,
      )

      index = VectorStoreIndex.from_vector_store(
          vector_store=vector_store,
          embed_model=embed_model,
      )

      retriever = index.as_retriever(
          similarity_top_k=3,
      )

      results = retriever.retrieve(query)

      print(f"\nQuery: {query}")
      print(f"Retrieved {len(results)} result(s)\n")

      for number, result in enumerate(results, start=1):
          metadata = result.node.metadata
          source = metadata.get("file_name", "Unknown source")

          print(f"--- Result {number} ---")
          print(f"Score: {result.score:.4f}")
          print(f"Source: {source}")
          print(result.node.get_content())
          print()


if __name__ == "__main__":
      main()
