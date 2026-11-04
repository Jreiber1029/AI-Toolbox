from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


PROJECT_DIR = Path("/home/jasonreiber/safety-rag")
DOCUMENTS_DIR = PROJECT_DIR / "documents"
STORAGE_DIR = PROJECT_DIR / "storage"
COLLECTION_NAME = "safety_rules"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def main():
      print(f"Loading documents from {DOCUMENTS_DIR}")

      documents = SimpleDirectoryReader(
          input_dir=str(DOCUMENTS_DIR),
          recursive=True,
          required_exts=[".pdf", ".txt", ".md"],
      ).load_data()

      if not documents:
          raise RuntimeError("No supported documents were found")

      print(f"Loaded {len(documents)} document sections/pages")

      embed_model = FastEmbedEmbedding(
          model_name=EMBEDDING_MODEL,
      )

      splitter = SentenceSplitter(
          chunk_size=400,
          chunk_overlap=50,
      )

      qdrant_client = QdrantClient(
          path=str(STORAGE_DIR),
      )

      vector_store = QdrantVectorStore(
          client=qdrant_client,
          collection_name=COLLECTION_NAME,
      )

      storage_context = StorageContext.from_defaults(
          vector_store=vector_store,
      )

      VectorStoreIndex.from_documents(
          documents,
          storage_context=storage_context,
          embed_model=embed_model,
          transformations=[splitter],
          show_progress=True,
      )

      print("Index created successfully")
      print(f"Stored in: {STORAGE_DIR}")
      print(f"Collection: {COLLECTION_NAME}")


if __name__ == "__main__":
      main()
