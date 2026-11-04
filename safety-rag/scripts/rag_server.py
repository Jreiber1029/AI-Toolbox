import json
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


STORAGE_DIR = Path("/home/jasonreiber/safety-rag/storage")
COLLECTION_NAME = "safety_rules"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
HOST = "127.0.0.1"
PORT = 8090
TOP_K = 3


print("Loading FastEmbed and the safety index...")

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
      similarity_top_k=TOP_K,
)


class RequestHandler(BaseHTTPRequestHandler):
      def send_json(self, status, data):
          encoded = json.dumps(data).encode("utf-8")

          self.send_response(status)
          self.send_header("Content-Type", "application/json")
          self.send_header("Content-Length", str(len(encoded)))
          self.end_headers()
          self.wfile.write(encoded)

      def do_POST(self):
          if self.path != "/retrieve":
              self.send_json(404, {"error": "Endpoint not found"})
              return

          try:
              content_length = int(
                  self.headers.get("Content-Length", "0")
              )
              request_body = self.rfile.read(content_length)
              request_data = json.loads(request_body)

              observation = request_data.get(
                  "observation", ""
              ).strip()

              if not observation:
                  self.send_json(
                      400,
                      {"error": "Observation is required"},
                  )
                  return

              retrieved_nodes = retriever.retrieve(observation)
              references = []

              for result in retrieved_nodes:
                  metadata = result.node.metadata

                  references.append(
                      {
                          "source": metadata.get(
                              "file_name", "Unknown source"
                          ),
                          "page": metadata.get("page_label"),
                          "score": result.score,
                          "text": result.node.get_content(),
                      }
                  )

              self.send_json(
                  200,
                  {
                      "observation": observation,
                      "references": references,
                  },
              )

          except Exception as error:
              self.send_json(
                  500,
                  {"error": str(error)},
              )

      def log_message(self, message_format, *args):
          print(
              f"{self.client_address[0]} - "
              f"{message_format % args}"
          )


def main():
      server = HTTPServer(
          (HOST, PORT),
          RequestHandler,
      )

      print(f"Safety RAG service listening on http://{HOST}:{PORT}")
      print("Press Ctrl+C to stop.")

      try:
          server.serve_forever()
      except KeyboardInterrupt:
          print("\nStopping safety RAG service.")
      finally:
          server.server_close()
          qdrant_client.close()


if __name__ == "__main__":
      main()
