from chroma_store import ChromaStore
from qdrant_store import QdrantStore
from weaviate_store import WeaviateStore
from scrape_docs import ScapeDocs
from vector_store import VectorStore
import reactivex as rx
from reactivex import operators as ops
from langchain_chroma import Chroma
from langchain_weaviate import WeaviateVectorStore as Weaviate
from langchain_qdrant import Qdrant
import threading


class DataIngestion:
    def __init__(self, vstore: VectorStore[Qdrant]):
        self.vstore = vstore
        self.scrape_docs = ScapeDocs()

    def doc_ingestion(self):
        done_event = threading.Event()
        buffered_docs = self.scrape_docs.buffer_docs().pipe(
            ops.flat_map(lambda docs: rx.from_iterable(docs))
        )

        def on_next(vector):
            print("Ingestion Start...")
            self.vstore.vector_store(vector)

        def on_error(error):
            print(f"Error during ingestion: {error}")
            done_event.set()  # Unblock even on error

        def on_completed():
            print("Completed ingestion.")
            done_event.set()  # Unblock after completion

        buffered_docs.subscribe(
            on_next=on_next,
            on_error=on_error,
            on_completed=on_completed,
        )

        done_event.wait()  # Block until observable complete (prevent docker from  early exit)


if __name__ == "__main__":
    qadrant_instance = QdrantStore()
    ingestion = DataIngestion(qadrant_instance)
    ingestion.doc_ingestion()
    # chroma_instance = ChromaStore()
    # ingestion = DataIngestion(chroma_instance)
    # ingestion.doc_ingestion()
