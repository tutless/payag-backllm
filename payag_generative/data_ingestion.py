from chroma_store import ChromaStore
from scrape_docs import ScapeDocs
from vector_store import VectorStore
import reactivex as rx
from reactivex import operators as ops


class DataIngestion:
    def __init__(self, vstore: VectorStore):
        self.vstore = vstore
        self.scrape_docs = ScapeDocs()

    def doc_ingestion(self):

        self.scrape_docs.buffer_docs().pipe(
            ops.flat_map(lambda docs: rx.from_iterable(docs))
        ).subscribe(
            on_next=lambda vector: self.vstore.vector_store(vector),
            on_error=lambda error: print(error),
            on_completed=lambda: print("Completed..."),
        )


if __name__ == "__main__":
    chroma_instance = ChromaStore()
    ingestion = DataIngestion(chroma_instance)
    ingestion.doc_ingestion()
