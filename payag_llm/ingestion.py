import os
import reactivex as rx
from reactivex import operators as ops
from langchain_community.document_loaders import TextLoader
from payag_llm.vstore import VectorStore


class Ingestion:
    def __init__(self):
        self.root_path = r"C:\DownloadedWeb\payag\lawphil.net".replace("\\", "/")

    def walk_through(self):

        return (
            rx.from_iterable(os.walk(self.root_path))
            .pipe(
                ops.flat_map(
                    lambda path: rx.from_iterable(path[2]).pipe(
                        ops.map(
                            lambda file: os.path.join(path[0], file).replace("\\", "/")
                        ),
                    )
                )
            )
            .pipe(ops.reduce(lambda accu, element: [*accu, element], []))
        )

    def text_loader(self):
        def load_text(path):
            print(path)
            loader = TextLoader(file_path=path, encoding="utf8")
            return loader.load()

        return rx.compose(ops.flat_map(lambda path: load_text(path)), ops.to_list())

    def rx_documents(self):
        return self.walk_through().pipe(
            ops.flat_map(lambda path: rx.from_iterable(path).pipe(self.text_loader())),
        )

    def rx_addstore(self):
        self.rx_documents().subscribe(
            lambda x: VectorStore().pinecone_store(x),
            lambda e: print(e),
            lambda: print("done"),
        )

    @classmethod
    def ingest_documents(cls):
        cls().rx_addstore()
