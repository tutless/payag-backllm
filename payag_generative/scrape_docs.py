import os
import reactivex as rx
from reactivex import operators as ops
from langchain_community.document_loaders import TextLoader


class ScapeDocs:
    def __init__(self):
        self.root_path = os.path.join(
            os.getcwd(), "payag_generative", "jurisprudence", "courts", "rules"
        )

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
            ops.flat_map(
                lambda path: rx.from_iterable(path).pipe(self.text_loader()),
            ),
        )

    def buffer_docs(self):
        return self.rx_documents().pipe(ops.buffer_with_count(200))


# if __name__ == "__main__":
#     data_ingest = ScapeDocs()
#     data_ingest.rx_documents().subscribe(
#         on_next=lambda x: print(len(x)),
#         on_error=lambda error: print(error),
#         on_completed=lambda: print("completed"),
#     )
