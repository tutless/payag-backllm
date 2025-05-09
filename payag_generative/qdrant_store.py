from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from vector_store import VectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from db.main import main_db
from sqlalchemy.orm import sessionmaker

from langchain_qdrant import QdrantVectorStore


class QdrantStore(VectorStore):
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.qdrant_client = QdrantClient(host="qdrant", port=6333)
        self.collection_name = "payag_legal_mid"
        self.create_collection()
        self.qdrant = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding=self.embedding_model,
        )

    def create_collection(self):
        if not self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )
        else:
            print(
                f"Collection '{self.collection_name}' already exists. Skipping creation."
            )

    def recursive_splitter(self, chunk: int, overlap: int):
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def splitted_docs(self, docs: list[Document]):

        section_map: dict[str, Document] = {}
        embedded_chunks: list[Document] = []

        def chunks(section_id: int, chnk: Document):
            chnk.metadata[section_id] = section_id
            embedded_chunks.append(chnk)

        def small_chunks(small_data: tuple[int, Document]):
            section_id, section = small_data
            section_map[section_id] = section.page_content
            small_chunks_size = self.recursive_splitter(300, 50).split_documents(
                documents=[section]
            )
            list(
                map(
                    lambda chunk: chunks(section_id=section_id, chnk=chunk),
                    small_chunks_size,
                )
            )

        def large_chunks(large_data: tuple[int, Document]):
            index, doc = large_data
            large_chunk_size = self.recursive_splitter(2048, 256).split_documents(
                documents=[doc]
            )
            list(
                map(
                    lambda small: small_chunks(small),
                    enumerate(large_chunk_size, start=0),
                ),
            )

        list(map(lambda large: large_chunks(large), enumerate(docs, start=0)))

        return section_map, embedded_chunks

    def recursive_split(self, docs: list[Document]):
        return RecursiveCharacterTextSplitter(
            chunk_size=1200, chunk_overlap=200, separators=["\n\n", "\n", ".", " ", ""]
        ).split_documents(documents=docs)

    def vector_store(self, docs: list[Document]):
        return self.qdrant.add_documents(documents=self.recursive_split(docs))
        # section, embedded = self.splitted_docs(docs=docs)
        # self.save_chunks(section)
        # return self.qdrant.add_documents(documents=embedded)

    # @main_db
    # def save_chunks(self, message: list[str]):
    #     for inserted in message:
    #         print(inserted)

    def vector_retriever(self):
        return self.qdrant.as_retriever(search_kwargs={"k": 3})

    def store_name(self):
        print("Qdrant")
