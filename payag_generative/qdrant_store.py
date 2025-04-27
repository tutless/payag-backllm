from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from vector_store import VectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from langchain_qdrant import QdrantVectorStore


class QdrantStore(VectorStore):
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="nlpaueb/legal-bert-base-uncased"
        )

        self.qdrant_client = QdrantClient(host="qdrant", port=6333)
        self.collection_name = "payag_legal"
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
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
        else:
            print(
                f"Collection '{self.collection_name}' already exists. Skipping creation."
            )

    def splitted_docs(self, docs: list[Document]):
        return RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=128, separators=["\n\n", "\n", ".", " ", ""]
        ).split_documents(documents=docs)

    def vector_store(self, docs: list[Document]):
        return self.qdrant.add_documents(documents=docs)

    def vector_retriever(self):
        return self.qdrant.as_retriever()

    def store_name(self):
        print("Qdrant")
