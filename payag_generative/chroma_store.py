import os
from vector_store import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


class ChromaStore(VectorStore):
    def __init__(self):
        self.db_path = os.path.join(os.getcwd(), "payag_generative", "payag_vector")

    def embedding(self):
        return HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")

    def splitted_docs(self, docs: list[Document]):
        return RecursiveCharacterTextSplitter(
            chunk_size=1200, chunk_overlap=200
        ).split_documents(documents=docs)

    def vector_store(self, docs: list[Document]):
        print("ingesting vectors into chroma...")
        return Chroma.from_documents(
            documents=self.splitted_docs(docs),
            embedding=self.embedding(),
            persist_directory=self.db_path,
        )

    def store_name(self):
        print("ChromaDB")
