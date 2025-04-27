import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from payag_generative.vector_store import VectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv()


class PineconeStore(VectorStore):
    def __init__(self):

        self.embedding = OpenAIEmbeddings()
        self.chat_openai = ChatOpenAI(verbose=True, temperature=0)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200, chunk_overlap=200
        )
        self.pinecone_store = PineconeVectorStore(embedding=self.embedding)

    def vector_store(self, docs):
        documents = self.text_splitter.split_documents(docs)
        self.pinecone_store.from_documents(
            documents=documents, embedding=self.embedding
        )

    def vector_retriever(self):
        return self.pinecone_store.as_retriever()

    def store_name(self):
        return "Pinecone store"
