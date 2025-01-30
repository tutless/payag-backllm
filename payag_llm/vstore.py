import os

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter


class VectorStore:
    def __init__(
        self,
    ):
        self.embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"), model=os.getenv("EMBEDDING_MODEL")
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=50
        )
        self.vectore_store = PineconeVectorStore(
            index_name=os.getenv("INDEX_NAME"), embedding=self.embeddings
        )
        self.chat = ChatOpenAI(verbose=True, temperature=0)

    def pinecone_store(self, raw_documents):
        print("adding documents into pinecone..")
        documents = self.text_splitter.split_documents(raw_documents)
        self.vectore_store.from_documents(
            documents=documents,
            embedding=self.embeddings,
            index_name=os.getenv("INDEX_NAME"),
        )
        # PineconeVectorStore.from_documents(
        #     documents=documents,
        #     embedding=self.embeddings,
        #     index_name=os.getenv("INDEX_NAME"),
        # )
