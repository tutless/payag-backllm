import requests
import os
import weaviate
from weaviate.exceptions import UnexpectedStatusCodeException
from weaviate.auth import AuthClientPassword, AuthBearerToken
from weaviate.connect import ConnectionParams, ProtocolParams
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Weaviate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vector_store import VectorStore
from tenacity import retry, wait_exponential, retry_if_exception_type


class WeaviateStore(VectorStore):
    def __init__(self):

        self.header = {"Content-Type": "application/x-www-form-urlencoded"}
        self.data = {
            "grant_type": "password",
            "client_id": os.getenv("KEY_CLIENT"),
            "username": os.getenv("KEY_CLOAK_USERNAME"),
            "password": os.getenv("KEY_CLOAK_PASS"),
            "scope": "openid offline_access",
        }
        self.embedding = HuggingFaceEmbeddings(
            model_name="nlpaueb/legal-bert-base-uncased"
        )

        self.client_noauth = weaviate.WeaviateClient(
            connection_params=ConnectionParams(
                http=ProtocolParams(host="weaviate", port=8080, secure=False),
                grpc=ProtocolParams(host="weaviate", port=50051, secure=False),
            )
        )

        self.weviate_client_classic = weaviate.Client(
            url=os.getenv("WEAVIATE_LOCAL_URL")
        )

        self.weviate_vector = Weaviate(
            client=self.weviate_client_classic,
            index_name="PayagLegalDocs",
            text_key="legal_text",
            embedding=self.embedding,
        )

    def get_token(self):
        try:
            response = requests.post(
                url=os.getenv("KEY_CLOAK_URL"), data=self.data, headers=self.header
            )
            response.raise_for_status()
            if response.status_code == 200:
                token_data = response.json()
                return token_data["access_token"]
            else:
                return None

        except requests.exceptions.HTTPError as httperr:
            print("Http Error", httperr)
            return None
        except requests.exceptions.RequestException as err:
            print("Request failed", err)
            return None

    def auth_client(self):
        return AuthClientPassword(
            username=os.getenv("KEY_CLOAK_USERNAME"),
            password=os.getenv("KEY_CLOAK_PASS"),
            client_id=os.getenv("KEY_CLIENT"),
            scope="openid offline_access",
        )

    def auth_client_bearer(self):
        return AuthBearerToken(self.get_token())

    def weaviate_client(self):
        client = weaviate.WeaviateClient(
            connection_params=ConnectionParams(
                http=ProtocolParams(host="weaviate", port=8080, secure=False),
                grpc=ProtocolParams(host="weaviate", port=50051, secure=False),
            ),
            auth_client_secret=self.auth_client_bearer(),
        )

        client.connect()
        return client

    def weaviate_init(self):

        try:
            self.client_noauth.connect()
            return Weaviate(
                client=self.client_noauth,
                index_name="PayagLegalDocs",
                text_key="legal_text",
                embedding=self.embedding,
            )

        finally:
            self.client_noauth.close()

    def splitted_docs(self, docs: list[Document]):
        return RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=128, separators=["\n\n", "\n", ".", " ", ""]
        ).split_documents(documents=docs)

    # @retry(
    #     retry=retry_if_exception_type(
    #         (UnexpectedStatusCodeException, requests.exceptions.RequestException)
    #     ),
    #     wait=wait_exponential(multiplier=2, min=2, max=30),
    #     reraise=True,
    # )
    def vector_store(self, docs: list[Document]):
        print("ingesting vectors into weaviate...")
        return self.weviate_vector.from_documents(
            documents=self.splitted_docs(docs), embedding=self.embedding
        )

    def vector_retriever(self):
        return self.weaviate_init().as_retriever()

    def store_name(self):
        print("Weaviate Store")

    @classmethod
    def print_test(cls):
        print(cls().get_token())
