from abc import ABC, abstractmethod
from langchain_core.documents import Document


class VectorStore(ABC):

    @abstractmethod
    def vector_store(self, docs: list[Document]):
        pass

    @abstractmethod
    def store_name(self):
        pass
