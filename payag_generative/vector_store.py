from abc import ABC, abstractmethod
from langchain_core.documents import Document
from typing import TypeVar, Generic

T = TypeVar("T")


class VectorStore(ABC, Generic[T]):

    @abstractmethod
    def vector_store(self, docs: list[Document]) -> T:
        pass

    @abstractmethod
    def vector_retriever(self) -> T:
        pass

    @abstractmethod
    def store_name(self):
        pass
