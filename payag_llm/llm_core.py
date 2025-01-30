from payag_llm.vstore import VectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain


class LLMCore(VectorStore):
    def __init__(self, query: str):
        super().__init__()
        self.repharese_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
        self.retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        self.query = query

    def run_llm(self):
        stuff_document_chain = create_stuff_documents_chain(
            self.chat, self.retrieval_qa_chat_prompt
        )
        history_aware_retriever = create_history_aware_retriever(
            llm=self.chat,
            retriever=self.vectore_store.as_retriever(),
            prompt=self.repharese_prompt,
        )
        qa = create_retrieval_chain(
            retriever=history_aware_retriever, combine_docs_chain=stuff_document_chain
        )
        result = qa.invoke(input={"input": self.query})
        return result["answer"]

    @classmethod
    def answer(cls, query: str):
        return cls(query).run_llm()
