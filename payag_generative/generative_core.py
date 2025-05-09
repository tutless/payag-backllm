import os
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain import hub
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from payag_generative.qdrant_store import QdrantStore
from payag_generative.vector_store import VectorStore
from payag_generative.pinecone_store import PineconeStore
from payag_generative.chroma_store import ChromaStore
from payag_generative.chat_history_trim import TrimmedChatMessageHistory
from payag_generative.llm_pipeline import LLModelPipeline
from langchain.memory import ConversationBufferWindowMemory


load_dotenv()


class GenerativeCore:
    def __init__(self, vstore: VectorStore[QdrantStore]):
        self.vstore = vstore
        self.repharase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
        self.store = {}
        self.chat_model = ChatNVIDIA(
            model="meta/llama-3.1-8b-instruct", temperature=0.1, max_tokens=1024
        )
        # self.chat_model = LLModelPipeline.load_pipeline(
        #     model_id="meta-llama/Llama-3.1-8B-Instruct"
        # )

    def main_prompt(self):

        system_prompt = """ You are a sophisticated AI assistant tailored to support lawyers,law students,law practitioners,and legal scholars in their professional tasks. 
        Your core purpose is to extract, analyze, and synthesize legal information swiftly and accurately from a wide variety of legal sources, including but not limited to case law, statutes, regulations, legal precedents, contracts, and evidence.
        Your responses should be articulate, structured, and clear, providing actionable legal insights that foster informed decision-making. Each interaction should be aimed at promoting efficiency and accuracy in legal work, considering the specific context and requirement of the query while maintaining professional integrity
        {context}"""

        return ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{input}")]
        )

    def contextual_prompt(self):

        CONTXTUAL_Q_PROMPT = """Given a chat history and the latest user question 
            which might reference context in the chat history, 
            formulate a standalone question which can be understood 
            without the chat history. Do NOT answer the question, 
            just reformulate it if needed and otherwise return it as is."""
        return ChatPromptTemplate.from_messages(
            [
                ("system", CONTXTUAL_Q_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    # return ChatPromptTemplate.from_messages(
    #     [
    #         ("system", self.repharase_prompt.template),
    #         MessagesPlaceholder("chat_history"),
    #         ("human", "{input}"),
    #     ]
    # )

    def retriever(self):
        return self.vstore.vector_retriever()

    def history_aware_retriever(self):
        return create_history_aware_retriever(
            llm=self.chat_model,
            retriever=self.retriever(),
            prompt=self.contextual_prompt(),
        )

    def rag_chain(self):
        question_answer_chain = create_stuff_documents_chain(
            llm=self.chat_model, prompt=self.main_prompt()
        )
        return create_retrieval_chain(
            retriever=self.history_aware_retriever(),
            combine_docs_chain=question_answer_chain,
        )

    # def get_session_history(self, session_id: str) -> ConversationBufferWindowMemory:
    #     if session_id not in self.store:
    #         self.store[session_id] = ConversationBufferWindowMemory(
    #             k=5, return_messages=True
    #         )
    #     return self.store[session_id]

    def get_session_history(self, session_id) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def conversational_rag_chain(self):
        # return RunnableWithMessageHistory(
        #     self.rag_chain(),
        #     lambda _: self.get_session_history(session_id=session_id).chat_memory,
        #     input_messages_key="input",
        #     history_messages_key="chat_history",
        #     output_messages_key="answer",
        # )
        return RunnableWithMessageHistory(
            self.rag_chain(),
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    @staticmethod
    def freeup_memory():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        gc.collect()

    # @classmethod
    # def answer(cls, query: str):
    #     store = QdrantStore()
    #     core = cls(store)
    #     try:
    #         final_answer = core.conversational_rag_chain(
    #             session_id="payag_session_1"
    #         ).invoke(
    #             {"input": query},
    #         )
    #         return final_answer["answer"]
    #     finally:
    #         del core
    #         cls.freeup_memory()

    @classmethod
    def answer(cls, query: str):
        store = QdrantStore()
        core = cls(store)
        try:
            final_answer = core.conversational_rag_chain().invoke(
                {"input": query},
                config={"configurable": {"session_id": "payag_session_1"}},
            )
            return final_answer["answer"]
        finally:
            del core
            cls.freeup_memory()


# if __name__ == "__main__":
#     chroma = ChromaStore()
#     gen_core = GenerativeCore(vstore=chroma)
#     ai_answer = gen_core.conversational_rag_chain().invoke(
#         {
#             "input": "The judge found the defendant guilty of frustrated rape. Was the verdict correct?"
#         },
#         config={"configurable": {"session_id": "sess"}},
#     )

#     print(ai_answer["answer"])


# if __name__ == "__main__":
#     pinecone = PineconeStore()
#     gen_core = GenerativeCore(vstore=pinecone)
#     final_answer = gen_core.conversational_rag_chain().invoke(
#         {"input": "what is the prescriptive period of cyber libel?"},
#         config={"configurable": {"session_id": "payag"}},
#     )

#     print(final_answer["answer"])
