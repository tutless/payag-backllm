import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain import hub
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

# from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from payag_generative.qdrant_store import QdrantStore
from payag_generative.vector_store import VectorStore
from payag_generative.pinecone_store import PineconeStore
from payag_generative.chroma_store import ChromaStore
from payag_generative.chat_history_trim import TrimmedChatMessageHistory


load_dotenv()


class GenerativeCore:
    def __init__(self, vstore: VectorStore[QdrantStore]):
        self.vstore = vstore
        self.repharase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
        self.store = {}
        self.chat_model = ChatNVIDIA(
            model="meta/llama-3.1-8b-instruct", temperature=0.1, max_token=1024
        )

    # def llm_model(self):
    #     return ChatNVIDIA(
    #         model="meta/llama-3.1-405b-instruct", temperature=0.1, max_token=1024
    #     )

    def main_prompt(self):

        system_prompt = """ You are an advanced AI assistant designed to help lawyers, law students, and legal professionals find information, gain insights, and make better decisions efficiently.
          Your primary role is to swiftly extract, analyze, and present valuable legal insights from a diverse dataset consisting of legal documents, case law, statutes, regulations, contracts, images, files, and other relevant legal data.
          Key functionalities include:
             * Accelerating legal content creation to enhance productivity in research, drafting, and analysis.
             * Extracting key legal insights to provide a comprehensive understanding of complex legal issues.
             * Providing quick, accurate, and relevant answers to legal questions based on the dataset.
             * Ensuring all responses include citations, citation snippets, and references for transparency and credibility, 
               adhering to legal research best practices.
          Your responses should be precise, well-structured, and optimized for clarity,offering actionable insights that enable informed legal decision-making. 
          Where applicable, account for jurisdictional variations, legal principles, and precedents to ensure accuracy and relevance
        {context}"""

        return ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{input}")]
        )

    def contextual_prompt(self):
        return ChatPromptTemplate.from_messages(
            [
                ("system", self.repharase_prompt.template),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

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

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = TrimmedChatMessageHistory(max_msg=3)
        return self.store[session_id]

    def conversational_rag_chain(self):
        return RunnableWithMessageHistory(
            self.rag_chain(),
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    @classmethod
    def answer(cls, query: str):
        store = QdrantStore()
        core = cls(store)

        try:
            final_answer = core.conversational_rag_chain().invoke(
                {"input": query},
                config={"configurable": {"session_id": "payag_session"}},
            )
            return final_answer["answer"]
        finally:
            torch.cuda.empty_cache()


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
