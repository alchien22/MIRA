from typing import List, Iterable, Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables.history import RunnableWithMessageHistory


def create_memory_chain(llm, base_chain, chat_memory):
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    runnable = contextualize_q_prompt | llm | base_chain

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        return chat_memory

    with_message_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    return with_message_history


class SimpleTextRetriever(BaseRetriever):
    docs: List[Document]
    """Documents."""

    @classmethod
    def from_texts(
            cls,
            texts: Iterable[str],
            **kwargs: Any,
    ):
        docs = [Document(page_content=t) for t in texts]
        return cls(docs=docs, **kwargs)

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self.docs
