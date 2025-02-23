import random

from langchain.globals import set_debug

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate

from inference_api import get_model

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages.base import BaseMessage

set_debug(True)

def format_docs(docs):
    """Joins the content of the documents with a newline separator into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def get_question(input):
    if not input:
        return None
    elif isinstance(input,str):
        return input
    elif isinstance(input,dict) and 'question' in input:
        return input['question']
    elif isinstance(input,BaseMessage):
        return input.content
    else:
        raise Exception("string or dict with 'question' key expected as RAG chain input.")


def make_rag_chain(model, rag_prompt):
    """Create a RAG chain with a retriever and a model.
    
    Args:
        model (InferenceClient): The model to use for generation.
        retriever (Callable): The retriever to use for retrieving documents.
        rag_prompt (ChatPromptTemplate): The prompt to use for the RAG chain.

    Returns:
        rag_chain: The RAG chain.
        retrieved_docs: The retrieved documents.
    """

    rag_chain = (
            {   
                "context": RunnablePassthrough(),
                "question": RunnablePassthrough()
            }
            | rag_prompt
            | model
    )

    return rag_chain


def create_full_chain(chat_memory=ChatMessageHistory()):
    model = get_model()
    
    system_prompt = """You are a medical expert AI assistant called MIRA. You are given a set of patient records.

    When asked about a patient's medical records, you should provide a response based on the information given below.

    Context: """

    prompt = ChatPromptTemplate(
        [
            ("system", system_prompt),
            ("system", "{context}"),
            ("human", "Question: {question}"),
        ]
    )

    rag_chain = make_rag_chain(model, rag_prompt=prompt)
    
    return rag_chain


def ask_question(chain, query, retriever):
    """ Invokes the retriever and the LLM chain to generate a response to the query given a context.

    Args:
        chain (Runnable): The RAG chain.
        query (str): The query to ask the model.
        retriever (Callable): The retriever to use for retrieving documents.

    Returns:
        {"response": str, "docs": List[Document], "confidence": float}: The response, the retrieved documents along with metadata, and the confidence score.
    """
    docs = retriever.invoke(query)

    if not docs:
        return {"response": "I couldn't find any relevant documents."}
    evidence = format_docs(docs)

    response_data = chain.invoke(
        {
            "context": evidence,
            "question": query
        },
        config={"configurable": {"session_id": "foo"}}
    )
    
    if "response" in response_data and "confidence" in response_data:
        return response_data
    
    # random confidence score
    conf = random.uniform(0.5, 1.0)

    return {"response": response_data, "docs": docs, "confidence": conf} # TODO: Make confidence score more meaningful

