from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages.base import BaseMessage

def find_similar(vs, query):
    docs = vs.similarity_search(query)
    return docs


def format_docs(docs):
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


def make_rag_chain(model, retriever, rag_prompt):
    """Create a RAG chain with a retriever and a model.
    
    Args:
        model (InferenceClient): The model to use for generation.
        retriever (Callable): The retriever to use for retrieving documents.
        rag_prompt (ChatPromptTemplate): The prompt to use for the RAG chain.

    Returns:
        Chain: The RAG chain.
    """

    rag_chain = (
            {
                "context": RunnableLambda(get_question) | retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | rag_prompt
            | model
    )

    return rag_chain
