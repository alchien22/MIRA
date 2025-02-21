from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate

from inference_api import get_model
from memory import create_memory_chain
from rag_chain import make_rag_chain


def create_full_chain(retriever, chat_memory=ChatMessageHistory()):
    model = get_model()
    
    system_prompt = """You are a medical expert AI assistant called MIRA.
    
    Use the following context and the users' chat history to help the user:
    If you don't know the answer, just say that you don't know. 
    
    Context: {context}
    
    Question: """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    rag_chain = make_rag_chain(model, retriever, rag_prompt=prompt)
    chain = create_memory_chain(model, rag_chain, chat_memory)
    
    return chain


def ask_question(chain, query):
    response_data = chain.invoke(
        {"question": query},
        config={"configurable": {"session_id": "foo"}}
    )
    if "response" in response_data and "confidence" in response_data:
        return response_data

    return {"response": str(response_data), "confidence": 0.0}
