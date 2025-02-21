from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate

from models.inference_api import get_model
from memory import create_memory_chain
from rag_chain import make_rag_chain


def create_full_chain(retriever, chat_memory=ChatMessageHistory(), confidence_method="entropy"):
    model, tokenizer = get_model()
    
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

    rag_chain = make_rag_chain(model, retriever, rag_prompt=prompt, tokenizer=tokenizer, confidence_method=confidence_method)
    chain = create_memory_chain(model, rag_chain, chat_memory)
    
    return chain


def is_ehr_query(query):
    """Determine if the query is related to EHR records."""
    ehr_keywords = ["patient", "record", "ehr", "diagnosis", "treatment history", "lab results", "discharge", "visit", "history"]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in ehr_keywords)


def ask_question(chain, query):
    use_rag = is_ehr_query(query)

    response_data = chain.invoke(
        {"question": query, "use_rag": use_rag},
        config={"configurable": {"session_id": "foo"}}
    )
    if "response" in response_data and "confidence" in response_data:
        return response_data

    return {"response": str(response_data), "confidence": 0.0}
