from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages.base import BaseMessage

from models.confidence import compute_confidence_score, batch_extract_latents
from models.inference_api import generate_response_with_latents

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


def make_rag_chain(model, retriever, rag_prompt, tokenizer, confidence_method="entropy"):
    """Create a RAG chain with a retriever and a model.
    Args:
        model (InferenceClient): The model to use for generation.
        retriever (Callable): The retriever to use for retrieving documents.
        rag_prompt (ChatPromptTemplate): The prompt to use for the RAG chain.

    Returns:
        Chain: The RAG chain.
    """
    def retrieve_with_confidence(input):
        query = get_question(input)

        use_rag = input.get("use_rag", True)
        # Not an EHR query: generate response without retrieval
        if not use_rag:
            response, _, base_confidence = generate_response_with_latents(model, tokenizer, query, confidence_method)
            return {"response": response, "confidence": base_confidence} 

        retrieved_docs = retriever(query)
        retrieved_texts = format_docs(retrieved_docs)
        full_prompt = rag_prompt.format(context=retrieved_texts, question=query)

        response, response_latents, base_confidence = generate_response_with_latents(model, tokenizer, full_prompt, confidence_method)
        
        # Extract latents for retrieved documents
        retrieved_texts_batch = [doc.page_content for doc in retrieved_docs]
        retrieved_latents = batch_extract_latents(model, tokenizer, retrieved_texts_batch, confidence_method)

        confidence_score = compute_confidence_score(response_latents, retrieved_latents, base_confidence, use_rag)

        return {"response": response, "confidence": confidence_score}
    
    rag_chain = (
            {
                "context": RunnableLambda(get_question) | retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | RunnableLambda(retrieve_with_confidence)
    )

    return rag_chain
