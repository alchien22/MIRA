from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages.base import BaseMessage

from models.confidence import compute_confidence_score, batch_extract_latents
from models.inference_api import generate_response_with_latents

import streamlit as st

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
        use_rag = input.get("use_rag", True)
        context = ''
        question = get_question(input)
        print(f"Debug: retrieve_with_confidence received query = {question}", flush=True)

        if use_rag:
            retrieved_docs = retriever(question)
            if retrieved_docs:
                context = format_docs(retrieved_docs)

        full_prompt = str(rag_prompt.format(context=context, question=question))

        response, response_latents, base_confidence = generate_response_with_latents(model, tokenizer, full_prompt, confidence_method)

        if use_rag and context:
            retrieved_texts_batch = [doc.page_content for doc in retrieved_docs]
            retrieved_latents = batch_extract_latents(model, tokenizer, retrieved_texts_batch, confidence_method)
            confidence_score = compute_confidence_score(response_latents, retrieved_latents, base_confidence, use_rag)
        else:
            confidence_score = base_confidence

        return {"response": response, "confidence": confidence_score}

    return RunnableLambda(retrieve_with_confidence)
