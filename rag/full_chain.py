from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages.base import BaseMessage

from models.inference_api import get_model, generate_response_with_latents
from models.confidence import compute_confidence_score
from models.critic import generate_critic_score
from models.prompt import MIRA_PROMPT

import torch


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


def remove_duplicates(documents):
    """ Removes duplicate documents from a list of documents."""
    seen = set()
    unique_docs = []
    for doc in documents:
        if doc.page_content not in seen:
            unique_docs.append(doc)
            seen.add(doc.page_content)
            
    return unique_docs


def make_rag_chain(model, rag_prompt, tokenizer):
    """Create a RAG chain with a retriever and a model."""
    def retrieve_with_confidence(input):
        question = get_question(input)
        # print(f"Debug: retrieve_with_confidence received query = {question}", flush=True)

        use_rag = input.get("use_rag", True)
        context = input.get("context", '')

        full_prompt = str(rag_prompt.format(context=context, question=question))
        print(full_prompt)

        response, response_latents, base_confidence = generate_response_with_latents(model, tokenizer, full_prompt)
        print(response)

        if use_rag and context:
            # Generate latents of the context
            _, retrieved_latents, _ = generate_response_with_latents(model, tokenizer, context)
            # Get factuality score
            factuality_score = generate_critic_score(model, tokenizer, critic_type="factuality", question=question, retrieved_info=context, generated_answer=response, model_type='gpt')
            # Get consistency score
            consistency_score = generate_critic_score(model, tokenizer, critic_type="consistency", retrieved_info=context, generated_answer=response, model_type='gpt')
            # Compute confidence score
            confidence_score = compute_confidence_score(base_confidence, response_latents, retrieved_latents, use_rag, factuality_score, consistency_score)
        else:
            confidence_score = base_confidence

        return {"response": response, "confidence": confidence_score}

    return RunnableLambda(retrieve_with_confidence)


def create_full_chain():
    model, tokenizer = get_model()
    return make_rag_chain(model, rag_prompt=MIRA_PROMPT, tokenizer=tokenizer)


def is_ehr_query(query):
    ehr_keywords = ["ehr", "history", "record", "patient", "visit", "discharge"]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in ehr_keywords)


def ask_question(chain, retriever, query):
    # print(f"Debug: ask_question received query = {query}", flush=True)

    use_rag = True #is_ehr_query(query)
    docs = []
    evidence = ''
    
    if use_rag:
        # print('Retrieving')
        torch.cuda.empty_cache()
        docs = retriever.invoke(query)
        docs = remove_duplicates(docs)
        print(f'Found {len(docs)} docs')
        if not docs:
            return {"response": "I couldn't find any relevant documents."}
        evidence = format_docs(docs)

    response_data = chain.invoke({"question": query, "use_rag": use_rag, "context": evidence}, config={"configurable": {"session_id": "foo"}})
    # print(f"Debug: Response data received = {response_data}", flush=True)

    return {"response": response_data["response"], "docs": docs, "confidence": response_data["confidence"]}