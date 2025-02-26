from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages.base import BaseMessage

from models.inference_api import get_model, generate_response_with_latents
from models.confidence import compute_confidence_score

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


def make_rag_chain(model, retriever, rag_prompt, tokenizer, confidence_method="entropy"):
    """Create a RAG chain with a retriever and a model."""
    def retrieve_with_confidence(input):
        question = get_question(input)
        print(f"Debug: retrieve_with_confidence received query = {question}", flush=True)

        use_rag = input.get("use_rag", True)
        context = input.get("context", '')

        full_prompt = str(rag_prompt.format(context=context, question=question))

        response, response_latents, base_confidence = generate_response_with_latents(model, tokenizer, full_prompt, confidence_method)

        if use_rag and context:
            _, retrieved_latents, _ = generate_response_with_latents(model, tokenizer, context, confidence_method)
            confidence_score = compute_confidence_score(response_latents, retrieved_latents, base_confidence, use_rag)
        else:
            confidence_score = base_confidence

        return {"response": response, "confidence": confidence_score}

    return RunnableLambda(retrieve_with_confidence)


def create_full_chain(retriever, chat_memory=ChatMessageHistory(), confidence_method="entropy"):
    model, tokenizer = get_model()
    tokenizer.chat_template = """
{% for message in messages %}
{% if loop.first %}
<|begin_of_text|>
{% endif %}
{% if message['role'] == 'system' %}
<|start_header_id|>system<|end_header_id|>
{{ message['content'] }}<|eot_id|>
{% elif message['role'] == 'user' %}
<|start_header_id|>user<|end_header_id|>
{% if message['context'] %}
Context:
{{ message['context'] }}
{% endif %}

Question:
{{ message['content'] }}<|eot_id|>
{% elif message['role'] == 'assistant' %}
<|start_header_id|>assistant<|end_header_id|>
{{ message['content'] }}<|eot_id|>
{% endif %}
{% if loop.last and add_generation_prompt %}
<|start_header_id|>assistant<|end_header_id|>
{% endif %}
{% endfor %}
    """

    chat = [
        {"role": "system", "content": (
            "You are a medical expert AI assistant called MIRA. "
            "Provide short and concise responses in under 200 tokens. "
            "When asked about a patient's medical records, you should provide a response based on the information given below.")
        },
        {
            "role": "user", 
            "context": "{context}",
            "content": "{question}"
        }
    ]
    prompt = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
    return make_rag_chain(model, retriever, rag_prompt=prompt, tokenizer=tokenizer, confidence_method=confidence_method)


def is_ehr_query(query):
    ehr_keywords = ["ehr", "history", "record", "patient", "visit", "discharge"]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in ehr_keywords)


def ask_question(chain, retriever, query):
    print(f"Debug: ask_question received query = {query}", flush=True)

    use_rag = is_ehr_query(query)
    docs = []
    evidence = ''
    
    if use_rag:
        print('Retrieving')
        torch.cuda.empty_cache()
        docs = retriever.invoke(query)
        docs = remove_duplicates(docs)
        print(f'Found {len(docs)} docs')
        if not docs:
            return {"response": "I couldn't find any relevant documents."}
        evidence = format_docs(docs)

    response_data = chain.invoke({"question": query, "use_rag": use_rag, "context": evidence}, config={"configurable": {"session_id": "foo"}})
    print(f"Debug: Response data received = {response_data}", flush=True)

    return {"response": response_data["response"], "docs": docs, "confidence": response_data["confidence"]}

# from .memory import create_memory_chain
# chain = create_memory_chain(model, rag_chain, chat_memory)