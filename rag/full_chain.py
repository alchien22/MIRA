from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate

from models.inference_api import get_model
from .rag_chain import make_rag_chain

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
        {"role": "system", "content": "You are a medical expert AI assistant called MIRA. Provide concise responses."},
        {
            "role": "user", 
            "context": "{context}",
            "content": "{question}"
        }
    ]
    prompt = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
    return make_rag_chain(model, retriever, rag_prompt=prompt, tokenizer=tokenizer, confidence_method=confidence_method)


def is_ehr_query(query):
    ehr_keywords = ["patient", "record", "ehr", "diagnosis", "treatment history", "lab results", "discharge", "visit", "history"]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in ehr_keywords)


def ask_question(chain, query):
    print(f"Debug: ask_question received query = {query}", flush=True)

    use_rag = is_ehr_query(query)
    print(f"Debug: use_rag = {use_rag}", flush=True)

    try:
        response_data = chain.invoke({"question": query, "use_rag": use_rag}, config={"configurable": {"session_id": "foo"}})
        print(f"Debug: Response data received = {response_data}", flush=True)
    except Exception as e:
        print(f"ERROR during chain.invoke(): {e}", flush=True)
        raise

    return response_data

# from .memory import create_memory_chain
# chain = create_memory_chain(model, rag_chain, chat_memory)