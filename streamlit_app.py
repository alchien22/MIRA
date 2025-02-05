import os
import streamlit as st

from dotenv import load_dotenv

from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from ensemble import ensemble_retriever_from_docs
from full_chain import create_full_chain, ask_question
from local_loader import load_txt_files

load_dotenv()

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
os.environ["MODEL_ID"] = MODEL_ID

st.set_page_config(
    page_title="MIRA",
    page_icon="💉",
    layout="centered",
    initial_sidebar_state="collapsed"
    )
st.title("Hi, I am MIRA! Your EHR Assistant 🩺🤖")

def show_ui(qa, prompt_to_user="How may I help you?"):
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": prompt_to_user}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ask_question(qa, prompt)
                st.markdown(response.content)
                
        message = {"role": "assistant", "content": response.content}
        st.session_state.messages.append(message)


@st.cache_resource
def get_retriever():
    docs = load_txt_files()
    return ensemble_retriever_from_docs(docs)


def get_chain():
    ensemble_retriever = get_retriever()
    chain = create_full_chain(ensemble_retriever, chat_memory=StreamlitChatMessageHistory(key="langchain_messages"))
    return chain


def get_secret_or_input(secret_key, secret_name, info_link=None):
    if secret_key in st.secrets:
        st.write("Found %s secret" % secret_key)
        secret_value = st.secrets[secret_key]
    else:
        st.write(f"Please provide your {secret_name}")
        secret_value = st.text_input(secret_name, key=f"input_{secret_key}", type="password")
        if secret_value:
            st.session_state[secret_key] = secret_value
        if info_link:
            st.markdown(f"[Get an {secret_name}]({info_link})")
    return secret_value


def run():
    ready = True

    huggingfacehub_api_token = st.session_state.get("HUGGINGFACEHUB_API_TOKEN")

    with st.sidebar:
        if not huggingfacehub_api_token:
            huggingfacehub_api_token = get_secret_or_input('HUGGINGFACEHUB_API_TOKEN', "HuggingFace Hub API Token",
                                                           info_link="https://huggingface.co/docs/huggingface_hub/main/en/quick-start#authentication")

    if not huggingfacehub_api_token:
        st.warning("Missing HUGGINGFACEHUB_API_TOKEN")
        ready = False

    if ready:
        chain = get_chain()
        st.subheader("Ask me anything about a patient's medical history, symptoms, or treatment!")
        show_ui(chain, "What would you like to know?")
    else:
        st.stop()

run()
