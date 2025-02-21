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
    initial_sidebar_state="expanded",
    )

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
                response_data = ask_question(qa, prompt)
                response = response_data.get("response", "I couldn't generate a response.")
                confidence = response_data.get("confidence", 0.0)
                st.markdown(response)
                st.write(f"**Confidence Score:** {confidence:.2f}")
                
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)


@st.cache_resource
def get_retriever():
    docs = load_txt_files()
    return ensemble_retriever_from_docs(docs)


def get_chain():
    ensemble_retriever = get_retriever()
    chain = create_full_chain(ensemble_retriever, chat_memory=StreamlitChatMessageHistory(key="langchain_messages"))
    return chain


def get_secret_or_input():
    if 'HUGGINGFACEHUB_API_TOKEN' in st.secrets:
        secret_value = st.secrets['HUGGINGFACEHUB_API_TOKEN']
    return secret_value


def run():
    ready = True

    huggingfacehub_api_token = st.session_state.get("HUGGINGFACEHUB_API_TOKEN")

    if not huggingfacehub_api_token:
        huggingfacehub_api_token = get_secret_or_input()

    if not huggingfacehub_api_token:
        st.warning("Missing HUGGINGFACEHUB_API_TOKEN")
        ready = False

    if ready:
        chain = get_chain()

        with st.sidebar:
            st.metric("Confidence", "High", "99%") # TODO: Write function that updates this

        st.title("Hi, I am MIRA! Your EHR Assistant🤖")
        st.subheader("Ask me about a patient's medical history!")
        show_ui(chain, "What would you like to know?")

    else:
        st.stop()

run()
