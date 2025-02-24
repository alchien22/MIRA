import os
import streamlit as st

from dotenv import load_dotenv

from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from vector_store import VectorStore
from full_chain import create_full_chain, ask_question
from local_loader import load_csv_files

load_dotenv()

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
os.environ["MODEL_ID"] = MODEL_ID

st.set_page_config(
    page_title="MIRA",
    page_icon="💉",
    layout="centered",
    initial_sidebar_state="expanded",
    )

def show_ui(qa, retriever, prompt_to_user="How may I help you?"):
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
                response_data = ask_question(qa, prompt, retriever)

                response = response_data.get("response", "I couldn't generate a response.")
                st.markdown(response.content)

                evidence = response_data.get("docs", [])

                # Display evidence. Remove this and confidence for 1 scenario run
                with st.expander("View Evidence"):
                    for i, doc in enumerate(evidence):
                        st.markdown(f"### Document {i+1}")
                        st.code(doc.page_content, language="plaintext")
                        st.code(f"Note ID: {doc.metadata['note_id']}", language="json")
                        st.code(f"Source: {doc.metadata['source']}", language="json")

                        st.divider()

                confidence = response_data.get("confidence", 0.0)
                st.write(f"**Confidence Score:** {confidence:.2f}")
                
        message = {"role": "assistant", "content": response.content}
        st.session_state.messages.append(message)


@st.cache_resource
def get_retriever():
    vector_store = VectorStore()
    docs = load_csv_files()
    vector_store.create_vector_db(docs)
    
    return vector_store.get_retriever()


def get_chain():
    retriever = get_retriever()
    chain = create_full_chain(chat_memory=StreamlitChatMessageHistory(key="langchain_messages"))
    return chain, retriever


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
        chain, retriever = get_chain()

        st.title("Hi, I am MIRA! Your EHR Assistant🤖")
        st.subheader("Ask me about a patient's medical history!")
        show_ui(chain, retriever, "What would you like to know?")

    else:
        st.stop()

run()
