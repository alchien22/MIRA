import os
import streamlit as st

from dotenv import load_dotenv

from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from retrieval.vector_store import VectorStore
from rag.full_chain import create_full_chain, ask_question
from retrieval.local_loader import load_csv_files

load_dotenv()

MODEL_ID = "johnsnowlabs/JSL-MedLlama-3-8B-v2.0"
os.environ["MODEL_ID"] = MODEL_ID

st.set_page_config(
    page_title="MIRA",
    page_icon="💉",
    layout="centered",
    initial_sidebar_state="collapsed"
    )
st.title("MIRA 🩺🤖")

def show_ui(qa, retriever, prompt_to_user="How may I help you today?"):
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
                response_data = ask_question(qa, retriever, prompt)
                response = response_data.get("response", "I couldn't generate a response.")
                confidence = response_data.get("confidence", 0.0)

                st.markdown(response)
                st.write(f"**Confidence:** {confidence:.2f}")

                evidence = response_data.get("docs", [])
                if evidence:
                    with st.expander("View Evidence"):
                    for i, doc in enumerate(evidence):
                        st.markdown(f"### Document {i+1}")
                        st.code(doc.page_content, language="plaintext")
                        st.code(f"Note ID: {doc.metadata['note_id']}", language="json")
                        st.code(f"Source: {doc.metadata['source']}", language="json")
                        st.divider()
                
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)


@st.cache_resource
def get_retriever():
    vector_store = VectorStore()
    docs = load_csv_files()
    vector_store.create_vector_db(docs)
    return vector_store.get_retriever()


def get_chain():
    retriever = get_retriever()
    chain = create_full_chain(retriever, chat_memory=StreamlitChatMessageHistory(key="langchain_messages"), confidence_method="entropy")
    return chain, retriever


def run():
    if "langchain_messages" not in st.session_state:
        st.session_state["langchain_messages"] = []
        
    chain, retriever = get_chain()
    st.subheader("Ask me anything about a patient's medical history, symptoms, or treatment!")
    show_ui(chain, retriever, "Hi, I'm MIRA, your personal medical assistant! What would you like to know?")

run()
