import os
import streamlit as st

from dotenv import load_dotenv

from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from retrieval.ensemble import ensemble_retriever_from_docs
from rag.full_chain import create_full_chain, ask_question
from retrieval.local_loader import load_txt_files

load_dotenv()

MODEL_ID = "ProbeMedicalYonseiMAILab/medllama3-v20"
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
                response_data = ask_question(qa, prompt)
                response = response_data.get("response", "I couldn't generate a response.")
                confidence = response_data.get("confidence", 0.0)

                # if confidence > 0.8:
                #     confidence_text = f"🟢 **Confidence Score:** {confidence:.2f} (High Confidence)"
                # elif confidence > 0.5:
                #     confidence_text = f"🟠 **Confidence Score:** {confidence:.2f} (Medium Confidence)"
                # else:
                #     confidence_text = f"🔴 **Confidence Score:** {confidence:.2f} (Low Confidence - Double Check Output)"

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
    chain = create_full_chain(ensemble_retriever, chat_memory=StreamlitChatMessageHistory(key="langchain_messages"), confidence_method="entropy")
    return chain


def run():
    if "langchain_messages" not in st.session_state:
        st.session_state["langchain_messages"] = []
        
    chain = get_chain()
    st.subheader("Ask me anything about a patient's medical history, symptoms, or treatment!")
    show_ui(chain, "What would you like to know?")

run()
