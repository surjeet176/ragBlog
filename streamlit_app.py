import streamlit as st
from dotenv import load_dotenv
from app import get_faq_chain

load_dotenv() 


def format_input(inputs):
    return f"Question: {inputs['question']}"


print("\n-----outer code run--------")
# chain = get_faq_chain()


st.title("FAQ chat")

if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "model" not in st.session_state:
    st.session_state["model"] = get_faq_chain()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter you Query ?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.session_state.model.invoke({
            "question" : f"{prompt}",
        })
        st.markdown(response.content)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response.content})