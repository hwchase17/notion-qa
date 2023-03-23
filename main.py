"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import faiss
from langchain.chains import VectorDBQAWithSourcesChain
import pickle
import os
from langchain.chat_models import ChatOpenAI

# Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)
openai_api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key
store.index = index
model_name = 'gpt-3.5-turbo'
chain = VectorDBQAWithSourcesChain.from_llm(llm=ChatOpenAI(model_name=model_name, openai_api_key=openai_api_key, temperature=0), vectorstore=store)


# From here down is all the StreamLit UI.
st.set_page_config(page_title="Xingfan's Notion QA Bot", page_icon=":robot:")
st.header("Xingfan's Notion QA Bot")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "What can you do for me?", key="input")
    return input_text


user_input = get_text()

if user_input:
    result = chain({"question": user_input})
    output = f"Answer: {result['answer']}\nSources: {result['sources']}"

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(avatar_style="miniavs", message=st.session_state["generated"][i], key=str(i))
        message(avatar_style="shapes", message=st.session_state["past"][i], is_user=True, key=str(i) + "_user")
