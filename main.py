"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
from langchain import LlamaCpp
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import FAISS

# Load the LangChain.
model_path = "ggml-vicuna-7b-4bit-rev1.bin"
embeddings = LlamaCppEmbeddings(model_path=model_path)
db = FAISS.load_local("faiss_index", embeddings)
chain = RetrievalQAWithSourcesChain.from_llm(llm=LlamaCpp(temperature=0, model_path=model_path), retriever=db.as_retriever())

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Blendle Notion QA Bot", page_icon=":robot:")
st.header("Blendle Notion QA Bot")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text


user_input = get_text()

if user_input:
    result = chain({"question": user_input})
    output = f"Answer: {result['answer']}\nSources: {result['sources']}"

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
