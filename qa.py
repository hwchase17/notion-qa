"""Ask a question to the notion database."""
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
import argparse
import faiss
import os
import pickle
import streamlit as st

openai_api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key
parser = argparse.ArgumentParser(description='Ask a question to the notion DB.')
parser.add_argument('question', type=str, help='The question to ask the notion DB')
args = parser.parse_args()
model_name = 'gpt-3.5-turbo'

# Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index
chain = VectorDBQAWithSourcesChain.from_llm(llm=ChatOpenAI(model_name=model_name, openai_api_key=openai_api_key, temperature=0), vectorstore=store)
result = chain({"question": args.question})
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
