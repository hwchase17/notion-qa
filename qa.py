"""Ask a question to the notion database."""
import faiss
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
import pickle
import argparse
from pprint import pprint

parser = argparse.ArgumentParser(description='Ask a question to the notion DB.')
parser.add_argument('question', type=str, help='The question to ask the notion DB')
args = parser.parse_args()

# Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index
chain = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(temperature=0), chain_type="stuff", retriever=store.as_retriever())
result = chain({"question": args.question})
pprint(f"Answer: {result['answer']}")
pprint(f"Sources: {result['sources']}")
