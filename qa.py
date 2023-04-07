"""Ask a question to the notion database."""
from langchain import LlamaCpp
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import FAISS
import argparse

parser = argparse.ArgumentParser(description='Ask a question to the notion DB.')
parser.add_argument('question', type=str, help='The question to ask the notion DB')
args = parser.parse_args()

model_path = "ggml-vicuna-7b-4bit-rev1.bin"
embeddings = LlamaCppEmbeddings(model_path=model_path)
db = FAISS.load_local("faiss_index", embeddings)

chain = RetrievalQAWithSourcesChain.from_llm(llm=LlamaCpp(temperature=0, model_path=model_path), retriever=db.as_retriever())
result = chain({"question": args.question})
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
