"""This is the logic for ingesting Notion data into LangChain."""

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import MarkdownTextSplitter
from langchain.vectorstores import FAISS
from pathlib import Path
import faiss
import os
import pickle
import streamlit as st
import tqdm

openai_api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key
# Here we load in the data in the format that Notion exports it in.
ps = list(Path("notion_dump/").glob("**/*.md"))

data = []
sources = []
for p in ps:
    with open(p) as f:
        data.append(f.read())
    sources.append(p)

print("Prepared documents for spliting and processing")
# Here we split the documents, as needed, into smaller chunks.
# We do this due to the context limits of the LLMs.
text_splitter = MarkdownTextSplitter(chunk_size=200, chunk_overlap=20) # try to split on paragraph level for embediings to improve performance
docs = []
metadatas = []
for i, d in enumerate(tqdm.tqdm(data)):
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    metadatas.extend([{"source": sources[i]}] * len(splits))

print("Finished splitting documents")
print("Created {} documents to process for embeddings".format(len(docs)))

# Here we create a vector store from the documents and save it to disk.
store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)

print("Finished creating vector store from OPENAI embeddings")
faiss.write_index(store.index, "docs.index")
store.index = None
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)
print("Dumped index")