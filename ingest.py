"""This is the logic for ingesting Notion data into LangChain."""
from langchain.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader("Notion_DB/", glob="**/*.md", loader_cls=TextLoader)
docs = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0, separator="\n")
docs = text_splitter.split_documents(docs)

embeddings = LlamaCppEmbeddings(model_path="ggml-model-q4_1.bin")
db = FAISS.from_documents(docs, embeddings)
db.save_local("faiss_index")
