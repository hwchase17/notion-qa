import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Load the index
# index = faiss.read_index("docs.index")
index = FAISS.load_local('faiss', OpenAIEmbeddings()).index

# print(index)

# # Get information about the index
print("Index Type: ", index.__class__.__name__)
print("Number of vectors: ", index.ntotal)
print("Vector dimension: ", index.d)
# print("Vectors: ", index.reconstruct_n(range(2)))
# print(index.reconstruct_n(range(index.ntotal), index.ntotal))


