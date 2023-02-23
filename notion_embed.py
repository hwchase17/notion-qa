import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
import numpy as np
import os


faiss_location = 'faiss_moonchaser_xs_embedding_feb23'
# Displaying the contents of the text file
docs = np.load('docs_np_moonchaser_xs_embedding_feb23.npy') # load
docs = list(docs)
metadatas = np.load('metadata_np_moonchaser_xs_embedding_feb23.npy') # load
metadatas = list(metadatas)

sources = []
for metadata in metadatas:
    sources.append({"source": metadata})

store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=sources)
        # This creates an index out of the store data structure
if os.path.exists(faiss_location):
    existing_index = FAISS.load_local(faiss_location, OpenAIEmbeddings())
    existing_index.add_texts(texts=docs, metadatas=sources)


    # existing_index = faiss.read_index("docs.index")
    # n = store.index.ntotal
    # d = store.index.d
    # print(store.index.reconstruct_n(0, n))
    # print(type(store.index.reconstruct_n))
    # print(len(store.index.reconstruct_n))
    # print(store.index.reconstruct_n.shape)
    # vectors = faiss.vector_to_array(store.index.reconstruct_n(0, n))
    # vectors = vectors.reshape(n, d)
    # existing_index.add(vectors)


    # Should probably stick to using FAISS (langchain) vs intermingling with faiss
    print(f'{os.getcwd()=}')
    existing_index.save_local(faiss_location)
    # faiss.write_index(existing_index, "docs.index")
else:
    store.save_local(faiss_location)
    # faiss.write_index(store.index, "docs.index")
# Now that it's saved to disk, we can remove the index from memory
store.index = None

