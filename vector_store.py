import logging
import os

from typing import List
from time import sleep

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

EMBED_DELAY = 0.02  # 20 milliseconds

# Embedding model config
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu", "trust_remote_code": True}
encode_kwargs = {'normalize_embeddings': False}

# This is to get the Streamlit app to use less CPU while embedding documents into Chromadb.
class EmbeddingProxy:
    def __init__(self, embedding):
        self.embedding = embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        sleep(EMBED_DELAY)
        return self.embedding.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        sleep(EMBED_DELAY)
        return self.embedding.embed_query(text)
    

# This happens all at once, not ideal for large datasets.
def create_vector_db(texts, embeddings=None, collection_name="chroma"):
    if not texts:
        logging.warning("Empty texts passed in to create vector database")
    if not embeddings:
        embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs,encode_kwargs=encode_kwargs)
    proxy_embeddings = EmbeddingProxy(embeddings)

    # Create a ChromaDB vectorstore from documents
    # this will be a chroma collection with a default name.
    db = Chroma(collection_name=collection_name,
                embedding_function=proxy_embeddings,
                persist_directory=os.path.join("store/", collection_name))
    
    db.add_documents(texts)

    return db

def find_similar(vs, query):
    docs = vs.similarity_search(query)
    return docs
