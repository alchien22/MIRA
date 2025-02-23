import logging
import os

from typing import List
from time import sleep

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

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
    
# Vector store class to create a Chroma database with embeddings and methods for retrieval
class VectorStore:
    def __init__(self, collection_name="EHRData"):
        self.collection_name = collection_name
        self.text_data = None
        self.vector_store = None

    def create_vector_db(self, data):
        print("Creating vector database")
        embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs,encode_kwargs=encode_kwargs)
        if not data:
            logging.warning("Empty data passed in to create vector database")

        self.text_data = data
        proxy_embeddings = EmbeddingProxy(embeddings)

        # Split the documents into more managable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        data = text_splitter.split_documents(data)

        # Create a new Chroma database locally
        vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=proxy_embeddings,
                persist_directory=os.path.join("store/", self.collection_name)
            )
        
        vector_store.add_documents(documents=data, ids=[str(i) for i in range(len(data))])

        self.vector_store = vector_store
        print("Vector database created")

    def get_retriever(self):
        """Initializes an ensemble retriever with BM25 and vector store retrievers."""
        vs_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5,
                "filter": {
                    'subject_id': '10000032'
                }
            }
        )

        bm25_retriever = BM25Retriever.from_texts(
            texts=[doc.page_content for doc in self.text_data],
            search_kwargs={
                "k": 5,
                "filter": {
                    'subject_id': '10000032'
                }
            }
        )

        return EnsembleRetriever(
            retrievers=[vs_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )
    