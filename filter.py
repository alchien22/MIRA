from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter, LongContextReorder
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever, MergerRetriever

from vector_store import create_vector_db

def create_retriever(texts):
    dense_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    sparse_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en",
                                                 encode_kwargs={'normalize_embeddings': False})
    dense_vs = create_vector_db(texts, collection_name="dense", embeddings=dense_embeddings)
    sparse_vs = create_vector_db(texts, collection_name="sparse", embeddings=sparse_embeddings)
    vector_stores = [dense_vs, sparse_vs]

    emb_filter = EmbeddingsRedundantFilter(embeddings=sparse_embeddings)
    reordering = LongContextReorder()
    pipeline = DocumentCompressorPipeline(transformers=[emb_filter, reordering])

    base_retrievers = [vs.as_retriever() for vs in vector_stores]
    lotr = MergerRetriever(retrievers=base_retrievers)

    compression_retriever_reordered = ContextualCompressionRetriever(
        base_compressor=pipeline, base_retriever=lotr, search_kwargs={"k": 5, "include_metadata": True}
    )
    return compression_retriever_reordered
