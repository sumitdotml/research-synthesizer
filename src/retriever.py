"""Vector index and retrieval module using Chroma and HuggingFace embeddings."""

import chromadb
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import CHROMA_DIR, EMBEDDING_MODEL


def get_embedding_model() -> HuggingFaceEmbedding:
    """Get the HuggingFace embedding model."""
    return HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)


def build_index(
    documents: list[Document],
    collection_name: str = "papers",
    persist_dir: str = CHROMA_DIR,
) -> VectorStoreIndex:
    """
    Build a vector index from documents using Chroma.

    Args:
        documents: List of LlamaIndex Document objects
        collection_name: Name for the Chroma collection
        persist_dir: Directory to persist Chroma data

    Returns:
        VectorStoreIndex ready for retrieval
    """
    try:
        chroma_client = chromadb.PersistentClient(path=persist_dir)
    except Exception as e:
        print(f"Warning: Failed to create persistent Chroma, using in-memory: {e}")
        chroma_client = chromadb.Client()

    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = get_embedding_model()
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    return index


def load_index(
    collection_name: str = "papers",
    persist_dir: str = CHROMA_DIR,
) -> VectorStoreIndex:
    """
    Load an existing vector index from Chroma.

    Args:
        collection_name: Name of the Chroma collection
        persist_dir: Directory where Chroma data is persisted

    Returns:
        VectorStoreIndex loaded from persisted data
    """
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    embed_model = get_embedding_model()
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )

    return index


def get_retriever(
    index: VectorStoreIndex,
    top_k: int = 5,
) -> VectorIndexRetriever:
    """
    Create a retriever from the index.

    Args:
        index: The vector store index
        top_k: Number of top results to retrieve

    Returns:
        VectorIndexRetriever configured for retrieval
    """
    return index.as_retriever(similarity_top_k=top_k)


def retrieve(
    query: str,
    index: VectorStoreIndex,
    top_k: int = 5,
) -> list[NodeWithScore]:
    """
    Retrieve relevant chunks for a query.

    Args:
        query: The search query
        index: The vector store index
        top_k: Number of top results to retrieve

    Returns:
        List of NodeWithScore objects with retrieved chunks
    """
    retriever = get_retriever(index, top_k=top_k)
    return retriever.retrieve(query)


if __name__ == "__main__":
    from src.ingest import chunk_papers

    print("Loading documents...")
    docs = chunk_papers()
    print(f"Loaded {len(docs)} document chunks")

    print("Building index...")
    index = build_index(docs)
    print("Index built and persisted")

    print("\nTesting retrieval for 'What is RAG?'...")
    results = retrieve("What is RAG?", index, top_k=3)
    for i, result in enumerate(results):
        print(f"\n--- Result {i+1} (score: {result.score:.4f}) ---")
        print(f"Title: {result.node.metadata.get('title', 'N/A')}")
        print(f"Text preview: {result.node.get_content()[:200]}...")
