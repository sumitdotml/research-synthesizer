"""Basic RAG query engine with source tracking."""

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer

from src.config import get_llm_with_fallback
from src.retriever import get_retriever, get_embedding_model


def create_query_engine(
    index: VectorStoreIndex,
    top_k: int = 5,
) -> RetrieverQueryEngine:
    """
    Create a basic RAG query engine.

    Args:
        index: Vector store index for retrieval
        top_k: Number of chunks to retrieve

    Returns:
        Query engine ready for questions
    """
    llm = get_llm_with_fallback()
    embed_model = get_embedding_model()

    Settings.llm = llm
    Settings.embed_model = embed_model

    retriever = get_retriever(index, top_k=top_k)
    response_synthesizer = get_response_synthesizer(
        llm=llm,
        response_mode="compact",
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )

    return query_engine


def query_with_sources(
    query_engine: RetrieverQueryEngine,
    question: str,
) -> dict:
    """
    Query the RAG system and return answer with sources.

    Args:
        query_engine: The query engine to use
        question: The question to answer

    Returns:
        Dict with 'answer' and 'sources' keys
    """
    response = query_engine.query(question)

    sources = []
    for node in response.source_nodes:
        sources.append(
            {
                "title": node.node.metadata.get("title", "Unknown"),
                "score": node.score,
                "text_preview": node.node.get_content()[:200] + "...",
            }
        )

    return {
        "answer": str(response),
        "sources": sources,
    }
