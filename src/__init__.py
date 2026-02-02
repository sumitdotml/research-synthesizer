"""Agentic RAG - Research Paper Synthesis Agent."""

from .config import get_llm_with_fallback
from .ingest import download_papers, chunk_papers
from .retriever import build_index, get_retriever, retrieve
from .query_engine import create_query_engine, query_with_sources
from .decomposition import decompose_query
from .agent import create_synthesis_agent
from .evaluate import load_test_questions, compare_approaches, print_comparison_table

__all__ = [
    "get_llm_with_fallback",
    "download_papers",
    "chunk_papers",
    "build_index",
    "get_retriever",
    "retrieve",
    "create_query_engine",
    "query_with_sources",
    "decompose_query",
    "create_synthesis_agent",
    "load_test_questions",
    "compare_approaches",
    "print_comparison_table",
]
