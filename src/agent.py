"""Agentic RAG synthesis agent with multi-hop retrieval."""

from typing import Any

from llama_index.core import VectorStoreIndex, Settings

from src.config import get_llm_with_fallback
from src.retriever import get_embedding_model
from src.query_engine import create_query_engine, query_with_sources
from src.decomposition import decompose_query


SYNTHESIS_PROMPT = """You are a research synthesis expert. Given a complex question and answers to related sub-questions, provide a comprehensive synthesized answer.

Original Question: {original_question}

Sub-questions and their answers:
{sub_qa_pairs}

Based on the above information, provide a comprehensive answer to the original question. Cite specific findings from the sub-answers where relevant. If there are conflicting findings, acknowledge them.

Synthesized Answer:"""


def create_synthesis_agent(
    index: VectorStoreIndex,
    top_k: int = 3,
):
    """
    Create a synthesis agent that can answer complex multi-part questions.

    Args:
        index: Vector store index for retrieval
        top_k: Number of chunks per sub-question

    Returns:
        A callable agent function
    """
    llm = get_llm_with_fallback()
    embed_model = get_embedding_model()

    Settings.llm = llm
    Settings.embed_model = embed_model

    query_engine = create_query_engine(index, top_k=top_k)

    def agent(question: str) -> dict[str, Any]:
        """
        Answer a complex question using query decomposition and synthesis.

        Args:
            question: The complex question to answer

        Returns:
            Dict with answer, sub_questions, sub_answers, and sources
        """
        print(f"Decomposing query: {question}")
        sub_questions = decompose_query(question, llm)
        print(f"Sub-questions: {sub_questions}")

        sub_answers = []
        all_sources = []

        for i, sub_q in enumerate(sub_questions):
            print(f"\nAnswering sub-question {i + 1}: {sub_q}")
            result = query_with_sources(query_engine, sub_q)
            sub_answers.append(
                {
                    "question": sub_q,
                    "answer": result["answer"],
                }
            )
            all_sources.extend(result["sources"])

        sub_qa_pairs = "\n\n".join(
            [
                f"Q{i + 1}: {sa['question']}\nA{i + 1}: {sa['answer']}"
                for i, sa in enumerate(sub_answers)
            ]
        )

        synthesis_prompt = SYNTHESIS_PROMPT.format(
            original_question=question,
            sub_qa_pairs=sub_qa_pairs,
        )

        print("\nSynthesizing final answer...")
        final_response = llm.complete(synthesis_prompt)

        seen_titles = set()
        unique_sources = []
        for source in all_sources:
            if source["title"] not in seen_titles:
                seen_titles.add(source["title"])
                unique_sources.append(source)

        return {
            "answer": final_response.text,
            "sub_questions": sub_questions,
            "sub_answers": sub_answers,
            "sources": unique_sources,
        }

    return agent


if __name__ == "__main__":
    from src.retriever import load_index

    print("Loading index...")
    index = load_index()

    print("\n" + "=" * 60)
    print("Testing Query Decomposition")
    print("=" * 60)

    complex_query = "Compare different retrieval methods mentioned in the papers and their effectiveness"
    llm = get_llm_with_fallback()
    sub_questions = decompose_query(complex_query, llm)
    print(f"\nOriginal: {complex_query}")
    print("\nSub-questions:")
    for i, q in enumerate(sub_questions):
        print(f"  {i + 1}. {q}")

    print("\n" + "=" * 60)
    print("Testing Synthesis Agent")
    print("=" * 60)

    agent = create_synthesis_agent(index)
    result = agent("What do papers say about chunking strategies in RAG systems?")

    print(f"\n\nFinal Answer:\n{result['answer']}")
    print(f"\nSources ({len(result['sources'])}):")
    for source in result["sources"]:
        print(f"  - {source['title']}")
