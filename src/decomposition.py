"""Query decomposition for complex questions."""

import json

from llama_index.core.llms import LLM

from src.config import get_llm_with_fallback


DECOMPOSITION_PROMPT = """You are a research assistant helping decompose complex questions into simpler sub-questions.

Given a complex question about research papers, break it down into 2-4 simpler sub-questions that can be answered individually and then synthesized.

Guidelines:
- Each sub-question should be self-contained
- Sub-questions should cover different aspects of the original question
- Keep sub-questions focused and specific
- Return ONLY a JSON array of strings, nothing else

Question: {question}

Sub-questions (JSON array):"""


def decompose_query(query: str, llm: LLM = None) -> list[str]:
    """
    Decompose a complex query into simpler sub-questions.

    Args:
        query: The complex question to decompose
        llm: LLM to use (will get default if not provided)

    Returns:
        List of sub-questions
    """
    if llm is None:
        llm = get_llm_with_fallback()

    prompt = DECOMPOSITION_PROMPT.format(question=query)
    response = llm.complete(prompt)

    try:
        text = response.text.strip()
        # handling markdown code blocks that LLMs sometimes return
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        sub_questions = json.loads(text)
        if isinstance(sub_questions, list):
            return sub_questions
    except (json.JSONDecodeError, IndexError):
        pass

    return [query]
