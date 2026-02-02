"""Configuration and LLM setup for the agentic RAG system."""

import os
from typing import Optional

from dotenv import load_dotenv
from llama_index.core.llms import LLM
from llama_index.llms.openai_like import OpenAILike

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

PRIMARY_MODEL = "moonshotai/kimi-k2.5"
FALLBACK_MODEL_1 = "arcee-ai/trinity-large-preview:free"
FALLBACK_MODEL_2 = "deepseek/deepseek-r1-0528:free"
MODEL_CHAIN = [PRIMARY_MODEL, FALLBACK_MODEL_1, FALLBACK_MODEL_2]

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DATA_DIR = "data"
PAPERS_DIR = "data/papers"
CHROMA_DIR = "data/chroma_db"
METADATA_FILE = "data/metadata.json"


def get_llm(model: Optional[str] = None, temperature: float = 0.1) -> LLM:
    """
    Get an LLM configured for OpenRouter.

    Args:
        model: Model name (defaults to PRIMARY_MODEL)
        temperature: Temperature for generation

    Returns:
        Configured LLM instance
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set in environment")

    model = model or PRIMARY_MODEL

    return OpenAILike(
        model=model,
        api_key=OPENROUTER_API_KEY,
        api_base=OPENROUTER_API_BASE,
        temperature=temperature,
        is_chat_model=True,
    )


def get_llm_with_fallback(temperature: float = 0.1) -> LLM:
    """
    Get an LLM, trying fallback models if primary fails.

    Args:
        temperature: Temperature for generation

    Returns:
        Working LLM instance
    """
    last_error = None

    for model in MODEL_CHAIN:
        try:
            llm = get_llm(model=model, temperature=temperature)
            response = llm.complete("Say 'OK' if you're working.")
            if response.text:
                print(f"Using model: {model}")
                return llm
        except Exception as e:
            print(f"Model {model} failed: {e}")
            last_error = e
            continue

    raise RuntimeError(f"All models failed. Last error: {last_error}")


if __name__ == "__main__":
    print("Testing LLM connection...")
    llm = get_llm_with_fallback()
    response = llm.complete("What is retrieval augmented generation in one sentence?")
    print(f"\nResponse: {response.text}")
