# Research Synthesizer

This is a relatively straightforward learning project exploring agentic RAG patterns. I built it for educational purposes to understand query decomposition and multi-hop retrieval beyond basic 2023-era RAG.

## What Makes This "Agentic"?

Unlike basic RAG that simply retrieves and generates, this system:

1. Query Decomposition: Complex questions are broken into 2-4 simpler sub-questions
2. Multi-hop Retrieval: Each sub-question retrieves relevant context independently
3. Synthesis: Sub-answers are combined into a comprehensive final response

## Architecture

(<3 Claude Opus 4.5 for the architecture diagram)

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Complex Query  │────▶│  Decomposition   │────▶│  Sub-questions   │
└──────────────────┘     │      (LLM)       │     │    (2-4 items)   │
                         └──────────────────┘     └────────┬─────────┘
                                                           │
                    ┌──────────────────────────────────────┘
                    ▼
         ┌──────────────────┐     ┌──────────────────┐
         │   Retrieval      │────▶│   Sub-answers    │
         │  (per question)  │     │   (with sources) │
         └──────────────────┘     └────────┬─────────┘
                                           │
                    ┌──────────────────────┘
                    ▼
         ┌──────────────────┐     ┌──────────────────┐
         │   Synthesis      │────▶│  Final Answer    │
         │      (LLM)       │     │  (with citations)│
         └──────────────────┘     └──────────────────┘
```

## Tech Stack

| Component       | Choice                             | Reason                                  |
| --------------- | ---------------------------------- | --------------------------------------- |
| Package Manager | `uv`                               | Fast, modern Python                     |
| Vector DB       | Chroma                             | Local, no credentials                   |
| Embeddings      | all-MiniLM-L6-v2                   | Free, local, fast                       |
| Framework       | LlamaIndex                         | RAG-focused abstractions                |
| LLM             | OpenRouter (kimi-k2.5 + fallbacks) | Flexible model routing, has free models |

## Setup

### 1. Clone and Install

```bash
git clone <repo-url>
cd research-synthesizer

# Install dependencies
uv sync
```

### 2. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your OpenRouter API key
# OPENROUTER_API_KEY=your_key_here
```

### 3. Download Papers and Build Index

```bash
# Download 5 papers on for instance, MoE topics
uv run python -c "from src.ingest import download_papers; download_papers('mixture of experts', 10)"

# Chunk papers
uv run python -c "from src.ingest import chunk_papers; docs = chunk_papers(); print(f'{len(docs)} chunks')"

# Build vector index
uv run python -m src.retriever
```

## Usage

### Basic RAG

```python
from src.retriever import load_index
from src.query_engine import create_query_engine, query_with_sources

index = load_index()
query_engine = create_query_engine(index)

result = query_with_sources(query_engine, "What is Mixture of Experts?")
print(result["answer"])
print(result["sources"])
```

### Agentic RAG (Synthesis Agent)

```python
from src.retriever import load_index
from src.agent import create_synthesis_agent

index = load_index()
agent = create_synthesis_agent(index)

result = agent("Compare different mixture of experts architectures and their effectiveness")
print(result["answer"])
print(result["sub_questions"])
print(result["sources"])
```

### Run Evaluation

```bash
uv run python -m src.evaluate
```

### Demo Notebook

```bash
cd notebooks
uv run jupyter notebook demo.ipynb
```

## Project Structure

```
research-synthesizer/
├── src/
│   ├── config.py        # LLM configuration with fallbacks
│   ├── ingest.py        # Arxiv download and PDF chunking
│   ├── retriever.py     # Chroma index and retrieval
│   ├── query_engine.py  # Basic RAG query engine
│   ├── decomposition.py # Query decomposition logic
│   ├── agent.py         # Synthesis agent (agentic RAG)
│   └── evaluate.py      # Evaluation framework
├── data/
│   ├── papers/          # Downloaded PDFs
│   ├── chroma_db/       # Vector index
│   ├── metadata.json    # Paper metadata
│   └── test_questions.json
├── notebooks/
│   └── demo.ipynb       # Interactive demo
└── tests/
```

## Evaluation Metrics

The system evaluates answers using LLM-as-judge on:

- Relevance (1-5): Does the answer address the question?
- Coverage (1-5): Does it cover expected topics?
- Coherence (1-5): Is it well-structured and clear?

Run `uv run python -m src.evaluate` to compare baseline vs agentic RAG.

## Key Features

- Fallback LLM chain: If primary model fails, automatically tries backups
- Persistent index: Chroma index persisted to disk for fast reloading
- Source citations: All answers include source documents
- Modular design: Easy to swap components (retriever, LLM, etc.)

## Documentation

- `WALKTHROUGH.md` - Step-by-step trace of how queries flow through the system
- `NOTES.md` - Detailed project notes, decisions, and learnings
- `CONTEXT.md` - Project background and motivation
- `CLAUDE.md` - Development guidelines
