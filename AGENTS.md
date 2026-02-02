# AGENTS.md

## Project Purpose

A personal learning project to build a modern **Agentic RAG system** that demonstrates understanding of current patterns beyond basic 2023-era RAG. The goal is to create something concrete, polished, and demonstrably "modern" that can serve as a portfolio piece.

**Language**: Python
**Goal**: A working agentic RAG system with query decomposition, multi-hop retrieval, and evaluation

Read `CONTEXT.md` for full background on what "modern" means in RAG context.

---

## Development Principles

### Code Quality

- Write clean, readable code with clear structure
- Add docstrings to classes and important functions
- Keep functions small and focused
- Use type hints consistently
- Prefer explicit over implicit

### Architecture

- Modular design - each component should be independently understandable
- Clear separation between: data ingestion, retrieval, generation, evaluation
- Make it easy to swap components (different retrievers, different LLMs)
- Configuration over hardcoding

### Dependencies

- Use `uv` (Astral) for all package management - NOT pip
- Prefer well-maintained libraries (LangChain, LlamaIndex, etc.)
- Minimize dependencies where possible - don't add a library for one function

### Testing & Evaluation

- Include evaluation metrics from the start, not as an afterthought
- Compare against a baseline (basic RAG) to show improvement
- Document what metrics mean and why they matter

---

## What NOT To Do

- Don't over-engineer - this is a demo, not production software
- Don't add features just because they're cool - every feature should serve the demo story
- Don't use bleeding-edge libraries with poor documentation
- Don't skip error handling - demo should work reliably
- Don't hardcode API keys - use environment variables

---

## Project Structure

```
research-synthesizer/
├── AGENTS.md           # This file (CLAUDE.md symlinks here)
├── CONTEXT.md          # Project background
├── NOTES.md            # Detailed project notes
├── README.md           # User-facing documentation
├── pyproject.toml      # Project config & dependencies (managed by uv)
├── uv.lock             # Lock file (auto-generated)
├── .env.example        # Environment variable template
├── src/
│   ├── __init__.py
│   ├── config.py        # LLM and configuration
│   ├── ingest.py        # Data ingestion
│   ├── retriever.py     # Retrieval components
│   ├── query_engine.py  # Basic RAG query engine
│   ├── decomposition.py # Query decomposition logic
│   ├── agent.py         # Synthesis agent (agentic RAG)
│   └── evaluate.py      # Evaluation metrics
├── data/               # Sample data for demo
├── notebooks/          # Jupyter notebooks for exploration/demo
└── tests/              # Basic tests
```

---

## Commands

```bash
# Initialize project (first time)
uv init

# Add dependencies
uv add langchain anthropic  # example

# Add dev dependencies
uv add --dev pytest ruff

# Run scripts
uv run python src/main.py

# Run tests
uv run pytest tests/

# Sync dependencies (after pulling changes)
uv sync
```

---

## Session Notes

_Update this section after each work session with key decisions, learnings, and next steps._

---

## Lessons Learned

_Add rules here when mistakes are made, so they don't repeat._

Example format:

- **Problem**: [What went wrong]
- **Rule**: [What to do instead]
