# Research Synthesizer: Implementation Plan

## Project Summary

**What it does**: An agentic RAG system that ingests research papers from arxiv, indexes them, and can answer complex questions requiring cross-paper synthesis—handling scale (50+ papers) that can't fit in any context window.

**Key capability**: Finds exact methodology passages across multiple papers, compares approaches, and traces citations. Basic LLMs with internet access can't do this at scale—they'd need to re-read everything for each query.

---

## Tech Stack Decisions (with Reasoning)

### 1. Data Source: Arxiv API

**Why**:
- Free, no authentication needed
- Provides paper metadata (title, authors, abstract) plus PDF links
- Demonstrates automated pipeline (more impressive than manual upload)
- Arxiv papers are open access—no copyright issues for demo

### 2. Vector Database: Chroma

**Why**:
- **Local-first**: No cloud account, no API keys, no external dependencies
- **Python-native**: Integrates cleanly with LlamaIndex
- **Simple API**: Easy to understand ("it's an in-process vector store")
- **Persistent**: Data survives restarts without external database setup
- **Well-documented**: Good community, easy troubleshooting

**Alternatives considered**:
- FAISS: Lower-level, more code to write
- Qdrant: More features but more complex setup
- Pinecone: Cloud-based, requires account, adds external dependency

### 3. Embeddings: sentence-transformers (local)

**Why**:
- **Free**: No API costs
- **Fast**: Local inference, no network latency
- **Quality**: `all-MiniLM-L6-v2` is proven for document retrieval
- **No API key needed**: One less credential to manage

**Alternatives considered**:
- OpenAI embeddings: Better quality but costs money, adds dependency
- Voyage AI: Good for docs but adds another API key

### 4. Framework: LlamaIndex

**Why**:
- **RAG-focused**: Built specifically for document retrieval use cases
- **Good abstractions**: VectorStoreIndex, QueryEngine work out of the box
- **Agentic support**: Built-in agent capabilities, tool calling
- **Active development**: Modern patterns, good documentation

**Alternatives considered**:
- LangChain: More general, can be verbose, more boilerplate
- Primitives: Full control but more code

### 5. LLM: OpenRouter (moonshotai/kimi-k2.5)

**Why**:
- OpenRouter allows easy model switching
- Compatible with OpenAI SDK (just change base_url)

**Fallback chain** (all verified working):
1. Primary: `moonshotai/kimi-k2.5`
2. Fallback 1: `arcee-ai/trinity-large-preview:free` (1.6s response)
3. Fallback 2: `deepseek/deepseek-r1-0528:free` (3.4s, has reasoning)

---

## Scale Strategy

| Phase              | Papers       | Purpose                               |
| ------------------ | ------------ | ------------------------------------- |
| Development (v0.1) | 5-10 papers  | Fast iteration, debug pipeline        |
| Demo (v1.0)        | 25-30 papers | Impressive scale, cross-paper queries |
| Stretch            | 50+ papers   | If time permits                       |

---

## Implementation Phases

### Phase 0: Project Setup
- Initialize project with uv
- Add dependencies
- Create project structure

### Phase 1: Data Ingestion Pipeline
- Arxiv paper downloader
- PDF parsing and chunking

### Phase 2: Vector Index
- Embedding and indexing with Chroma
- Basic retrieval test

### Phase 3: RAG Query Engine
- Basic RAG pipeline
- OpenRouter integration

### Phase 4: Agentic Capabilities
- Query decomposition
- Multi-hop retrieval
- Cross-paper synthesis

### Phase 5: Evaluation
- Create test question set
- Implement evaluation metrics
- Compare baseline vs. agentic

### Phase 6: Demo & Documentation
- Interactive demo notebook
- Documentation

---

## Autonomous Fallback Strategies

| Failure | Fallback | Action |
|---------|----------|--------|
| Primary LLM fails | Try free fallback models | Auto-switch in config |
| Arxiv rate limited | Sleep 60s, retry (max 3) | Built into ingest.py |
| PDF parse fails | Skip paper, continue | Log and continue |
| Chroma persist fails | Use in-memory mode | Log warning |
| Import error | Re-run `uv sync` | Auto-fix |
| Network timeout | Retry with backoff | Built into requests |

---

## Verification Commands

```bash
# After Phase 0:
uv run python -c "import llama_index; import chromadb; import sentence_transformers; print('Dependencies OK')"

# After Phase 1:
ls data/papers/*.pdf | wc -l  # Should show 5+

# After Phase 2:
uv run python -c "from src.retriever import load_index; idx = load_index(); print('Index loaded')"

# After Phase 3:
uv run python -m src.agent  # Should return coherent answer

# After Phase 5:
uv run python -m src.evaluate  # Should print metrics table
```

---

## Key Talking Points

1. **What it does**: Research paper synthesis agent that handles multiple papers—more than any context window
2. **What makes it modern/agentic**: Query decomposition, multi-hop retrieval, cross-paper synthesis
3. **How it's evaluated**: LLM-as-judge with relevance, coverage, coherence metrics
4. **Why these tech choices**: Local-first (Chroma, sentence-transformers), RAG-optimized (LlamaIndex), cost-efficient
5. **Future improvements**: Knowledge graph of citations, fine-tuned embeddings, multi-modal (figures/tables)
