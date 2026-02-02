# Project Context: Research Synthesizer

## Why This Project

I wanted to deepen my understanding of modern RAG patterns by building something hands-on. While I have experience with ML fundamentals (implemented Transformer and Seq2Seq papers from scratch), I hadn't built a production-style RAG system with agentic capabilities.

**Goal**: Build a polished agentic RAG project that demonstrates modern patterns, not just 2023-era basic RAG.

---

## My Background

**ML experience** (personal projects):
- Replicated Transformer paper from scratch (PyTorch)
- Replicated Seq2Seq paper, deployed to HuggingFace
- Working on MoE (Mixture of Experts) research project
- Built small tools with Claude API

**What I wanted to learn**:
- Production RAG patterns
- Agentic AI implementation
- Working with vector databases
- Evaluation frameworks for RAG

---

## What "Modern" Means (Not 2023 RAG)

Old-school RAG (what to avoid as the main focus):
- Simple cosine similarity retrieval
- Chunk → embed → retrieve → generate
- No evaluation, no iteration, no tool use

Modern patterns to consider:
- **Agentic RAG**: Agent decides when/how to retrieve, can use multiple tools
- **Query rewriting/decomposition**: Break complex queries into sub-queries
- **Self-reflection/correction**: Agent evaluates its own answers
- **Hybrid retrieval**: Combining dense + sparse, or multiple retrieval strategies
- **Evaluation frameworks**: Measuring retrieval quality, answer quality
- **Graph-based approaches**: Knowledge graphs, entity relationships
- **Multi-step reasoning**: Chain of thought, tree of thought over retrieved context

---

## Technical Preferences

- **Language**: Python
- **Package manager**: uv (Astral) - NOT pip/requirements.txt
- **Open to**: LangChain, LlamaIndex, or building with primitives
- **LLM**: OpenRouter API (flexible model switching)
- **Vector DB**: Chroma (local, no credentials needed)
- **Deployment**: Not required, can be local/notebook

---

## Project Criteria

Must have:
- [x] Demonstrates multiple modern patterns beyond basic RAG
- [x] Clean, readable code
- [x] Proper evaluation with metrics
- [x] Can explain every component
- [x] Comparison with baseline (basic RAG vs. agentic approach)

Nice to have:
- [x] Interesting use case / demo scenario
- [ ] Visualizations of how it works
- [x] Documentation / README
- [x] Modular design showing software engineering practices

---

## What Was Built

A **Research Paper Synthesis Agent** that:
1. Ingests papers from arxiv on any topic
2. Indexes them in a vector database (Chroma)
3. Answers complex questions requiring cross-paper synthesis
4. Uses query decomposition and multi-hop retrieval
5. Evaluates itself against a baseline

---

## LLMs Used

Configured via OpenRouter with fallback chain:

- Primary: `moonshotai/kimi-k2.5`
- Fallback 1: `arcee-ai/trinity-large-preview:free` (free, fast)
- Fallback 2: `deepseek/deepseek-r1-0528:free` (free, has reasoning)

There is a `.env` file with the OpenRouter API key. See `.env.example` for the template.

---

## Key Learnings

1. **Query decomposition** significantly improves answers for complex questions
2. **Local embeddings** (sentence-transformers) are good enough for many use cases
3. **LLM-as-judge** evaluation is practical and scalable
4. **Chroma** is excellent for local development and demos
5. **LlamaIndex** provides great RAG-specific abstractions
