# Project Notes: Agentic RAG System

This document captures the reasoning, decisions, complexities, and lessons learned while building this agentic RAG system.

---

## What This Project Is

An **Agentic RAG (Retrieval-Augmented Generation) system** that synthesizes information from research papers. Unlike basic RAG which simply retrieves and generates, this system uses:

- **Query Decomposition**: Breaking complex questions into simpler sub-questions
- **Multi-hop Retrieval**: Each sub-question retrieves independently
- **Synthesis**: Combining sub-answers into a comprehensive response

### The "Agentic" Part

The term "agentic" refers to the system's ability to reason about how to answer a question, rather than just doing retrieval → generation in a single pass. Specifically:

1. **Planning**: The system analyzes the question and decides to break it into parts
2. **Iterative Retrieval**: Multiple retrieval operations, not just one
3. **Reasoning**: Synthesis step that reasons across multiple pieces of evidence

---

## Why These Technology Choices

### Package Manager: `uv` (not pip)

- **Speed**: `uv` is significantly faster than pip for dependency resolution
- **Modern**: Built by Astral (the Ruff team), represents modern Python tooling
- **Lockfile**: Generates `uv.lock` for reproducible environments

### Vector Database: Chroma (not Pinecone, Weaviate, etc.)

- **Local-first**: No API keys, no cloud dependency, runs in-process
- **Python-native**: Simple integration, just `pip install chromadb`
- **Persistent storage**: Can save to disk and reload
- **Trade-off**: Not as scalable as cloud solutions, but perfect for demos and small-medium datasets

### Embeddings: `all-MiniLM-L6-v2` (not OpenAI embeddings)

- **Free**: No API costs
- **Local**: Runs on CPU, no network calls
- **Fast**: Small model (80MB), quick inference
- **Good enough**: 384-dimension embeddings work well for semantic search
- **Trade-off**: Not as powerful as OpenAI's ada-002 or newer models, but cost-free

### Framework: LlamaIndex (not LangChain)

- **RAG-focused**: Purpose-built for retrieval and indexing use cases
- **Better abstractions**: `VectorStoreIndex`, `QueryEngine`, `ResponseSynthesizer` are RAG-specific
- **Less boilerplate**: More opinionated, fewer decisions to make
- **Trade-off**: Less flexible than LangChain for non-RAG use cases

### LLM: OpenRouter (not direct OpenAI/Anthropic)

- **Model flexibility**: Access to many models through one API
- **Fallback chain**: Can try multiple models if one fails
- **Cost management**: Can use free/cheap models for development
- **Models used**:
  - Primary: `moonshotai/kimi-k2.5` (good quality, has credits)
  - Fallback 1: `arcee-ai/trinity-large-preview:free` (free, fast)
  - Fallback 2: `deepseek/deepseek-r1-0528:free` (free, has reasoning)

---

## How It Works

For a comprehensive step-by-step walkthrough of how queries flow through both traditional RAG and this project's agentic RAG, see **[WALKTHROUGH.md](WALKTHROUGH.md)**.

That document traces a concrete example question through every step, showing inputs, outputs, and the "why" at each stage.

---

## What Makes This "Modern" (vs 2023-era RAG)

### 2023 Basic RAG (The Naive Approach)
- Single retrieval pass
- Fixed chunk sizes
- No reasoning about the query
- Direct retrieval → generation

This was the standard pattern: embed the user query, find similar chunks, stuff them into a prompt, generate an answer. It works for simple factual questions but falls apart when:
- The question requires information spread across multiple documents
- The query phrasing doesn't match how information is stated in documents
- The question is complex and needs to be broken down

### This Project (Modern Patterns)

#### 1. Query Understanding: Decomposition Before Retrieval

**What it is**: Before doing any retrieval, the system analyzes the query and breaks it into simpler sub-questions.

**Why it matters**: A complex question like "Compare the evaluation methods used in RAG systems and their limitations" actually contains multiple implicit questions:
- What evaluation methods exist for RAG?
- How does each method work?
- What are the limitations of each?

If you search for the original complex query, you might get chunks that mention evaluation tangentially but don't address all aspects. By decomposing first, each sub-question can retrieve more targeted, relevant content.

**How it works in this project**:
```python
# In agent.py - decompose_query()
# LLM is prompted to analyze the question and output 2-4 sub-questions
# Each sub-question should be:
#   - Self-contained (answerable independently)
#   - Specific (not too broad)
#   - Relevant (contributes to answering the original)
```

**Trade-off**: Adds one LLM call before retrieval. Worth it for complex questions, overkill for "What is RAG?"

---

#### 2. Multi-hop Retrieval: Multiple Retrieval Operations Per Query

**What it is**: Instead of a single retrieval pass, the system performs separate retrieval for each sub-question, potentially following chains of information.

**Why it matters**: Information needed to answer a complex question is often scattered:
- Paper A defines the concept
- Paper B describes the methodology
- Paper C provides evaluation results

A single retrieval pass optimizes for one query embedding. Multi-hop retrieval can gather evidence from different "locations" in the embedding space.

**How it works in this project**:
```python
# For each sub-question from decomposition:
#   1. Embed the sub-question
#   2. Retrieve top-K chunks relevant to that specific sub-question
#   3. Generate a sub-answer using those chunks
# Result: Multiple sets of evidence, each targeted to a specific aspect
```

**The "hop" concept**: In more advanced systems, retrieval can be chained—the answer to one query informs the next query. This project uses parallel hops (all sub-questions retrieve independently) rather than sequential hops.

---

#### 3. Synthesis Over Aggregation: LLM Reasons Across Evidence

**What it is**: The final answer is generated by reasoning across all gathered evidence, not just concatenating retrieved chunks or sub-answers.

**Why it matters**: Basic RAG often produces answers that feel like a list of disconnected facts. The synthesis step produces coherent, integrated responses.

**Aggregation (basic approach)**:
```
Q: Compare methods A and B
Retrieved: [chunk about A], [chunk about B]
Answer: "Method A does X. Method B does Y." (just restating)
```

**Synthesis (this project)**:
```
Q: Compare methods A and B
Sub-answers: [detailed answer about A], [detailed answer about B]
Final: "While both methods address the same problem, they differ
       fundamentally in their approach. A prioritizes X at the cost of Y,
       whereas B trades off Z for better W. For use cases requiring..."
```

**How it works in this project**:
```python
# In create_synthesis_agent():
# After gathering all sub-answers with their sources,
# a synthesis prompt asks the LLM to:
#   1. Identify connections between sub-answers
#   2. Resolve any contradictions
#   3. Create a unified narrative that addresses the original question
#   4. Cite specific sources for claims
```

---

#### 4. Evaluation Built-in: Measuring Quality, Not Just Shipping

**What it is**: The system includes evaluation infrastructure from day one, not as an afterthought.

**Why it matters**: Without evaluation, you can't know if your RAG system is actually good. You can't measure improvements. You're flying blind.

**What's evaluated**:
- **Relevance**: Does the answer actually address what was asked?
- **Coverage**: Does it cover the key aspects of the topic?
- **Coherence**: Is the answer well-structured and readable?

**Comparison methodology**: Run the same questions through basic RAG and agentic RAG, score both, compare. This shows concrete improvement (or not) from the added complexity.

**Why this matters for a portfolio**: It demonstrates awareness that ML systems need evaluation, not just implementation.

---

#### 5. Fallback Handling: Graceful Degradation

**What it is**: When the primary LLM fails (rate limit, timeout, error), the system automatically tries backup models instead of crashing.

**Why it matters**: Real systems fail. APIs go down. Rate limits hit. A robust system handles these gracefully.

**How it works in this project**:
```python
# In config.py - MODEL_FALLBACK_CHAIN
FALLBACK_CHAIN = [
    "moonshotai/kimi-k2.5",           # Primary: good quality
    "arcee-ai/trinity-large-preview:free",  # Backup: free, fast
    "deepseek/deepseek-r1-0528:free"  # Last resort: free, has reasoning
]

# get_llm() tries models in order until one works
```

**Not just about LLMs**: The same principle applies throughout:
- PDF parsing fails? Skip that paper, continue with others
- Chroma persist fails? Fall back to in-memory mode
- Network timeout? Retry with exponential backoff

---

## LLM-as-Judge: A Deep Dive

### What is LLM-as-Judge?

LLM-as-Judge is an evaluation paradigm where a large language model is used to assess the quality of outputs from another system (often another LLM). Instead of human annotators scoring outputs, an LLM does the scoring.

### Why Use LLM-as-Judge?

**Traditional evaluation options**:

| Method | Pros | Cons |
|--------|------|------|
| Human evaluation | Gold standard, nuanced | Slow, expensive, doesn't scale |
| Automatic metrics (BLEU, ROUGE) | Fast, reproducible | Poor correlation with quality for open-ended tasks |
| Ground truth comparison | Objective | Requires labeled data, only one "right" answer |

**LLM-as-Judge offers**:
- **Scalability**: Evaluate thousands of outputs quickly
- **Cost-effective**: Much cheaper than human annotators
- **Nuanced assessment**: Can evaluate subjective qualities (coherence, helpfulness)
- **Customizable criteria**: Can define any rubric you want

### How LLM-as-Judge Works

**Basic pattern**:
```python
evaluation_prompt = """
You are evaluating the quality of an answer to a question.

Question: {question}
Answer: {answer}

Rate the answer on the following criteria (1-5 scale):

1. Relevance: Does the answer address the question?
   - 1: Completely off-topic
   - 3: Partially relevant
   - 5: Directly addresses all aspects

2. Coverage: Does it cover the expected topics?
   - 1: Missing most key points
   - 3: Covers some topics
   - 5: Comprehensive coverage

3. Coherence: Is it well-structured and clear?
   - 1: Incoherent/confusing
   - 3: Understandable but disorganized
   - 5: Clear, logical, well-organized

Output JSON: {"relevance": N, "coverage": N, "coherence": N, "reasoning": "..."}
"""
```

### Implementation in This Project

**Location**: `src/evaluate.py`

**Process**:
1. Load test questions with expected topics
2. Run each question through both basic RAG and agentic RAG
3. For each answer, call the judge LLM with the evaluation prompt
4. Parse scores, compute averages
5. Compare approaches

**Code structure**:
```python
def evaluate_answer(question: str, answer: str, expected_topics: list[str]) -> dict:
    """Use LLM to judge answer quality."""
    prompt = EVALUATION_PROMPT.format(
        question=question,
        answer=answer,
        expected_topics=", ".join(expected_topics)
    )
    response = llm.complete(prompt)
    return parse_evaluation(response.text)

def compare_approaches(questions: list[dict]) -> dict:
    """Compare basic vs agentic RAG on same questions."""
    basic_scores = []
    agentic_scores = []

    for q in questions:
        basic_answer = basic_rag.query(q["question"])
        agentic_answer = agentic_rag.query(q["question"])

        basic_scores.append(evaluate_answer(q["question"], basic_answer, q["topics"]))
        agentic_scores.append(evaluate_answer(q["question"], agentic_answer, q["topics"]))

    return aggregate_scores(basic_scores, agentic_scores)
```

### Limitations of LLM-as-Judge

**Known issues**:

1. **Verbosity bias**: LLMs tend to rate longer answers higher, even if shorter answers are better
2. **Self-preference**: When the same model generates and judges, it may prefer its own style
3. **Position bias**: In pairwise comparisons, LLMs may prefer the first or second option
4. **Inconsistency**: Same LLM may give different scores on repeated evaluations
5. **No ground truth**: The judge can be confidently wrong

**Mitigations used in this project**:
- Explicit scoring rubrics (1-5 with descriptions)
- JSON output format for structured parsing
- Multiple criteria, not single score
- Reasoning required (forces the judge to explain)

### When LLM-as-Judge Is Appropriate

**Good fit**:
- Comparing approaches (A vs B)
- Rapid iteration during development
- Subjective quality dimensions
- When you need scale

**Not a replacement for**:
- Critical production evaluations (still need human review)
- Factual accuracy checking (LLMs can hallucinate judgments too)
- Specialized domain evaluation (may need domain experts)

### Further Reading

- [Judging LLM-as-a-Judge (paper)](https://arxiv.org/abs/2306.05685) - Foundational paper on this approach
- RAGAS (Retrieval-Augmented Generation Assessment) - Framework with LLM-based metrics
- MT-Bench - Multi-turn benchmark using LLM judges

---

## Complexities and Challenges

### 1. PDF Parsing
- **Problem**: PDFs are notoriously hard to parse. Tables, figures, multi-column layouts break extraction.
- **Solution**: Used `pypdf` which handles most academic papers reasonably well
- **Limitation**: Some papers still have garbled text, especially those with complex layouts

### 2. Chunking Strategy
- **Problem**: How big should chunks be? Too small = missing context. Too large = noise.
- **Solution**: 1024 characters with 200 character overlap (sentence-aware splitting)
- **Trade-off**: This is a reasonable default but not optimized per-document

### 3. Query Decomposition Quality
- **Problem**: LLM might generate bad sub-questions (too similar, too broad, off-topic)
- **Solution**: Prompt engineering with specific guidelines
- **Limitation**: Still depends on LLM quality; no verification of sub-question quality

### 4. Retrieval Precision
- **Problem**: Embedding similarity doesn't always equal semantic relevance
- **Solution**: Using top-5 retrieval to increase recall
- **Not implemented**: Reranking (would improve precision significantly)

### 5. Synthesis Hallucination
- **Problem**: LLM might add information not in the retrieved context
- **Solution**: Prompts instruct to cite specific findings
- **Limitation**: No automated fact-checking against sources

### 6. Latency vs Quality Trade-off
- **Problem**: Agentic approach is slower (multiple LLM calls)
- **Measured**: ~3-5x slower than basic RAG
- **Trade-off**: Worth it for complex questions, overkill for simple ones

---

## Limitations

### Current Limitations

1. **Scale**: Tested with 5 papers (~50 chunks). Might need optimization for 100+ papers
2. **No reranking**: Retrieved chunks aren't reordered by relevance
3. **No hybrid search**: Only vector search, no BM25/keyword component
4. **No streaming**: Responses wait for full generation
5. **No memory**: Each query is independent, no conversation history
6. **No source verification**: Can't verify if synthesis matches sources
7. **Fixed chunking**: Same strategy for all document types

### What Would Improve It

1. **Hybrid retrieval**: Combine dense (vector) + sparse (BM25) search
2. **Reranking**: Use a cross-encoder to rerank retrieved chunks
3. **Adaptive chunking**: Different strategies for different content types
4. **Query routing**: Decide whether to use basic or agentic based on query complexity
5. **Streaming**: Stream responses for better UX
6. **Citation verification**: Check that claims are grounded in sources

---


## Key Decisions and Rationale

| Decision | Rationale |
|----------|-----------|
| Use LlamaIndex over LangChain | More RAG-specific, less boilerplate |
| Local embeddings over API | Free, fast, no network dependency |
| Chroma over cloud vector DB | Simple, local, good for demos |
| OpenRouter over direct API | Fallback capability, model flexibility |
| 1024-char chunks | Balance between context and precision |
| Top-5 retrieval | Higher recall, let LLM filter noise |
| LLM-as-judge evaluation | Scalable, automated, good enough for comparison |

---

## Lessons Learned

1. **Relative paths in notebooks are tricky**: Had to change working directory for imports to work
2. **LlamaIndex OpenAI wrapper validates model names**: Needed `OpenAILike` for OpenRouter
3. **PDF parsing is never perfect**: Some papers just don't extract well
4. **Decomposition quality varies**: Sometimes the LLM generates redundant sub-questions
5. **Evaluation is hard**: LLM-as-judge is convenient but not perfect

---

## Questions This Project Can Answer

About the papers:
- "What is retrieval augmented generation?"
- "What evaluation metrics are used for RAG systems?"
- "How do different chunking strategies affect RAG performance?"
- "Compare dense vs sparse retrieval methods"

About the system:
- "Why did you choose Chroma over Pinecone?"
- "What's the difference between basic and agentic RAG?"
- "How does query decomposition work?"
- "What are the limitations of this approach?"

---

## Future Improvements (If Continued)

1. **Add reranking**: Use `sentence-transformers` cross-encoder
2. **Hybrid search**: Add BM25 alongside vector search
3. **Better evaluation**: Add RAGAS metrics (faithfulness, answer relevancy)
4. **Query routing**: Automatically choose basic vs agentic
5. **More papers**: Scale to 50+ papers
6. **Web UI**: Gradio or Streamlit interface
7. **Streaming**: Stream synthesis for better UX
