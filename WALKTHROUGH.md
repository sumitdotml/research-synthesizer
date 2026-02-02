# System Walkthrough: How Queries Flow Through the System

This document traces through exactly what happens when you ask a question, step by step. We'll use the same example question for both traditional RAG and this project's agentic RAG, so you can see the difference.

**Example question we'll trace:**
> "Compare the evaluation methods used for RAG systems and discuss their limitations"

This is a complex question that requires:
- Finding information about multiple evaluation methods
- Understanding how each method works
- Identifying limitations of each
- Synthesizing a comparison

---

## Part 1: Traditional RAG Flow

Traditional RAG follows a simple pipeline: **Query → Embed → Search → Generate**

### Step 1: Query Embedding

**What happens:** The user's question is converted into a vector (list of numbers) that represents its semantic meaning.

**Where:** `src/retriever.py` → uses the embedding model configured in the index

**Why:** Vector databases can't search text directly. They search by comparing vectors. We need to convert the question into the same vector space as our document chunks.

**Input:**
```
"Compare the evaluation methods used for RAG systems and discuss their limitations"
```

**Output:**
```python
# A 384-dimensional vector (using all-MiniLM-L6-v2)
[0.0234, -0.0891, 0.0412, ..., -0.0156]  # 384 numbers
```

**How it works internally:**
```python
# The embedding model (sentence-transformers) tokenizes the text,
# passes it through a transformer, and pools the output into a single vector
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
query_vector = model.encode("Compare the evaluation methods...")
```

---

### Step 2: Vector Search

**What happens:** The query vector is compared against all document chunk vectors in Chroma. The most similar chunks are returned.

**Where:** `src/retriever.py` → `retrieve()` function → Chroma's similarity search

**Why:** We want to find the chunks most likely to contain information relevant to the question. Vector similarity (cosine similarity) finds text with similar semantic meaning, even if the exact words differ.

**Input:**
```python
query_vector = [0.0234, -0.0891, ...]  # from Step 1
top_k = 5  # how many chunks to retrieve
```

**Output:**
```python
# Top 5 most similar chunks with their similarity scores
[
    {
        "text": "Evaluation of RAG systems typically involves metrics such as...",
        "metadata": {"source": "paper_1.pdf", "page": 4},
        "score": 0.82
    },
    {
        "text": "BLEU and ROUGE scores have been adapted for RAG evaluation...",
        "metadata": {"source": "paper_3.pdf", "page": 7},
        "score": 0.78
    },
    # ... 3 more chunks
]
```

**How it works internally:**
```python
# Chroma computes cosine similarity between query vector and all stored vectors
# Returns the top-k highest scoring chunks
results = collection.query(
    query_embeddings=[query_vector],
    n_results=5
)
```

**The problem with this step:** The search optimizes for the *original query's* embedding. But our question asks about multiple things (evaluation methods AND their limitations). A single embedding can't perfectly represent all aspects of a complex question.

---

### Step 3: Prompt Construction

**What happens:** The retrieved chunks are combined with the original question into a prompt for the LLM.

**Where:** `src/query_engine.py` → `create_query_engine()` → LlamaIndex's `ResponseSynthesizer`

**Why:** The LLM needs both the question and the relevant context to generate an answer. We "stuff" the retrieved chunks into the prompt.

**Input:**
```python
question = "Compare the evaluation methods used for RAG systems..."
chunks = [chunk1, chunk2, chunk3, chunk4, chunk5]  # from Step 2
```

**Output (the actual prompt sent to the LLM):**
```
Context information is below.
---------------------
[1] Source: paper_1.pdf, page 4
Evaluation of RAG systems typically involves metrics such as...

[2] Source: paper_3.pdf, page 7
BLEU and ROUGE scores have been adapted for RAG evaluation...

[3] Source: paper_2.pdf, page 12
Recent work has proposed using LLM-based evaluation...

[4] Source: paper_1.pdf, page 8
The limitations of automatic metrics include...

[5] Source: paper_4.pdf, page 3
Faithfulness metrics measure whether the generated response...
---------------------
Given the context information and not prior knowledge, answer the query.
Query: Compare the evaluation methods used for RAG systems and discuss their limitations
Answer:
```

**The problem with this step:** The chunks might not cover all aspects of the question. If the vector search missed chunks about "limitations" (because the query embedding emphasized "evaluation methods"), the LLM won't have that information.

---

### Step 4: LLM Generation

**What happens:** The LLM reads the prompt and generates an answer based on the provided context.

**Where:** `src/query_engine.py` → LlamaIndex calls the configured LLM (via OpenRouter)

**Why:** The LLM can understand the context and synthesize it into a coherent answer. It's doing the "reading comprehension" for us.

**Input:** The prompt from Step 3

**Output:**
```
Based on the provided context, RAG systems are typically evaluated using
several methods:

1. Traditional NLP metrics like BLEU and ROUGE, which measure n-gram overlap
   between generated and reference text.

2. LLM-based evaluation, where another language model judges the quality
   of responses.

3. Faithfulness metrics that check if the response is grounded in the
   retrieved context.

The main limitation mentioned is that automatic metrics don't always
correlate well with human judgment...
```

**The problem with this output:** The answer is incomplete. It only covers what was in the retrieved chunks. If the chunks didn't include information about all evaluation methods or all limitations, the answer misses those aspects. The LLM can only work with what it's given.

---

### Traditional RAG: Summary

```
┌─────────────────┐
│  User Question  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Embed Query    │  → Single vector for entire complex question
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Search  │  → Finds chunks similar to that ONE embedding
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stuff into     │  → Hopes the chunks cover all aspects
│  Prompt         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Generate   │  → Does its best with available context
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Answer         │  → May be incomplete for complex questions
└─────────────────┘
```

**Total LLM calls:** 1
**Total retrievals:** 1

---

## Part 2: Agentic RAG Flow (This Project)

This project adds three key steps: **Decomposition → Multi-hop Retrieval → Synthesis**

### Step 1: Query Decomposition

**What happens:** Before any retrieval, an LLM analyzes the question and breaks it into simpler sub-questions.

**Where:** `src/decomposition.py` → `decompose_query()` function

**Why:** Complex questions contain multiple implicit information needs. By decomposing, we can retrieve targeted information for each need separately, then combine the results.

**Input:**
```
"Compare the evaluation methods used for RAG systems and discuss their limitations"
```

**The decomposition prompt sent to LLM:**
```
You are a research assistant. Your task is to break down a complex question
into 2-4 simpler sub-questions that, when answered together, would fully
address the original question.

Guidelines:
- Each sub-question should be self-contained and answerable independently
- Sub-questions should cover different aspects of the original question
- Avoid redundant or overlapping sub-questions
- Output as a JSON array of strings

Original question: Compare the evaluation methods used for RAG systems and
discuss their limitations

Sub-questions:
```

**Output (from LLM):**
```json
[
    "What are the main evaluation methods used to assess RAG systems?",
    "How do automatic metrics like BLEU and ROUGE work for RAG evaluation?",
    "What are LLM-based evaluation approaches for RAG?",
    "What are the known limitations of current RAG evaluation methods?"
]
```

**Why this helps:** Each sub-question is more focused. When we embed "What are the known limitations of current RAG evaluation methods?", the resulting vector will be optimized for finding limitation-related content—something the original complex query's embedding wouldn't do as well.

---

### Step 2: Multi-hop Retrieval (Per Sub-question)

**What happens:** Each sub-question is processed independently: embedded, searched, and chunks retrieved.

**Where:** `src/retriever.py` → `retrieve()` called once per sub-question

**Why:** Each sub-question might need information from different "regions" of the vector space. A question about "limitations" will match different chunks than a question about "how BLEU works."

**Input:** The 4 sub-questions from Step 1

**Process for each sub-question:**

```
Sub-question 1: "What are the main evaluation methods used to assess RAG systems?"
  → Embed → Search → Top 5 chunks about evaluation methods overview

Sub-question 2: "How do automatic metrics like BLEU and ROUGE work for RAG evaluation?"
  → Embed → Search → Top 5 chunks specifically about BLEU/ROUGE

Sub-question 3: "What are LLM-based evaluation approaches for RAG?"
  → Embed → Search → Top 5 chunks about LLM-as-judge

Sub-question 4: "What are the known limitations of current RAG evaluation methods?"
  → Embed → Search → Top 5 chunks specifically about limitations
```

**Output:** 4 sets of chunks, each targeted to a specific aspect

```python
retrieved_chunks = {
    "sub_q_1": [chunk about methods overview, chunk about taxonomy, ...],
    "sub_q_2": [chunk about BLEU formula, chunk about ROUGE variants, ...],
    "sub_q_3": [chunk about GPT-4 as judge, chunk about evaluation prompts, ...],
    "sub_q_4": [chunk about metric limitations, chunk about bias issues, ...]
}
```

**Why this is better:** We now have chunks that specifically address each aspect of the question. The "limitations" sub-question retrieved chunks that might not have appeared in a single-pass retrieval because they weren't the top-5 most similar to the original complex query.

**Total retrievals:** 4 (one per sub-question)

---

### Step 3: Sub-answer Generation

**What happens:** For each sub-question, an LLM generates an answer using that sub-question's retrieved chunks.

**Where:** `src/agent.py` → inside `create_synthesis_agent()` → calls query engine per sub-question

**Why:** Each sub-answer is focused and grounded in relevant context. This is easier for the LLM than trying to answer a complex question with potentially noisy context.

**Input (for sub-question 4):**
```
Context:
[chunks specifically about limitations...]

Question: What are the known limitations of current RAG evaluation methods?
```

**Output (sub-answer 4):**
```
Current RAG evaluation methods have several known limitations:

1. **Automatic metrics (BLEU, ROUGE) limitations:**
   - Low correlation with human judgment for open-ended generation
   - Only measure surface-level similarity, not semantic correctness
   - Require reference answers, which may not exist for novel queries

2. **LLM-based evaluation limitations:**
   - Verbosity bias: longer answers often rated higher regardless of quality
   - Self-preference: models may prefer their own generation style
   - Inconsistency: same model can give different scores on repeated runs

3. **Faithfulness metrics limitations:**
   - Computationally expensive (require NLI models or LLM calls)
   - May not catch subtle hallucinations
   - Binary classification misses degrees of unfaithfulness

Sources: paper_2.pdf (p.12), paper_4.pdf (p.8)
```

**This sub-answer is better because:**
- The chunks were specifically retrieved for the "limitations" question
- The LLM could focus on one aspect without being distracted by other parts of the complex question
- The answer is detailed and comprehensive for this specific aspect

**Total LLM calls so far:** 1 (decomposition) + 4 (sub-answers) = 5

---

### Step 4: Synthesis

**What happens:** All sub-answers are combined, and an LLM synthesizes them into a final coherent answer that addresses the original complex question.

**Where:** `src/agent.py` → inside `create_synthesis_agent()` → synthesis prompt

**Why:** The sub-answers are individually good but disconnected. Synthesis creates a unified response that shows relationships, makes comparisons, and flows naturally.

**The synthesis prompt:**
```
You are synthesizing multiple sub-answers into a comprehensive response.

Original question: Compare the evaluation methods used for RAG systems and
discuss their limitations

Sub-answers:
---
Q1: What are the main evaluation methods used to assess RAG systems?
A1: [sub-answer 1 text]
Sources: paper_1.pdf (p.4), paper_3.pdf (p.2)
---
Q2: How do automatic metrics like BLEU and ROUGE work for RAG evaluation?
A2: [sub-answer 2 text]
Sources: paper_3.pdf (p.7), paper_1.pdf (p.5)
---
Q3: What are LLM-based evaluation approaches for RAG?
A3: [sub-answer 3 text]
Sources: paper_2.pdf (p.3), paper_4.pdf (p.1)
---
Q4: What are the known limitations of current RAG evaluation methods?
A4: [sub-answer 4 text]
Sources: paper_2.pdf (p.12), paper_4.pdf (p.8)
---

Instructions:
- Synthesize these sub-answers into a coherent, well-structured response
- Address ALL aspects of the original question
- Identify connections and contrasts between the sub-answers
- Cite sources for specific claims
- Do not add information not present in the sub-answers
```

**Output (final synthesized answer):**
```
## Evaluation Methods for RAG Systems

RAG systems are evaluated using three main approaches, each with distinct
strengths and limitations.

### 1. Traditional NLP Metrics (BLEU, ROUGE)

These metrics measure n-gram overlap between generated text and reference
answers. BLEU focuses on precision (how much of the generated text matches
the reference), while ROUGE emphasizes recall (how much of the reference
is captured). [paper_3.pdf, p.7]

**Limitations:** These metrics show low correlation with human judgment for
open-ended generation tasks. They measure surface-level similarity rather
than semantic correctness and require reference answers that may not exist
for novel queries. [paper_2.pdf, p.12]

### 2. LLM-Based Evaluation

Recent approaches use large language models as judges, prompting them to
rate response quality on dimensions like relevance, coherence, and
helpfulness. This enables evaluation without reference answers and can
assess subjective qualities. [paper_2.pdf, p.3]

**Limitations:** LLM judges exhibit verbosity bias (preferring longer
answers), self-preference (rating their own style higher), and
inconsistency across repeated evaluations. [paper_4.pdf, p.8]

### 3. Faithfulness Metrics

These specialized metrics verify whether the generated response is
grounded in the retrieved context, detecting hallucinations where the
model generates information not present in the sources. [paper_4.pdf, p.1]

**Limitations:** Faithfulness evaluation is computationally expensive,
may miss subtle hallucinations, and typically provides binary
classifications rather than nuanced assessments. [paper_2.pdf, p.12]

### Comparison

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| BLEU/ROUGE | Fast, reproducible | Poor semantic understanding |
| LLM-as-Judge | Nuanced, no references needed | Biased, inconsistent |
| Faithfulness | Detects hallucinations | Expensive, binary |

The field is moving toward hybrid approaches that combine multiple
evaluation methods to address individual limitations. [paper_1.pdf, p.4]
```

**Why this synthesis is better than traditional RAG:**
- Covers all aspects of the original question (methods AND limitations)
- Shows relationships between different methods
- Includes a comparison table that the LLM constructed from the evidence
- Well-organized with clear sections
- All claims are cited

**Total LLM calls:** 1 (decomposition) + 4 (sub-answers) + 1 (synthesis) = 6

---

### Agentic RAG: Summary

```
┌─────────────────┐
│  User Question  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Decompose      │  → LLM breaks into 2-4 focused sub-questions
│  (LLM call #1)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  For each sub-question:                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ Embed sub-Q │  │ Embed sub-Q │  │ Embed sub-Q │ ... │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
│         │                │                │             │
│         ▼                ▼                ▼             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ Vector      │  │ Vector      │  │ Vector      │ ... │
│  │ Search      │  │ Search      │  │ Search      │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
│         │                │                │             │
│         ▼                ▼                ▼             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ Generate    │  │ Generate    │  │ Generate    │ ... │
│  │ sub-answer  │  │ sub-answer  │  │ sub-answer  │     │
│  │ (LLM #2)    │  │ (LLM #3)    │  │ (LLM #4+)   │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
│         │                │                │             │
└─────────┼────────────────┼────────────────┼─────────────┘
          │                │                │
          └────────────────┼────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Synthesis      │  → Combine into coherent answer
                  │  (LLM call #N)  │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Final Answer   │  → Comprehensive, well-cited
                  └─────────────────┘
```

---

## Part 3: Side-by-Side Comparison

Using the same question: *"Compare the evaluation methods used for RAG systems and discuss their limitations"*

| Aspect | Traditional RAG | Agentic RAG (This Project) |
|--------|-----------------|---------------------------|
| **LLM calls** | 1 | 6 (1 decompose + 4 sub-answers + 1 synthesis) |
| **Retrievals** | 1 (5 chunks) | 4 (20 chunks total, 5 per sub-question) |
| **Query understanding** | None - uses query as-is | Decomposes into focused sub-questions |
| **Chunk relevance** | Mixed - single embedding can't capture all aspects | High - each retrieval targets specific aspect |
| **Answer completeness** | May miss aspects not in top-5 chunks | Covers all decomposed aspects |
| **Answer structure** | Often list-like, less organized | Synthesized, coherent narrative |
| **Latency** | Fast (~2-3 seconds) | Slower (~10-15 seconds) |
| **Cost** | 1 LLM call | 6 LLM calls |

### When to Use Which

**Use Traditional RAG when:**
- Questions are simple and factual ("What is RAG?")
- Latency is critical
- Cost is a concern
- The answer is likely contained in a small number of chunks

**Use Agentic RAG when:**
- Questions are complex, requiring multiple pieces of information
- Questions involve comparison, analysis, or synthesis
- Completeness is more important than speed
- The information is spread across multiple documents

---

## Part 4: Concrete Code Trace

Here's exactly what happens in code when you call the agentic RAG:

```python
# User calls:
from src.retriever import load_index
from src.agent import create_synthesis_agent

index = load_index()  # Load Chroma index from disk
agent = create_synthesis_agent(index)  # Create the agentic pipeline

result = agent("Compare evaluation methods for RAG and their limitations")
```

### Inside `create_synthesis_agent()`:

```python
def create_synthesis_agent(index):
    def agent(question: str) -> dict:
        # Step 1: Decompose
        sub_questions = decompose_query(question)
        # Returns: ["What are the main evaluation methods...",
        #           "How do automatic metrics work...", ...]

        # Step 2 & 3: Retrieve and answer each sub-question
        sub_answers = []
        all_sources = []

        query_engine = index.as_query_engine(similarity_top_k=5)

        for sub_q in sub_questions:
            # This does: embed → search → generate
            response = query_engine.query(sub_q)
            sub_answers.append({
                "question": sub_q,
                "answer": str(response),
                "sources": [node.metadata for node in response.source_nodes]
            })
            all_sources.extend(response.source_nodes)

        # Step 4: Synthesize
        synthesis_prompt = build_synthesis_prompt(question, sub_answers)
        final_response = llm.complete(synthesis_prompt)

        return {
            "answer": str(final_response),
            "sub_questions": sub_questions,
            "sub_answers": sub_answers,
            "sources": deduplicate_sources(all_sources)
        }

    return agent
```

### The return value structure:

```python
{
    "answer": "## Evaluation Methods for RAG Systems\n\nRAG systems are...",
    "sub_questions": [
        "What are the main evaluation methods used to assess RAG systems?",
        "How do automatic metrics like BLEU and ROUGE work for RAG evaluation?",
        "What are LLM-based evaluation approaches for RAG?",
        "What are the known limitations of current RAG evaluation methods?"
    ],
    "sub_answers": [
        {"question": "What are...", "answer": "The main methods...", "sources": [...]},
        {"question": "How do...", "answer": "BLEU measures...", "sources": [...]},
        # ...
    ],
    "sources": [
        {"source": "paper_1.pdf", "page": 4},
        {"source": "paper_2.pdf", "page": 12},
        # ... deduplicated list
    ]
}
```

---

## Key Takeaways

1. **Traditional RAG** is a single-pass pipeline that works well for simple queries but struggles with complex questions requiring multiple types of information.

2. **Agentic RAG** adds reasoning about the query (decomposition) and multiple targeted retrievals, trading latency and cost for comprehensiveness and quality.

3. **The "agentic" part** is the system's ability to plan (decompose), execute multiple operations (multi-hop), and reason (synthesize)—rather than blindly following a fixed pipeline.

4. **Trade-offs are explicit**: More LLM calls = more cost and latency, but better coverage for complex questions. The right choice depends on the use case.
