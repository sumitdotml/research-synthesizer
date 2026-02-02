"""Evaluation module for comparing baseline vs agentic RAG."""

import json
import time
from typing import Any

from src.config import get_llm_with_fallback
from src.retriever import load_index
from src.query_engine import create_query_engine, query_with_sources
from src.agent import create_synthesis_agent


RELEVANCE_PROMPT = """You are evaluating the relevance of an answer to a question about research papers.

Question: {question}
Expected topics: {expected_topics}
Answer: {answer}

Rate the answer on these criteria (1-5 scale):
1. Relevance: Does the answer address the question? (1=off-topic, 5=directly answers)
2. Coverage: Does the answer cover the expected topics? (1=none, 5=comprehensive)
3. Coherence: Is the answer well-structured and clear? (1=incoherent, 5=very clear)

Respond with ONLY a JSON object:
{{"relevance": <1-5>, "coverage": <1-5>, "coherence": <1-5>, "reasoning": "<brief explanation>"}}"""


def load_test_questions(path: str = "data/test_questions.json") -> list[dict]:
    """Load test questions from JSON file."""
    with open(path) as f:
        return json.load(f)


def evaluate_answer(
    question: str,
    answer: str,
    expected_topics: list[str],
    llm=None,
) -> dict[str, Any]:
    """
    Evaluate an answer using LLM-as-judge.

    Args:
        question: The original question
        answer: The generated answer
        expected_topics: Topics the answer should cover
        llm: LLM to use for evaluation

    Returns:
        Dict with relevance, coverage, coherence scores and reasoning
    """
    if llm is None:
        llm = get_llm_with_fallback()

    prompt = RELEVANCE_PROMPT.format(
        question=question,
        expected_topics=", ".join(expected_topics),
        answer=answer[:2000],  # truncating long answers
    )

    response = llm.complete(prompt)

    try:
        text = response.text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        scores = json.loads(text)
        return {
            "relevance": scores.get("relevance", 3),
            "coverage": scores.get("coverage", 3),
            "coherence": scores.get("coherence", 3),
            "reasoning": scores.get("reasoning", ""),
        }
    except (json.JSONDecodeError, KeyError):
        return {
            "relevance": 3,
            "coverage": 3,
            "coherence": 3,
            "reasoning": "Failed to parse evaluation",
        }


def run_evaluation(
    questions: list[dict],
    use_agentic: bool = False,
) -> list[dict]:
    """
    Run evaluation on a set of questions.

    Args:
        questions: List of question dicts with 'question' and 'expected_topics'
        use_agentic: Whether to use the agentic (synthesis) approach

    Returns:
        List of result dicts with question, answer, scores, and timing
    """
    print("Loading index...")
    index = load_index()
    llm = get_llm_with_fallback()

    if use_agentic:
        print("Creating synthesis agent...")
        agent = create_synthesis_agent(index)
    else:
        print("Creating basic query engine...")
        query_engine = create_query_engine(index)

    results = []

    for i, q in enumerate(questions):
        print(f"\n[{i + 1}/{len(questions)}] {q['question'][:50]}...")

        start_time = time.time()

        try:
            if use_agentic:
                result = agent(q["question"])
                answer = result["answer"]
                sources = result["sources"]
                sub_questions = result.get("sub_questions", [])
            else:
                result = query_with_sources(query_engine, q["question"])
                answer = result["answer"]
                sources = result["sources"]
                sub_questions = []

            elapsed = time.time() - start_time

            scores = evaluate_answer(
                q["question"],
                answer,
                q.get("expected_topics", []),
                llm,
            )

            results.append(
                {
                    "question_id": q["id"],
                    "question": q["question"],
                    "complexity": q.get("complexity", "unknown"),
                    "answer": answer,
                    "sources": [s["title"] for s in sources],
                    "sub_questions": sub_questions,
                    "scores": scores,
                    "time_seconds": elapsed,
                }
            )

        except Exception as e:
            print(f"  Error: {e}")
            results.append(
                {
                    "question_id": q["id"],
                    "question": q["question"],
                    "complexity": q.get("complexity", "unknown"),
                    "answer": f"Error: {str(e)}",
                    "sources": [],
                    "sub_questions": [],
                    "scores": {"relevance": 0, "coverage": 0, "coherence": 0},
                    "time_seconds": 0,
                }
            )

    return results


def compare_approaches(questions: list[dict]) -> dict:
    """
    Compare baseline RAG vs agentic RAG on test questions.

    Args:
        questions: List of test questions

    Returns:
        Comparison results with metrics for both approaches
    """
    print("\n" + "=" * 60)
    print("Running Baseline RAG Evaluation")
    print("=" * 60)
    baseline_results = run_evaluation(questions, use_agentic=False)

    print("\n" + "=" * 60)
    print("Running Agentic RAG Evaluation")
    print("=" * 60)
    agentic_results = run_evaluation(questions, use_agentic=True)

    def calc_metrics(results):
        scores = [r["scores"] for r in results]
        times = [r["time_seconds"] for r in results]
        return {
            "avg_relevance": sum(s["relevance"] for s in scores) / len(scores),
            "avg_coverage": sum(s["coverage"] for s in scores) / len(scores),
            "avg_coherence": sum(s["coherence"] for s in scores) / len(scores),
            "avg_time": sum(times) / len(times),
            "total_time": sum(times),
        }

    baseline_metrics = calc_metrics(baseline_results)
    agentic_metrics = calc_metrics(agentic_results)

    return {
        "baseline": {
            "results": baseline_results,
            "metrics": baseline_metrics,
        },
        "agentic": {
            "results": agentic_results,
            "metrics": agentic_metrics,
        },
        "comparison": {
            "relevance_improvement": agentic_metrics["avg_relevance"]
            - baseline_metrics["avg_relevance"],
            "coverage_improvement": agentic_metrics["avg_coverage"]
            - baseline_metrics["avg_coverage"],
            "coherence_improvement": agentic_metrics["avg_coherence"]
            - baseline_metrics["avg_coherence"],
            "time_overhead": agentic_metrics["avg_time"] - baseline_metrics["avg_time"],
        },
    }


def print_comparison_table(comparison: dict):
    """Print a formatted comparison table."""
    baseline = comparison["baseline"]["metrics"]
    agentic = comparison["agentic"]["metrics"]
    diff = comparison["comparison"]

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\n{'Metric':<20} {'Baseline':<12} {'Agentic':<12} {'Diff':<12}")
    print("-" * 56)
    print(
        f"{'Relevance':<20} {baseline['avg_relevance']:<12.2f} {agentic['avg_relevance']:<12.2f} {diff['relevance_improvement']:+.2f}"
    )
    print(
        f"{'Coverage':<20} {baseline['avg_coverage']:<12.2f} {agentic['avg_coverage']:<12.2f} {diff['coverage_improvement']:+.2f}"
    )
    print(
        f"{'Coherence':<20} {baseline['avg_coherence']:<12.2f} {agentic['avg_coherence']:<12.2f} {diff['coherence_improvement']:+.2f}"
    )
    print(
        f"{'Avg Time (s)':<20} {baseline['avg_time']:<12.2f} {agentic['avg_time']:<12.2f} {diff['time_overhead']:+.2f}"
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    overall_baseline = (
        baseline["avg_relevance"] + baseline["avg_coverage"] + baseline["avg_coherence"]
    ) / 3
    overall_agentic = (
        agentic["avg_relevance"] + agentic["avg_coverage"] + agentic["avg_coherence"]
    ) / 3

    print(f"\nBaseline Overall Score: {overall_baseline:.2f}/5")
    print(f"Agentic Overall Score:  {overall_agentic:.2f}/5")
    print(
        f"Improvement: {(overall_agentic - overall_baseline):+.2f} ({((overall_agentic / overall_baseline - 1) * 100):+.1f}%)"
    )


if __name__ == "__main__":
    questions = load_test_questions()
    print(f"Loaded {len(questions)} test questions")

    quick_questions = questions[:3]
    comparison = compare_approaches(quick_questions)

    print_comparison_table(comparison)

    with open("data/evaluation_results.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print("\nFull results saved to data/evaluation_results.json")
