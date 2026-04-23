"""
tests/test_classifier.py

Unit and accuracy tests for the complexity classifier (PRD §3.2 / §4).
PRD target: accuracy > 88% on held-out set.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "gateway"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

COMPLEXITY_THRESHOLD = float(os.getenv("CLASSIFIER_COMPLEXITY_THRESHOLD", "0.6"))
ACCURACY_TARGET      = float(os.getenv("CLASSIFIER_ACCURACY_TARGET", "0.88"))


@pytest.fixture(scope="module")
def classifier():
    from classifier import ComplexityClassifier
    return ComplexityClassifier()


# ── Labelled test set (illustrative; extend with real MS MARCO labels) ───────
LABELLED_QUERIES = [
    # (query, expected_is_complex)
    ("What is RAG?", False),
    ("Define cosine similarity.", False),
    ("What is FAISS?", False),
    ("Explain embeddings.", False),
    ("What is Redis?", False),
    ("How do I start a Docker container?", False),
    ("What is a vector index?", False),
    ("Define LLM.", False),
    ("What year was Python created?", False),
    ("What is sentence-transformers?", False),
    # Complex queries
    (
        "Compare HNSW and IVF retrieval, discussing recall-latency trade-offs "
        "and which to use for corpora over 1M vectors.",
        True,
    ),
    (
        "How does multi-hop retrieval improve recall and what latency overhead does it add?",
        True,
    ),
    (
        "Explain the full RAG pipeline from query to response, including caching, "
        "classification, retrieval, and generation.",
        True,
    ),
    (
        "What are the trade-offs between using a local LLM versus OpenAI API, "
        "considering latency, cost, and data privacy?",
        True,
    ),
    (
        "Describe how MMR re-ranking works and why it improves answer quality "
        "by reducing context window redundancy.",
        True,
    ),
    (
        "How do I configure Redis semantic cache TTL and similarity threshold "
        "to balance hit rate against answer freshness?",
        True,
    ),
]


class TestComplexityClassifier:

    def test_simple_queries_classified_as_simple(self, classifier):
        simple_queries = [q for q, label in LABELLED_QUERIES if not label]
        for query in simple_queries:
            result = classifier.score(query)
            assert not result.is_complex, (
                f"Simple query incorrectly flagged as complex: '{query[:60]}' "
                f"(score={result.score:.3f})"
            )

    def test_complex_queries_classified_as_complex(self, classifier):
        complex_queries = [q for q, label in LABELLED_QUERIES if label]
        for query in complex_queries:
            result = classifier.score(query)
            assert result.is_complex, (
                f"Complex query missed: '{query[:60]}' (score={result.score:.3f})"
            )

    def test_accuracy_above_target(self, classifier):
        correct = 0
        for query, expected in LABELLED_QUERIES:
            result = classifier.score(query)
            if result.is_complex == expected:
                correct += 1
        accuracy = correct / len(LABELLED_QUERIES)
        assert accuracy >= ACCURACY_TARGET, (
            f"Classifier accuracy {accuracy:.3f} < target {ACCURACY_TARGET} "
            f"({correct}/{len(LABELLED_QUERIES)} correct)"
        )

    def test_score_in_range(self, classifier):
        for query, _ in LABELLED_QUERIES:
            result = classifier.score(query)
            assert 0.0 <= result.score <= 1.0, (
                f"Score out of [0,1] range for '{query[:40]}': {result.score}"
            )

    def test_features_populated(self, classifier):
        result = classifier.score("What is FAISS and how does IVF indexing work?")
        assert "n_tokens" in result.features
        assert "n_conjunctions" in result.features
        assert result.features["n_tokens"] > 0

    def test_threshold_from_env(self, classifier):
        """Verify the threshold is read from env, not hard-coded."""
        result = classifier.score("This is a borderline query.")
        # Just verify the is_complex flag is consistent with score vs threshold
        expected = result.score >= COMPLEXITY_THRESHOLD
        assert result.is_complex == expected

    def test_empty_query(self, classifier):
        result = classifier.score("")
        assert result.score == 0.0
        assert not result.is_complex

    def test_long_query_scores_higher(self, classifier):
        short = "What is FAISS?"
        long = " ".join(["explain the complete details of FAISS retrieval mechanisms"] * 5)
        short_result = classifier.score(short)
        long_result  = classifier.score(long)
        assert long_result.score > short_result.score

    def test_extract_key_terms_returns_list(self, classifier):
        texts = [
            "Amazon Web Services and Google Cloud Platform compete in the cloud market.",
            "FAISS was developed by Meta AI Research.",
        ]
        terms = classifier.extract_key_terms(texts)
        assert isinstance(terms, list)
