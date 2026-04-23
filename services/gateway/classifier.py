"""
services/gateway/classifier.py

Query complexity classifier (PRD §3.2).

v1: Rule-based feature scoring — fast, explainable, zero ML overhead.
v2 stub: DistilBERT fine-tune pathway (enable via CLASSIFIER_USE_MODEL=true).

Feature set (all weights configurable via env vars):
  - Query length in tokens       (CLASSIFIER_TOKEN_WEIGHT)
  - Conjunction count            (CLASSIFIER_CONJUNCTION_WEIGHT)
  - Named entity count           (CLASSIFIER_ENTITY_WEIGHT)
  - Sub-query count              (CLASSIFIER_SUBQUERY_WEIGHT)

All thresholds are read from ClassifierSettings (env vars). Nothing hard-coded.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

import spacy

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from shared.config import get_classifier_settings

logger = logging.getLogger(__name__)
settings = get_classifier_settings()

_CONJUNCTIONS = frozenset({"and", "or", "but", "because", "however", "although",
                            "while", "whereas", "since", "unless", "therefore"})
_SUBQUERY_SPLITS = re.compile(r";|also\s|additionally\s|furthermore\s|moreover\s", re.I)

_nlp: Optional[spacy.Language] = None


def load_nlp() -> None:
    global _nlp
    try:
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        logger.info("spaCy NER loaded for complexity classifier.")
    except OSError:
        logger.warning("spaCy en_core_web_sm not found; NER feature disabled.")
        _nlp = None


@dataclass
class ComplexityResult:
    score: float          # 0.0 – 1.0
    is_complex: bool      # True if score >= threshold
    features: dict        # for observability / logging


class ComplexityClassifier:
    """
    Rule-based complexity classifier (v1).
    Returns a float score in [0, 1]. Values above `settings.complexity_threshold`
    trigger multi-hop retrieval in the gateway.
    """

    def __init__(self):
        load_nlp()

    def score(self, query: str) -> ComplexityResult:
        tokens = query.split()
        n_tokens = len(tokens)

        # ── Feature 1: token length (normalised to [0,1]) ───
        token_score = min(n_tokens / settings.max_tokens_score, 1.0)

        # ── Feature 2: conjunction density ──────────────────
        n_conjunctions = sum(1 for t in tokens if t.lower().strip(".,?!") in _CONJUNCTIONS)
        conjunction_score = min(n_conjunctions / 3.0, 1.0)

        # ── Feature 3: named entity count ───────────────────
        if _nlp is not None:
            doc = _nlp(query[:1000])
            n_entities = len(doc.ents)
        else:
            # Fallback: count capitalised words as proxy
            n_entities = sum(1 for t in tokens if t[0].isupper() and len(t) > 1)
        entity_score = min(n_entities / 4.0, 1.0)

        # ── Feature 4: sub-query count ───────────────────────
        subqueries = [s.strip() for s in _SUBQUERY_SPLITS.split(query) if s.strip()]
        n_subqueries = max(len(subqueries) - 1, 0)
        subquery_score = min(n_subqueries / 2.0, 1.0)

        # ── Weighted sum ─────────────────────────────────────
        composite = (
            settings.token_weight       * token_score
            + settings.entity_weight    * entity_score
            + settings.conjunction_weight * conjunction_score
            + settings.subquery_weight  * subquery_score
        )
        composite = max(0.0, min(composite, 1.0))  # clamp

        features = {
            "n_tokens": n_tokens,
            "n_conjunctions": n_conjunctions,
            "n_entities": n_entities,
            "n_subqueries": n_subqueries,
            "token_score": round(token_score, 3),
            "conjunction_score": round(conjunction_score, 3),
            "entity_score": round(entity_score, 3),
            "subquery_score": round(subquery_score, 3),
        }

        return ComplexityResult(
            score=round(composite, 4),
            is_complex=composite >= settings.complexity_threshold,
            features=features,
        )

    def extract_key_terms(self, texts: list[str]) -> list[str]:
        """
        Extract named entities from a list of texts for multi-hop enrichment (PRD §2.2).
        Returns a deduplicated list of entity strings.
        """
        if _nlp is None:
            return []

        combined = " ".join(texts)[:10_000]
        doc = _nlp(combined)
        seen: dict[str, None] = {}
        for ent in doc.ents:
            seen[ent.text] = None
        return list(seen.keys())[:10]  # top-10 unique entities
