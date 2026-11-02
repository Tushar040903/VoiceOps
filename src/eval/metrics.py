"""Evaluation metrics for RAG retrieval, decision accuracy, and groundedness."""

import re
from typing import Dict, List, Optional, Set, Tuple, Any


def parse_gold_evidence(gold_str: Optional[str]) -> Set[str]:
    """Parse gold evidence string into a set of normalized identifiers."""
    if not gold_str or str(gold_str).strip() in ("", "nan", "None"):
        return set()
    parts = str(gold_str).split(";")
    return {p.strip() for p in parts if p.strip()}


def calculate_retrieval_metrics(retrieved_ids: List[str], gold_str: Optional[str]) -> Dict[str, float]:
    """Calculate retrieval Precision, Recall, and F1 against gold evidence."""
    gold_set = parse_gold_evidence(gold_str)
    retrieved_set = set(retrieved_ids)

    # If question has no gold evidence (unanswerable question)
    if not gold_set:
        return {
            "precision": 1.0 if not retrieved_set else 0.5,
            "recall": 1.0,
            "f1": 1.0 if not retrieved_set else 0.67,
            "hit": 1.0 if not retrieved_set else 0.0
        }

    intersection = gold_set.intersection(retrieved_set)
    precision = len(intersection) / len(retrieved_set) if retrieved_set else 0.0
    recall = len(intersection) / len(gold_set) if gold_set else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    hit = 1.0 if len(intersection) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "hit": hit
    }


def check_must_include(text: str, must_include_str: Optional[str]) -> Tuple[bool, List[str]]:
    """Check if text contains required facts. Supports ' ; ' (AND) and ' / ' (OR)."""
    if not must_include_str or str(must_include_str).strip() in ("", "nan", "None"):
        return True, []

    text_lower = text.lower()
    required_clauses = str(must_include_str).split(" ; ")
    missing = []

    for clause in required_clauses:
        alternatives = [alt.strip().lower() for alt in clause.split(" / ")]
        matched = any(alt in text_lower for alt in alternatives if alt)
        if not matched:
            missing.append(clause)

    return (len(missing) == 0), missing


def check_must_not_include(text: str, must_not_include_str: Optional[str]) -> Tuple[bool, List[str]]:
    """Check that text does NOT contain prohibited / hallucinated facts."""
    if not must_not_include_str or str(must_not_include_str).strip() in ("", "nan", "None"):
        return True, []

    text_lower = text.lower()
    forbidden = [f.strip().lower() for f in str(must_not_include_str).split(" ; ") if f.strip()]
    violations = [f for f in forbidden if f in text_lower]

    return (len(violations) == 0), violations

