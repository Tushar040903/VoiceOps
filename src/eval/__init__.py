"""Evaluation package for scoring RAG retrieval, decision accuracy, and multi-turn conversations."""

from src.eval.metrics import calculate_retrieval_metrics, check_must_include, check_must_not_include
from src.eval.evaluator import BenchmarkEvaluator

__all__ = ["calculate_retrieval_metrics", "check_must_include", "check_must_not_include", "BenchmarkEvaluator"]

