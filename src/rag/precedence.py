"""Document precedence and conflicting policy arbitration engine."""

from typing import List, Tuple
from src.rag.chunker import KBChunk


class PrecedenceRanker:
    """Enforces document precedence rules as defined in [KB-00.1].

    Precedence order:
    1. Current Policy (doc_type: policy, status: current)
    2. Agent Guides (doc_type: agent_guide)
    3. Help Articles (doc_type: help)
    4. FAQs (doc_type: faq)
    5. Marketing (doc_type: marketing)
    6. Superseded Documents (status: superseded)
    """

    # Multipliers applied to base retrieval score
    WEIGHTS = {
        ("policy", "current"): 1.30,
        ("meta", "current"): 1.20,
        ("agent_guide", "current"): 1.15,
        ("help", "current"): 1.00,
        ("faq", "current"): 0.85,
        ("marketing", "current"): 0.65,
        ("policy", "superseded"): 0.20,
    }

    # Known hard negative suppressions
    SUPPRESSIONS = {
        # If higher authority present -> suppress hard negative
        "REF-2": ["FAQ-1"],
        "KB-00.1": ["FAQ-1"],
        "RET-2": ["RETv3-2"],
        "RET-3": ["RETv3-2"],
        "RET-6": ["FAQ-2"],
    }

    @classmethod
    def get_weight(cls, chunk: KBChunk, query_text: str = "") -> float:
        """Calculate precedence multiplier for a chunk given the query."""
        status = chunk.status.lower()
        doc_type = chunk.doc_type.lower()

        # If user explicitly asks about legacy orders / before August 1, 2026
        if "before 1 august" in query_text.lower() or "before august" in query_text.lower():
            if status == "superseded" and "v3" in chunk.section_id.lower():
                return 1.40

        return cls.WEIGHTS.get((doc_type, status), 0.50 if status == "superseded" else 0.90)

    @classmethod
    def apply_precedence_and_filter(
        cls,
        scored_chunks: List[Tuple[KBChunk, float]],
        query_text: str = "",
        top_k: int = 4
    ) -> List[Tuple[KBChunk, float]]:
        """Re-weight retrieval scores by document precedence and suppress hard negatives."""
        weighted: List[Tuple[KBChunk, float]] = []
        for chunk, score in scored_chunks:
            weight = cls.get_weight(chunk, query_text)
            weighted.append((chunk, score * weight))

        # Sort descending by weighted score
        weighted.sort(key=lambda x: x[1], reverse=True)

        # Detect active high-authority section IDs
        active_ids = {chunk.section_id for chunk, _ in weighted[:top_k]}

        # Filter out hard negatives if suppressed
        filtered: List[Tuple[KBChunk, float]] = []
        for chunk, score in weighted:
            suppressed = False
            for auth_id, neg_ids in cls.SUPPRESSIONS.items():
                if auth_id in active_ids and chunk.section_id in neg_ids:
                    suppressed = True
                    break
            if not suppressed:
                filtered.append((chunk, score))

        return filtered[:top_k]

