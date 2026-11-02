"""Tests for Knowledge Base chunking, precedence filtering, and RAG retrieval."""

from src.rag.chunker import KBChunker
from src.rag.precedence import PrecedenceRanker
from src.rag.retriever import HybridRetriever, normalize_asr_query


def test_chunker_loads_all_sections():
    chunker = KBChunker()
    chunks = chunker.chunk_all()
    assert len(chunks) >= 80
    sec_ids = {c.section_id for c in chunks}
    assert "RET-2" in sec_ids
    assert "REF-2" in sec_ids
    assert "WAR-2" in sec_ids
    assert "KB-00.1" in sec_ids


def test_precedence_hard_negative_suppression():
    chunker = KBChunker()
    chunks = chunker.chunk_all()
    chunk_map = {c.section_id: c for c in chunks}

    # Simulate candidates containing both REF-2 (policy) and FAQ-1 (faq)
    candidates = [
        (chunk_map["FAQ-1"], 0.95),
        (chunk_map["REF-2"], 0.90),
    ]

    filtered = PrecedenceRanker.apply_precedence_and_filter(candidates, query_text="how long refunds take", top_k=2)
    filtered_ids = [c.section_id for c, _ in filtered]

    # REF-2 must be ranked above FAQ-1 and FAQ-1 should be suppressed
    assert "REF-2" in filtered_ids
    assert "FAQ-1" not in filtered_ids


def test_retriever_query_and_entities():
    retriever = HybridRetriever()

    # Test SKU detection and Assured warranty retrieval
    res = retriever.retrieve("SKU ELE-0012 pe ZentroMart Assured hai kya? Claim kaise karein?")
    assert "WAR-2" in res["evidence_list"]
    assert "products.csv:ELE-0012" in res["evidence_list"]

    # Test ASR normalization
    norm = normalize_asr_query("so sku ele-zero zero one to pe zentromart assured hai")
    assert "ELE-0012" in norm

