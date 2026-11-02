"""Tests for ConversationalAgent multi-turn flows, verification, escalation, and voice normalizer."""

from src.agent.conversational_agent import ConversationalAgent
from src.agent.speech_normalizer import normalize_for_voice


def test_speech_normalization():
    raw = "Your reverse pickup fee is ₹49 on order 89782837 delivered on 2026-10-18."
    spoken = normalize_for_voice(raw)
    assert "forty-nine rupees" in spoken
    assert "8 9 7 8, 2 8 3 7" in spoken
    assert "the 18th of October 2026" in spoken


def test_agent_multi_turn_flow():
    agent = ConversationalAgent()
    sess_id = "test_multi_turn_session"

    # Turn 1: Opening greeting
    t1 = agent.respond("Hello, mujhe order ke baare mein help chahiye.", session_id=sess_id)
    assert t1["decision"] == "clarify"
    assert "STY-1" in t1["evidence_ids"]

    # Turn 2: Order ID provided
    t2 = agent.respond("Order number hai 8978 2837.", session_id=sess_id)
    assert t2["decision"] == "clarify"
    assert "lookup_order(order_id=89782837)" in t2["tool_calls"]
    assert "IDV-2" in t2["evidence_ids"]

    # Turn 3: Phone suffix verification
    t3 = agent.respond("Last four digits 4533 hain.", session_id=sess_id)
    assert t3["decision"] == "answer"
    assert any("4533" in tc for tc in t3["tool_calls"])
    assert t3["is_verified"] is True

    # Turn 4: High value damaged TV escalation
    t4 = agent.respond("Maine Zentro Vista 43-inch 4K Smart TV damage report kiya tha. Abhi replacement approve kar do na?", session_id=sess_id)
    assert t4["decision"] == "escalate"
    assert any("create_ticket" in tc for tc in t4["tool_calls"])
    assert "ESC-1" in t4["evidence_ids"]


def test_agent_refusal_on_failed_retries():
    agent = ConversationalAgent()
    sess_id = "test_refusal_session"

    agent.respond("Order number 8978 2837", session_id=sess_id)
    # Attempt 1 failed
    t_fail1 = agent.respond("Last 4 digits 0000", session_id=sess_id)
    assert t_fail1["decision"] == "clarify"

    # Attempt 2 failed (Exceeds 1 retry)
    t_fail2 = agent.respond("Last 4 digits 1111", session_id=sess_id)
    assert t_fail2["decision"] == "refuse"
    assert "IDV-3" in t_fail2["evidence_ids"]

