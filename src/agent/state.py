"""Conversation state and session context tracking."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class ConversationState:
    """Maintains multi-turn context, caller verification state, and tool results."""
    conversation_id: str
    turn: int = 0
    order_id: Optional[str] = None
    customer_id: Optional[str] = None
    is_verified: bool = False
    verification_attempts: int = 0
    pending_verification: bool = False
    order_data: Optional[Dict[str, Any]] = None
    escalated: bool = False
    ticket_id: Optional[str] = None
    language: str = "en"
    history: List[Dict[str, str]] = field(default_factory=list)
    last_evidence: List[str] = field(default_factory=list)

    def add_turn(self, role: str, message: str, evidence: Optional[List[str]] = None):
        """Append a message turn to history."""
        self.history.append({"role": role, "content": message})
        if evidence:
            self.last_evidence = evidence
        if role == "user":
            self.turn += 1

    def get_context_summary(self) -> str:
        """Create a compact conversational context string for LLM prompting."""
        lines = []
        for msg in self.history[-6:]:
            prefix = "Customer" if msg["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {msg['content']}")
        return "\n".join(lines)

