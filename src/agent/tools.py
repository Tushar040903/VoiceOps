"""Tool wrappers for order verification, database lookup, and human handover tickets."""

import sys
import logging
from typing import Dict, Any, Optional
from src.config import TOOLS_DIR, DATASET_DIR

logger = logging.getLogger(__name__)

# Ensure dataset tools folder is importable
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

try:
    from tools.tools import lookup_order as _raw_lookup_order
    from tools.tools import create_ticket as _raw_create_ticket
except ImportError:
    # Fallback import directly from tools module
    try:
        from tools import lookup_order as _raw_lookup_order
        from tools import create_ticket as _raw_create_ticket
    except ImportError:
        logger.warning("Could not import raw mock tools. Using internal fallback.")
        _raw_lookup_order = None
        _raw_create_ticket = None


def execute_lookup_order(order_id: str, verification_value: Optional[str] = None) -> Dict[str, Any]:
    """Execute order lookup with caller identity verification."""
    clean_oid = "".join(ch for ch in str(order_id) if ch.isdigit())
    if _raw_lookup_order:
        return _raw_lookup_order(clean_oid, verification_value=verification_value)

    # In case external import fails, safe stub
    return {
        "verification_required": True,
        "order_id": clean_oid,
        "message": "Ask for the registered phone last 4 digits OR delivery pincode."
    }


def execute_create_ticket(
    summary: str,
    order_id: Optional[str] = None,
    reason: Optional[str] = None,
    intent: Optional[str] = None
) -> Dict[str, Any]:
    """Execute ticket creation for escalation to human support agents."""
    if _raw_create_ticket:
        return _raw_create_ticket(summary=summary, order_id=order_id, reason=reason, intent=intent)

    from uuid import uuid4
    return {"created": True, "ticket_id": f"TKT{uuid4().hex[:12]}"}

