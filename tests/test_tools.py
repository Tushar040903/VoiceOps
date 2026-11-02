"""Tests for order lookup and ticket creation tools."""

from src.agent.tools import execute_lookup_order, execute_create_ticket


def test_order_lookup_unverified():
    res = execute_lookup_order("89782837")
    assert res.get("verification_required") is True
    # Private customer order fields must NOT be present
    assert "status" not in res
    assert "items" not in res
    assert "total_inr" not in res


def test_order_lookup_verified():
    # Order 89782837 registered phone ends with 4533, pincode is 781006
    res_phone = execute_lookup_order("89782837", verification_value="4533")
    assert res_phone.get("verified") is True
    assert res_phone.get("found") is True
    assert "items" in res_phone
    assert res_phone["customer_first_name"] == "Kabir"

    res_pin = execute_lookup_order("89782837", verification_value="781006")
    assert res_pin.get("verified") is True


def test_ticket_creation():
    # Valid ticket
    res = execute_create_ticket(
        summary="Customer reports broken screen on 43 inch TV. Verified caller. High value escalation.",
        order_id="89782837",
        reason="high_value_damage",
    )
    assert res.get("created") is True
    assert "ticket_id" in res

    # Too short summary fails validation (< 20 chars)
    res_fail = execute_create_ticket(summary="Short summary")
    assert res_fail.get("created") is False

