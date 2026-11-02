"""Tests for date calculations, business days, and holiday rules."""

from datetime import date
from src.data.date_utils import (
    is_business_day,
    add_business_days,
    count_business_days,
    is_return_eligible,
    format_spoken_date,
    format_written_date,
)


def test_business_days_and_holidays():
    # 2026-10-17 is Saturday -> business day
    assert is_business_day(date(2026, 10, 17)) is True

    # 2026-10-18 is Sunday -> non-business day
    assert is_business_day(date(2026, 10, 18)) is False

    # 2026-10-19 is Warehouse closure festival holiday -> non-business day
    assert is_business_day(date(2026, 10, 19)) is False

    # 2026-10-20 is Tuesday -> business day
    assert is_business_day(date(2026, 10, 20)) is True


def test_add_business_days():
    # Starting Saturday 17 Oct 2026:
    # 18 Oct (Sun) skip, 19 Oct (Holiday) skip.
    # Day 1: 20 Oct (Tue)
    # Day 2: 21 Oct (Wed)
    # Day 3: 22 Oct (Thu) -> UPI max SLA (+3 business days)
    res_upi = add_business_days("2026-10-17", 3)
    assert res_upi == date(2026, 10, 22)

    # Day 4: 23 Oct (Fri)
    # Day 5: 24 Oct (Sat) -> Net banking max SLA (+5 business days)
    res_net = add_business_days("2026-10-17", 5)
    assert res_net == date(2026, 10, 24)

    # 25 Oct (Sun) skip.
    # Day 6: 26 Oct (Mon)
    # Day 7: 27 Oct (Tue) -> Credit card max SLA (+7 business days)
    res_card = add_business_days("2026-10-17", 7)
    assert res_card == date(2026, 10, 27)


def test_return_eligibility():
    # Delivered on 2026-10-15, window 7 days -> deadline is 2026-10-22
    # Reference date is 2026-10-20 -> eligible
    assert is_return_eligible("2026-10-15", 7, ref_date=date(2026, 10, 20)) is True

    # Delivered on 2026-10-05, window 7 days -> deadline was 2026-10-12 -> expired
    assert is_return_eligible("2026-10-05", 7, ref_date=date(2026, 10, 20)) is False


def test_format_spoken_date():
    spoken = format_spoken_date("2026-10-24")
    assert spoken == "the 24th of October 2026"

