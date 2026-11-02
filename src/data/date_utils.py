"""Calendar, business day, and date calculation utilities for ZentroMart."""

from datetime import date, datetime, timedelta
from typing import Set, Union
from src.config import REFERENCE_DATE

# Known warehouse holidays from holidays.csv
DEFAULT_HOLIDAYS: Set[date] = {
    date(2026, 8, 15),  # Independence Day
    date(2026, 10, 2),   # Gandhi Jayanti
    date(2026, 10, 19),  # Warehouse closure (festival)
    date(2026, 11, 9),   # Warehouse closure (festival)
}


def parse_date(d: Union[str, date, datetime, None]) -> Union[date, None]:
    """Parse date from string or date object. Handles None gracefully."""
    if d is None:
        return None
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    try:
        s = str(d).strip()
        if not s:
            return None
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def is_business_day(d: Union[str, date], holidays: Set[date] = None) -> bool:
    """Check if date is a business day (Monday=0 to Saturday=5, excluding holidays)."""
    parsed = parse_date(d)
    if not parsed:
        return False
    active_holidays = holidays if holidays is not None else DEFAULT_HOLIDAYS
    # Monday to Saturday (0 to 5) are business days. Sunday (6) is non-business.
    if parsed.weekday() == 6:
        return False
    return parsed not in active_holidays


def add_business_days(start: Union[str, date], n: int, holidays: Set[date] = None) -> date:
    """Add n business days to start_date, skipping Sundays and holidays."""
    cur = parse_date(start) or REFERENCE_DATE
    active_holidays = holidays if holidays is not None else DEFAULT_HOLIDAYS
    added = 0
    step = 1 if n >= 0 else -1
    target = abs(n)

    while added < target:
        cur += timedelta(days=step)
        if is_business_day(cur, active_holidays):
            added += 1

    return cur


def count_business_days(start: Union[str, date], end: Union[str, date], holidays: Set[date] = None) -> int:
    """Count business days between start (exclusive) and end (inclusive)."""
    d1 = parse_date(start)
    d2 = parse_date(end)
    if not d1 or not d2 or d1 >= d2:
        return 0

    active_holidays = holidays if holidays is not None else DEFAULT_HOLIDAYS
    count = 0
    cur = d1 + timedelta(days=1)
    while cur <= d2:
        if is_business_day(cur, active_holidays):
            count += 1
        cur += timedelta(days=1)

    return count


def is_return_eligible(delivery_date: Union[str, date], window_days: int, ref_date: date = REFERENCE_DATE) -> bool:
    """Check return eligibility using calendar days from delivery date."""
    d = parse_date(delivery_date)
    if not d or window_days <= 0:
        return False
    deadline = d + timedelta(days=window_days)
    return ref_date <= deadline


def days_since_delivery(delivery_date: Union[str, date], ref_date: date = REFERENCE_DATE) -> int:
    """Return calendar days between delivery and reference date."""
    d = parse_date(delivery_date)
    if not d:
        return 9999
    return (ref_date - d).days


def format_spoken_date(d: Union[str, date, datetime, None]) -> str:
    """Format date for voice output ('the 24th of October 2026')."""
    parsed = parse_date(d)
    if not parsed:
        return ""
    day = parsed.day
    if 11 <= (day % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    month_name = parsed.strftime("%B")
    return f"the {day}{suffix} of {month_name} {parsed.year}"


def format_written_date(d: Union[str, date, datetime, None]) -> str:
    """Format date for written chat ('24 October 2026')."""
    parsed = parse_date(d)
    if not parsed:
        return ""
    return parsed.strftime("%d %B %Y").lstrip("0")

