from __future__ import annotations

from datetime import datetime


# Months considered "spring" for wet avalanche risk
_SPRING_MONTHS = {3, 4, 5}

# Hour ranges (inclusive start, exclusive end) in local time
_ELEVATED_HOURS = (10, 16)   # 10:00–15:59
_MODERATE_HOURS = (8, 18)    # 08:00–09:59 and 16:00–17:59


def wet_avalanche_risk(dt: datetime) -> dict:
    """
    Return a simple heuristic wet avalanche advisory for the given datetime.

    The heuristic is based on season and time of day only.  It does not
    account for aspect, elevation, recent weather, or snowpack structure.

    Returns a dict with keys:
      level  — "low" | "moderate" | "elevated"
      reason — human-readable explanation string
    """
    month = dt.month
    hour = dt.hour

    is_spring = month in _SPRING_MONTHS

    if is_spring and _ELEVATED_HOURS[0] <= hour < _ELEVATED_HOURS[1]:
        return {
            "level": "elevated",
            "reason": (
                f"Spring ({dt.strftime('%B')}) afternoon ({dt.strftime('%H:%M')}): "
                "solar warming peaks — wet slab and wet loose avalanches are likely "
                "on sun-exposed aspects. Consider an early start or avoid south/east "
                "faces after mid-morning."
            ),
        }

    if is_spring and _MODERATE_HOURS[0] <= hour < _MODERATE_HOURS[1]:
        return {
            "level": "moderate",
            "reason": (
                f"Spring ({dt.strftime('%B')}), time {dt.strftime('%H:%M')}: "
                "temperatures rising — wet avalanche activity may begin on "
                "south/southeast aspects. Monitor conditions closely."
            ),
        }

    if is_spring:
        return {
            "level": "low",
            "reason": (
                f"Spring ({dt.strftime('%B')}), early morning or evening "
                f"({dt.strftime('%H:%M')}): snowpack typically refrozen — "
                "wet avalanche hazard is lower, but verify overnight refreeze."
            ),
        }

    return {
        "level": "low",
        "reason": (
            f"{dt.strftime('%B')} — outside prime wet avalanche season. "
            "Standard dry slab assessment applies."
        ),
    }
