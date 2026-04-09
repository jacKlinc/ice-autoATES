from datetime import datetime

import pytest

from ates.advisory import wet_avalanche_risk


@pytest.mark.parametrize(
    "dt, expected_level",
    [
        # Spring afternoon → elevated
        (datetime(2025, 4, 15, 13, 0), "elevated"),
        (datetime(2025, 3, 20, 10, 0), "elevated"),
        (datetime(2025, 5, 1, 15, 59), "elevated"),
        # Spring early morning → low
        (datetime(2025, 4, 15, 6, 0), "low"),
        (datetime(2025, 3, 20, 7, 59), "low"),
        # Spring morning transition / evening → moderate
        (datetime(2025, 4, 15, 8, 0), "moderate"),
        (datetime(2025, 4, 15, 9, 30), "moderate"),
        (datetime(2025, 4, 15, 16, 0), "moderate"),
        (datetime(2025, 4, 15, 17, 59), "moderate"),
        # Winter (non-spring months) → always low regardless of time
        (datetime(2025, 1, 15, 13, 0), "low"),
        (datetime(2025, 12, 25, 14, 0), "low"),
        (datetime(2025, 2, 10, 11, 0), "low"),
        # Autumn → low
        (datetime(2025, 10, 15, 13, 0), "low"),
        # Summer → low (not in spring months definition)
        (datetime(2025, 7, 15, 13, 0), "low"),
    ],
)
def test_advisory_level(dt, expected_level):
    result = wet_avalanche_risk(dt)
    assert result["level"] == expected_level


def test_advisory_has_reason():
    result = wet_avalanche_risk(datetime(2025, 4, 15, 13, 0))
    assert "reason" in result
    assert len(result["reason"]) > 0


def test_elevated_reason_mentions_solar():
    result = wet_avalanche_risk(datetime(2025, 4, 15, 13, 0))
    assert "solar" in result["reason"].lower() or "wet" in result["reason"].lower()


def test_low_winter_reason_mentions_dry_slab():
    result = wet_avalanche_risk(datetime(2025, 1, 15, 13, 0))
    assert "dry slab" in result["reason"].lower()
