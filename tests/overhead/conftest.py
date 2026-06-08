"""Scheduling test fixtures."""

import pytest
from astropy.time import Time


@pytest.fixture
def start_time():
    """Nighttime observation start at FYST (UTC, ~22:00 local)."""
    return Time("2026-06-15T02:00:00", scale="utc")


@pytest.fixture
def end_time():
    """Observation end time (8 hours after start)."""
    return Time("2026-06-15T10:00:00", scale="utc")
