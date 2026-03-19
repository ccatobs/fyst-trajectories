"""Tests for the pattern registry."""

import pytest

from fyst_trajectories.patterns import (
    get_pattern,
    list_patterns,
)


class TestPatternRegistry:
    """Tests for pattern registry functions."""

    def test_list_patterns_returns_sorted_list(self):
        """Test that list_patterns returns a sorted list of patterns."""
        patterns = list_patterns()

        assert isinstance(patterns, list)
        assert len(patterns) > 0
        assert patterns == sorted(patterns)

    def test_all_expected_patterns_registered(self):
        """Test that all expected patterns are registered."""
        patterns = list_patterns()

        expected = ["constant_el", "daisy", "linear", "planet", "pong", "sidereal"]
        for name in expected:
            assert name in patterns, f"Expected pattern '{name}' not found"

    def test_get_pattern_returns_class(self):
        """Test that get_pattern returns a pattern class with required interface."""
        pattern_cls = get_pattern("pong")

        assert isinstance(pattern_cls, type)
        assert hasattr(pattern_cls, "generate")
        assert hasattr(pattern_cls, "get_metadata")

    def test_get_pattern_unknown_raises(self):
        """Test that get_pattern raises for unknown pattern."""
        with pytest.raises(KeyError, match="Unknown pattern 'nonexistent'"):
            get_pattern("nonexistent")

    def test_get_pattern_error_lists_available(self):
        """Test that error message lists available patterns."""
        try:
            get_pattern("nonexistent")
        except KeyError as e:
            error_msg = str(e)
            assert "pong" in error_msg
            assert "daisy" in error_msg
