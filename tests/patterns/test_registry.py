"""Tests for the pattern registry."""

import pytest

from fyst_trajectories.patterns import (
    get_pattern,
    list_patterns,
)


class TestPatternRegistry:
    """Registry lookups: the sorted name list, the expected names, and unknown-key errors."""

    def test_list_patterns_returns_sorted_list(self):
        patterns = list_patterns()

        assert isinstance(patterns, list)
        assert len(patterns) > 0
        assert patterns == sorted(patterns)

    def test_all_expected_patterns_registered(self):
        patterns = list_patterns()

        expected = [
            "constant_el",
            "daisy",
            "daisy_altaz",
            "linear",
            "planet",
            "pong",
            "pong_altaz",
            "satellite",
            "sidereal",
        ]
        for name in expected:
            assert name in patterns, f"Expected pattern '{name}' not found"

    def test_get_pattern_returns_class(self):
        pattern_cls = get_pattern("pong")

        assert isinstance(pattern_cls, type)
        assert hasattr(pattern_cls, "generate")
        assert hasattr(pattern_cls, "get_metadata")

    def test_get_pattern_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown pattern 'nonexistent'"):
            get_pattern("nonexistent")

    def test_get_pattern_error_lists_available(self):
        with pytest.raises(KeyError) as exc_info:
            get_pattern("nonexistent")

        error_msg = str(exc_info.value)
        assert "pong" in error_msg
        assert "daisy" in error_msg


class TestRegistryErrorsAreInTheHierarchy:
    """Duplicate-registration errors are ``PointingError``, not bare ``ValueError``.

    ``PointingError`` subclasses ``ValueError``, so an ``except ValueError``
    catches them and a caller can also catch the library's own hierarchy. The
    unknown-name lookups keep raising ``KeyError``: those are genuine key
    misses on a registry dict and match the convention the rest of the
    package uses for unknown keys.
    """

    def test_duplicate_pattern_name_raises_pointing_error(self):
        from fyst_trajectories.exceptions import PointingError
        from fyst_trajectories.patterns.registry import register_pattern

        with pytest.raises(PointingError, match="already registered"):

            @register_pattern("pong")
            class _Duplicate:
                pass

    def test_duplicate_config_mapping_raises_pointing_error(self):
        """And a refused registration leaves the registry untouched.

        Writing the name into the registry before checking the config map
        would leave a half-registered pattern behind on a config-half clash,
        which ``list_patterns`` would then advertise.
        """
        from fyst_trajectories.exceptions import PointingError
        from fyst_trajectories.patterns import PongScanConfig
        from fyst_trajectories.patterns.registry import list_patterns, register_pattern

        before = list_patterns()
        with pytest.raises(PointingError, match="already mapped"):

            @register_pattern("pong_duplicate_config", config=PongScanConfig)
            class _Duplicate:
                pass

        assert list_patterns() == before

    @pytest.mark.parametrize("name", [None, 42, "", "   "], ids=["none", "int", "empty", "blank"])
    def test_a_name_that_is_not_a_usable_key_is_refused(self, name):
        """A registry key that cannot be sorted poisons ``list_patterns`` permanently.

        ``list_patterns`` sorts the keys, so a single ``None`` key makes it
        raise ``TypeError`` for the rest of the process; the name is checked
        before either registry is written.
        """
        from fyst_trajectories.exceptions import PointingError
        from fyst_trajectories.patterns.registry import list_patterns, register_pattern

        before = list_patterns()
        with pytest.raises(PointingError, match="non-blank string"):
            register_pattern(name)

        assert list_patterns() == before

    def test_unrecognised_config_on_the_builder_raises_pointing_error(self):
        from fyst_trajectories import get_fyst_site
        from fyst_trajectories.exceptions import PointingError
        from fyst_trajectories.patterns import TrajectoryBuilder

        class _NotAConfig:
            pass

        with pytest.raises(PointingError, match="Unknown config type"):
            TrajectoryBuilder(get_fyst_site()).with_config(_NotAConfig())
