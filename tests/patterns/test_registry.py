"""Tests for the pattern registry."""

import typing

import pytest

from fyst_trajectories.patterns import (
    AltAzPattern,
    CelestialPattern,
    PlanetTrackPattern,
    PongScanPattern,
    SatelliteTrackPattern,
    ScanPattern,
    TrajectoryBuilder,
    TrajectoryMetadata,
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
        assert patterns == expected

    def test_get_pattern_returns_class(self):
        assert get_pattern("pong") is PongScanPattern

    def test_get_pattern_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown pattern 'nonexistent'"):
            get_pattern("nonexistent")

    def test_get_pattern_error_lists_available(self):
        with pytest.raises(KeyError) as exc_info:
            get_pattern("nonexistent")

        error_msg = str(exc_info.value)
        assert "pong" in error_msg
        assert "daisy" in error_msg


class TestRegistryArgumentErrorsArePlainValueError:
    """Registration and builder argument errors are plain ``ValueError``.

    A duplicate or unusable registration and an unrecognised builder config
    are malformed requests, so they raise ``ValueError`` and not
    ``PointingError``, which is kept for well-formed requests that cannot be
    satisfied. The unknown-name lookups keep raising ``KeyError``: those are
    genuine key misses on a registry dict and match the convention the rest
    of the package uses for unknown keys.
    """

    def test_duplicate_pattern_name_raises_value_error(self):
        from fyst_trajectories.exceptions import PointingError
        from fyst_trajectories.patterns.registry import register_pattern

        with pytest.raises(ValueError, match="already registered") as exc_info:

            @register_pattern("pong")
            class _Duplicate:
                pass

        assert not isinstance(exc_info.value, PointingError)

    def test_duplicate_config_mapping_raises_value_error(self):
        """And a refused registration leaves the registry untouched.

        Writing the name into the registry before checking the config map
        would leave a half-registered pattern behind on a config-half clash,
        which ``list_patterns`` would then advertise.
        """
        from fyst_trajectories.exceptions import PointingError
        from fyst_trajectories.patterns import PongScanConfig
        from fyst_trajectories.patterns.registry import list_patterns, register_pattern

        before = list_patterns()
        with pytest.raises(ValueError, match="already mapped") as exc_info:

            @register_pattern("pong_duplicate_config", config=PongScanConfig)
            class _Duplicate:
                pass

        assert not isinstance(exc_info.value, PointingError)
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
        with pytest.raises(ValueError, match="non-blank string") as exc_info:
            register_pattern(name)

        assert not isinstance(exc_info.value, PointingError)
        assert list_patterns() == before

    def test_unrecognised_config_on_the_builder_raises_value_error(self):
        """The registry's ``KeyError`` message is passed on without its quoting."""
        from fyst_trajectories import get_fyst_site
        from fyst_trajectories.exceptions import PointingError
        from fyst_trajectories.patterns import TrajectoryBuilder

        class _NotAConfig:
            pass

        with pytest.raises(ValueError, match="^Unknown config type") as exc_info:
            TrajectoryBuilder(get_fyst_site()).with_config(_NotAConfig())

        assert not isinstance(exc_info.value, PointingError)
        assert not str(exc_info.value).endswith("'")


# Whether each registered pattern needs a start time: the celestial patterns
# and the two ephemeris trackers do, the native az/el patterns do not.
_EXPECTED_REQUIRES_START_TIME = {
    "constant_el": False,
    "daisy": True,
    "daisy_altaz": False,
    "linear": False,
    "planet": True,
    "pong": True,
    "pong_altaz": False,
    "satellite": True,
    "sidereal": True,
}


class TestRegisteredName:
    """``register_pattern`` sets each pattern's ``name`` from the registered name."""

    @pytest.mark.parametrize("name", list_patterns())
    def test_class_attribute_is_the_registered_name(self, name):
        assert get_pattern(name).name == name

    def test_a_subclass_registered_under_its_own_name_reads_its_own(self):
        assert PlanetTrackPattern.name == "planet"
        assert SatelliteTrackPattern.name == "satellite"

    def test_a_registration_sets_the_name_on_the_class(self):
        from fyst_trajectories.patterns import registry

        name = "registry_test_throwaway"

        class _ThrowawayConfig:
            pass

        class _Throwaway(AltAzPattern):
            def generate(self, site, duration, start_time=None, atmosphere=None):
                raise NotImplementedError

            def get_metadata(self):
                return TrajectoryMetadata(pattern_type=self.name)

        try:
            registered = registry.register_pattern(name, config=_ThrowawayConfig)(_Throwaway)
            assert registered is _Throwaway
            assert _Throwaway.name == name
            assert _Throwaway().get_metadata().pattern_type == name
            assert get_pattern(name) is _Throwaway
            assert registry.get_pattern_for_config(_ThrowawayConfig) == name
        finally:
            registry._PATTERN_REGISTRY.pop(name, None)
            registry._CONFIG_TO_PATTERN_NAME.pop(_ThrowawayConfig, None)

        assert name not in list_patterns()

    def test_a_refused_registration_leaves_the_class_without_a_name(self):
        from fyst_trajectories.patterns.registry import register_pattern

        class _Duplicate:
            pass

        with pytest.raises(ValueError, match="already registered"):
            register_pattern("pong")(_Duplicate)

        assert "name" not in vars(_Duplicate)

    def test_a_refused_config_mapping_leaves_the_class_without_a_name(self):
        from fyst_trajectories.patterns import PongScanConfig
        from fyst_trajectories.patterns.registry import register_pattern

        class _Duplicate:
            pass

        with pytest.raises(ValueError, match="already mapped"):
            register_pattern("registry_test_unused", config=PongScanConfig)(_Duplicate)

        assert "name" not in vars(_Duplicate)
        assert "registry_test_unused" not in list_patterns()


class TestPatternInterface:
    """The ``ScanPattern`` protocol lists what the builder reads from a pattern class."""

    def test_protocol_declares_name_and_requires_start_time(self):
        hints = typing.get_type_hints(ScanPattern)

        assert hints["name"] == typing.ClassVar[str]
        assert hints["requires_start_time"] == typing.ClassVar[bool]

    def test_a_structural_pattern_must_declare_requires_start_time(self):
        class _Structural:
            name = "structural"

            def generate(self, site, duration, start_time, atmosphere=None):
                raise NotImplementedError

            def get_metadata(self):
                raise NotImplementedError

        assert not isinstance(_Structural(), ScanPattern)

        _Structural.requires_start_time = False

        assert isinstance(_Structural(), ScanPattern)

    @pytest.mark.parametrize("base", [CelestialPattern, AltAzPattern])
    def test_name_is_not_an_abstract_member_of_either_base(self, base):
        assert base.__abstractmethods__ == frozenset({"generate", "get_metadata"})

    @pytest.mark.parametrize("name", list_patterns())
    def test_requires_start_time_is_the_bool_the_builder_reads(self, name):
        pattern_cls = get_pattern(name)

        assert isinstance(pattern_cls.requires_start_time, bool)
        assert pattern_cls.requires_start_time is TrajectoryBuilder._needs_start_time(pattern_cls)
        assert pattern_cls.requires_start_time is _EXPECTED_REQUIRES_START_TIME[name]
