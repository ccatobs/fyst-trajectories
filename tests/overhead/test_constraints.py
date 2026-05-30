"""Tests for scheduling constraints."""

import pytest
from astropy.time import Time

from fyst_trajectories import Coordinates, get_fyst_site
from fyst_trajectories.overhead.constraints import (
    ElevationConstraint,
    MinDurationConstraint,
    MoonAvoidanceConstraint,
    SunAvoidanceConstraint,
)
from fyst_trajectories.overhead.models import ObservingPatch


@pytest.fixture
def patch():
    return ObservingPatch(
        name="test",
        ra_center=180.0,
        dec_center=-30.0,
        width=4.0,
        height=4.0,
        scan_type="pong",
        velocity=0.5,
    )


@pytest.fixture
def coords():
    return Coordinates(get_fyst_site())


@pytest.fixture
def time():
    return Time("2026-06-15T04:00:00", scale="utc")


class TestElevationConstraint:
    """Tests for ElevationConstraint."""

    def test_in_range(self, patch, time, coords):
        c = ElevationConstraint(el_min=30.0, el_max=80.0)
        assert c.score(patch, time, 180.0, 50.0, coords) == 1.0

    def test_below_min(self, patch, time, coords):
        c = ElevationConstraint(el_min=30.0, el_max=80.0)
        assert c.score(patch, time, 180.0, 20.0, coords) == 0.0

    def test_above_max(self, patch, time, coords):
        c = ElevationConstraint(el_min=30.0, el_max=80.0)
        assert c.score(patch, time, 180.0, 85.0, coords) == 0.0

    def test_at_boundary(self, patch, time, coords):
        c = ElevationConstraint(el_min=30.0, el_max=80.0)
        assert c.score(patch, time, 180.0, 30.0, coords) == 1.0
        assert c.score(patch, time, 180.0, 80.0, coords) == 1.0

    def test_invalid_range(self):
        with pytest.raises(ValueError, match="el_min"):
            ElevationConstraint(el_min=80.0, el_max=30.0)


class TestSunAvoidanceConstraint:
    """Tests for SunAvoidanceConstraint."""

    def test_safe_position(self, patch, time, coords):
        c = SunAvoidanceConstraint(min_angle=45.0)
        sun_az, sun_el = coords.get_sun_altaz(time)
        safe_az = (sun_az + 180.0) % 360.0
        score = c.score(patch, time, safe_az, 50.0, coords)
        assert score == 1.0

    def test_near_sun(self, patch, coords):
        """A point at the Sun's position scores 0.0 (inside the exclusion angle)."""
        daytime = Time("2026-06-15T16:00:00", scale="utc")
        c = SunAvoidanceConstraint(min_angle=45.0)
        sun_az, sun_el = coords.get_sun_altaz(daytime)
        # Point the target AT the Sun: separation = 0 < min_angle. The score is a
        # pure angular-separation check (it ignores horizon), so this is
        # deterministic regardless of whether the Sun is currently up.
        score = c.score(patch, daytime, sun_az, sun_el, coords)
        assert score == 0.0

    def test_negative_angle(self):
        with pytest.raises(ValueError, match="non-negative"):
            SunAvoidanceConstraint(min_angle=-1.0)


class TestMoonAvoidanceConstraint:
    """Tests for MoonAvoidanceConstraint.

    Mirrors :class:`TestSunAvoidanceConstraint` so a typo of
    ``min_angle`` -> ``max_angle`` (or any sign-flip in the comparison
    against the Moon separation) fails closed. The asymmetry between
    moon and sun safety in the planner layer is intentional (see the
    constraint class docstring) but the scheduler-side score logic
    must remain symmetric.
    """

    def test_safe_position(self, patch, coords):
        """A point antipodal to the Moon scores 1.0 (well outside threshold)."""
        # Use a time when the Moon is above the FYST horizon. The score is
        # purely an angular-separation check and ignores horizon, so the
        # antipodal point exercises the safe-position branch even though
        # it is itself below horizon.
        moon_up_time = Time("2026-06-15T18:00:00", scale="utc")
        c = MoonAvoidanceConstraint(min_angle=20.0)
        moon_az, moon_el = coords.get_body_altaz("moon", moon_up_time)
        safe_az = (moon_az + 180.0) % 360.0
        score = c.score(patch, moon_up_time, safe_az, 50.0, coords)
        assert score == 1.0

    def test_near_moon(self, patch, coords):
        """A point at the Moon's exact position scores 0.0."""
        moon_time = Time("2026-06-15T18:00:00", scale="utc")
        c = MoonAvoidanceConstraint(min_angle=20.0)
        moon_az, moon_el = coords.get_body_altaz("moon", moon_time)
        # Point the target AT the Moon: separation = 0 < min_angle. Like the Sun
        # check, the score ignores horizon, so this is deterministic.
        score = c.score(patch, moon_time, moon_az, moon_el, coords)
        assert score == 0.0

    def test_negative_angle(self):
        with pytest.raises(ValueError, match="non-negative"):
            MoonAvoidanceConstraint(min_angle=-1.0)


class TestMinDurationConstraint:
    """Tests for MinDurationConstraint."""

    def test_sufficient_duration(self, patch, time, coords):
        """A source observable for >= min_duration scores 1.0.

        The previous ``if el > 30`` guard made this vacuous when the source was
        low (it is el ~ 23 deg at the fixture time), so the score is computed
        unconditionally now. The rejecting (setting-source) branch is covered by
        ``test_setting_source_below_min_duration_scores_zero``.
        """
        c = MinDurationConstraint(min_duration=60.0)
        az, el = coords.radec_to_altaz(180.0, -30.0, time)
        score = c.score(patch, time, az, el, coords)
        assert score == 1.0

    def test_setting_source_below_min_duration_scores_zero(self, patch, coords):
        """A source that sets within ``min_duration`` scores 0.0 (reject branch).

        At 2026-06-15T04:00 UTC the patch (180, -30) is up (el ~ 23 deg, above the
        20 deg limit) but sets within two hours, so a 2-hour min_duration pushes
        the forward elevation check below the limit.
        """
        time = Time("2026-06-15T04:00:00", scale="utc")
        c = MinDurationConstraint(min_duration=7200.0)
        az, el = coords.radec_to_altaz(180.0, -30.0, time)
        assert el > 20.0  # currently observable...
        assert c.score(patch, time, az, el, coords) == 0.0  # ...but sets too soon

    def test_negative_duration(self):
        with pytest.raises(ValueError, match="non-negative"):
            MinDurationConstraint(min_duration=-1.0)


class TestObservingPatchValidation:
    """ObservingPatch.__post_init__ rejects invalid fields."""

    @staticmethod
    def _kwargs(**overrides):
        base = dict(
            name="p",
            ra_center=180.0,
            dec_center=-30.0,
            width=4.0,
            height=4.0,
            scan_type="pong",
            velocity=0.5,
        )
        base.update(overrides)
        return base

    def test_negative_height_raises(self):
        with pytest.raises(ValueError, match="height must be positive"):
            ObservingPatch(**self._kwargs(height=-1.0))

    def test_non_positive_priority_raises(self):
        with pytest.raises(ValueError, match="priority must be positive"):
            ObservingPatch(**self._kwargs(priority=0.0))

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="weight must be non-negative"):
            ObservingPatch(**self._kwargs(weight=-0.5))
