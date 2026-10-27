"""Azimuth and Sun regression tests for the overhead scheduler.

The overhead subpackage is an OFFLINE observing-night simulator. These
tests pin:

- Cable-wrap azimuth normalization. Raw astropy azimuths must be
  placed in the telescope's cable-wrap window, relative to the current
  pose, before any slew or boresight math: a north-straddling pair
  (350 deg and 10 deg) must give the short-path slew (~20 deg, not
  ~340 deg), and a slew block's boresight must sit at the arithmetic
  mid-travel azimuth of the coherent from/to pair.
- Sun-aware pong/daisy duration clip. A pong/daisy scan whose
  commanded pose is inside the Sun exclusion zone, or enters it
  mid-scan (a pinned elevation or an injected Sun model), must have
  its duration trimmed, mirroring the constant_el branch.
- Both Sun checks evaluate a pinned ``patch.elevation`` when the
  patch declares one, not the field's own drifting elevation.
- A science subscan is recorded in the cable wrap the slew placed it on,
  so the next slew is priced and Sun-checked from the pose the mount holds.
"""

import pytest
from astropy.time import Time, TimeDelta

from fyst_trajectories import Coordinates, get_fyst_site
from fyst_trajectories.dispatch import estimate_slew_time
from fyst_trajectories.overhead import (
    CalibrationPolicy,
    generate_timeline,
    schedule_to_trajectories,
)
from fyst_trajectories.overhead.constraints import MinDurationConstraint
from fyst_trajectories.overhead.models import BlockType, ObservingPatch, OverheadModel
from fyst_trajectories.overhead.scheduler.helpers import (
    _compute_scan_duration,
    _time_inside_el_limits,
    _time_until_sun_unsafe,
)
from fyst_trajectories.overhead.utils import _normalize_az

# A time/place where a target offset ~20 deg in RA from the Sun is well
# above the FYST horizon but inside the 45 deg Sun exclusion radius.
_SUN_TIME = Time("2026-06-15T17:30:00", scale="utc")


@pytest.fixture
def sun_radec(coordinates):
    """Sun RA/Dec at the test time (for placing sunward targets)."""
    ra, dec = coordinates.get_body_radec("sun", _SUN_TIME)
    return float(ra), float(dec)


class TestNorthStraddleSlew:
    """Cable-wrap normalization drives slew distance."""

    def test_short_path_slew_time(self, site):
        """From a pose at az 10, raw azimuth 350 is a 20 deg move, not 340 deg.

        The scheduler places the target on the in-limits representative
        nearest the current pose (``ref=``), so across north it picks -10.
        The slew-time pin follows the site's azimuth velocity and
        acceleration limits, which await operations-team ratification, and
        moves with them.
        """
        pose = 10.0
        target = _normalize_az(350.0, site, ref=pose)
        assert target == pytest.approx(-10.0, abs=1e-6)

        t_norm = estimate_slew_time(pose, 50.0, target, 50.0, site=site)
        # 20 deg az slew (FYST az vel=3, accel=1.5): d_accel=6 < 20, so
        # t = 2*t_accel + (20 - 6)/3 = 4 + 4.667 = 8.667 s.
        assert t_norm == pytest.approx(8.667, abs=0.05)

        # The un-normalized direct path would be abs(10 - 350) = 340 deg,
        # an order of magnitude longer.
        t_raw = estimate_slew_time(pose, 50.0, 350.0, 50.0, site=site)
        assert t_raw > 100.0
        assert t_norm < t_raw / 5.0


class TestSunAwarePongDuration:
    """Pong/daisy duration is clipped to the sun-safe sub-window."""

    def test_time_until_sun_unsafe_immediate(self, coordinates, sun_radec, site):
        """A target already inside the exclusion radius is unsafe from t=0."""
        sun_ra, sun_dec = sun_radec
        # ~20 deg in RA from the Sun -> ~18 deg separation, inside 45 deg.
        ra = (sun_ra + 20.0) % 360
        dur = _time_until_sun_unsafe(
            ra, sun_dec, _SUN_TIME, 3600.0, coordinates, site.sun_avoidance.exclusion_radius
        )
        assert dur == pytest.approx(0.0, abs=1.0)

    def test_time_until_sun_unsafe_far_target(self, coordinates, sun_radec, site):
        """A target far from the Sun stays safe for the whole window."""
        sun_ra, sun_dec = sun_radec
        ra = (sun_ra + 120.0) % 360
        dur = _time_until_sun_unsafe(
            ra, sun_dec, _SUN_TIME, 3600.0, coordinates, site.sun_avoidance.exclusion_radius
        )
        assert dur == pytest.approx(3600.0, abs=1.0)

    def test_pong_duration_clipped_by_sun(self, coordinates, sun_radec, site):
        """A sunward pong scan is clipped though it stays above the el floor.

        The target is inside the Sun exclusion radius but well above the
        elevation limit, so an elevation-only check returns the full
        max_scan_duration, while the sun-aware clip trims it to 0.
        """
        sun_ra, sun_dec = sun_radec
        ra = (sun_ra + 20.0) % 360
        end_time = _SUN_TIME + TimeDelta(8 * 3600, format="sec")
        overhead = OverheadModel()
        patch = ObservingPatch(
            name="sunward",
            ra_center=ra,
            dec_center=sun_dec,
            width=2.0,
            height=2.0,
            scan_type="pong",
            velocity=0.5,
        )

        # Elevation-only window is the full max_scan_duration (target is up).
        el_only = _time_inside_el_limits(patch, _SUN_TIME, 3600.0, coordinates)
        assert el_only == pytest.approx(3600.0, abs=1.0)

        # But the sun-aware duration is clipped to zero.
        dur = _compute_scan_duration(patch, _SUN_TIME, end_time, site, coordinates, overhead, 43.0)
        assert dur == 0.0


class TestPinnedElevationIsClippedAtThePinnedPose:
    """The Sun clip evaluates the pose the visit commands, not the field's.

    A patch that pins ``elevation`` is scored at that elevation, slewed to
    it and recorded at it on every emitted block, so the clip that decides
    how long the visit may run has to ask about the same pose. Asking
    about the field's own drifting elevation instead would book a visit
    for the whole window while the pose it records sits inside the zone
    for most of it.
    """

    def _model(self, t0):
        """Build a model that turns unsafe above 45 deg after 600 s."""

        def sun_safe(az, el, t):
            if (t - t0).sec < 600.0:
                return True
            return float(el) < 45.0

        return sun_safe

    def test_clip_follows_the_pinned_elevation(self, coordinates, site):
        t0 = Time("2026-06-15T08:00:00", scale="utc")
        end_time = t0 + TimeDelta(8 * 3600, format="sec")
        overhead = OverheadModel()
        # The test field (RA 24, dec -32): up and well away from the Sun on this night, and
        # rising from 30 to 42 deg over the hour, so its own elevation
        # never reaches the 45 deg the model closes at.
        kwargs = dict(
            name="pinned",
            ra_center=24.0,
            dec_center=-32.0,
            width=10.0,
            height=10.0,
            scan_type="pong",
            velocity=1.0,
        )
        _, true_el = coordinates.radec_to_altaz(24.0, -32.0, t0)
        assert float(true_el) < 45.0  # the two elevations disagree here

        pinned = ObservingPatch(**kwargs, elevation=50.0)
        free = ObservingPatch(**kwargs)
        model = self._model(t0)

        pinned_dur = _compute_scan_duration(
            pinned, t0, end_time, site, coordinates, overhead, float(true_el), sun_safe=model
        )
        free_dur = _compute_scan_duration(
            free, t0, end_time, site, coordinates, overhead, float(true_el), sun_safe=model
        )

        # The pinned pose sits at 50 deg, which the model closes at 600 s.
        assert pinned_dur == pytest.approx(600.0, abs=30.0)
        # The field's own track never reaches 45 deg here, so an unpinned
        # patch is not clipped at all: the difference is the pose asked about.
        assert free_dur == pytest.approx(overhead.max_scan_duration, abs=1.0)

    def test_min_duration_forward_check_uses_the_pinned_elevation(self, coordinates):
        t0 = Time("2026-06-15T08:00:00", scale="utc")
        seen: list[tuple[float, Time]] = []

        def sun_safe(az, el, t):
            seen.append((float(el), t))
            return True

        patch = ObservingPatch(
            name="pinned",
            ra_center=24.0,
            dec_center=-32.0,
            width=10.0,
            height=10.0,
            scan_type="pong",
            velocity=1.0,
            elevation=50.0,
        )
        az, el = coordinates.radec_to_altaz(24.0, -32.0, t0)
        constraint = MinDurationConstraint(min_duration=600.0, sun_safe=sun_safe)

        assert constraint.score(patch, t0, float(az), float(el), coordinates) == 1.0
        assert [pose_el for pose_el, _ in seen] == [50.0]
        assert (seen[0][1] - t0).sec == pytest.approx(600.0, abs=1e-6)


class TestScienceIsRecordedInTheSlewsWrap:
    """A subscan's azimuths are the wrap the slew drove the mount to.

    D6 sits at azimuth 270-280 on 2026-03-01 near 11:15 UTC, so it has two
    in-limits images, near -87 and near 273. The slew reaches it from 180
    on the 273 branch while the planner solves the daisy on the -87 branch.
    Recording the planner's branch put the mount 360 deg from where the
    slew left it, and the next slew to NE (azimuth 38) was priced and
    Sun-checked from -86.5, a short path that is clear, rather than from
    273.5, whose only path to 38 crosses the Sun (azimuth 91, elevation 16).
    """

    @pytest.fixture(scope="class")
    def night(self):
        site = get_fyst_site()
        coordinates = Coordinates(site)
        ra_ne, dec_ne = coordinates.altaz_to_radec(
            40.0, 45.0, Time("2026-03-01T11:35:00", scale="utc")
        )
        daisy = dict(
            width=1.0, height=1.0, scan_type="daisy", velocity=0.3, scan_params={"radius": 0.3}
        )
        patches = [
            ObservingPatch(name="D6", ra_center=195.0, dec_center=-5.0, priority=1.0, **daisy),
            ObservingPatch(
                name="NE", ra_center=float(ra_ne), dec_center=float(dec_ne), weight=0.01, **daisy
            ),
        ]
        return generate_timeline(
            patches,
            site,
            "2026-03-01T10:50:00",
            "2026-03-01T12:30:00",
            calibration_policy=CalibrationPolicy(retune_cadence=0.0),
        )

    def test_the_subscan_holds_the_pose_the_slew_reached(self, night):
        blocks = night.blocks
        i_slew = next(i for i, b in enumerate(blocks) if b.patch_name == "slew_to_D6")
        slew = blocks[i_slew]
        science = next(b for b in blocks[i_slew:] if b.block_type == BlockType.SCIENCE)
        assert science.patch_name == "D6"
        assert slew.az_end > 180.0
        assert science.az_start <= slew.az_end <= science.az_end
        assert science.az_final == pytest.approx(273.50, abs=0.01)

    def test_the_sun_crossing_slew_from_the_recorded_pose_is_refused(self, night):
        assert not any(b.patch_name == "slew_to_NE" for b in night.blocks)
        refused = [
            b
            for b in night.blocks
            if b.block_type == BlockType.IDLE and b.metadata.get("reason") == "sun_path"
        ]
        assert refused
        assert all(b.az_start == pytest.approx(273.50, abs=0.01) for b in refused)

    def test_a_rebuild_returns_the_planners_wrap(self, night):
        (block, scan) = next(
            (b, s) for b, s in schedule_to_trajectories(night) if b.patch_name == "D6"
        )
        az = scan.trajectory.az
        assert block.az_start - float(az.min()) == pytest.approx(360.0, abs=1e-9)
        assert block.az_end - float(az.max()) == pytest.approx(360.0, abs=1e-9)
