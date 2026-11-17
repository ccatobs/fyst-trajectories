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
  pattern is inside the Sun exclusion zone, or enters it mid-scan as the
  field moves, must have its duration trimmed, mirroring the constant_el
  branch.
- Both Sun checks evaluate the pose the visit commands: the field's own
  for a pong or daisy, the patch's ``elevation`` for a constant-elevation
  patch.
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
    _pattern_reach,
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
        # The readiness lead is read only for a constant-elevation patch.
        dur = _compute_scan_duration(
            patch, _SUN_TIME, end_time, site, coordinates, overhead, 43.0, ce_ready_lead=0.0
        )
        assert dur == 0.0


class TestTheSunChecksAskAboutThePoseTheVisitCommands:
    """The Sun checks evaluate the pose the visit commands.

    A pong or daisy runs where the field is on the sky, so the clip that
    decides how long its visit may run, and the selection's forward check,
    follow the field. A constant-elevation patch is scanned at its
    ``elevation``, so the forward check asks about that elevation.
    """

    @staticmethod
    def _model(t0, ceiling):
        """Build a model that turns unsafe at or above ``ceiling`` deg after 600 s."""

        def sun_safe(az, el, t):
            if (t - t0).sec < 600.0:
                return True
            return float(el) < ceiling

        return sun_safe

    def test_the_pong_clip_follows_the_field(self, coordinates, site):
        t0 = Time("2026-06-15T08:00:00", scale="utc")
        end_time = t0 + TimeDelta(8 * 3600, format="sec")
        overhead = OverheadModel()
        # The test field (RA 24, dec -32): up and well away from the Sun on
        # this night, and rising from 30 to 42 deg over the hour, so its
        # 2 x 2 deg pattern stays between 27 and 45 deg.
        patch = ObservingPatch(
            name="free",
            ra_center=24.0,
            dec_center=-32.0,
            width=2.0,
            height=2.0,
            scan_type="pong",
            velocity=1.0,
        )
        _, first_el = coordinates.radec_to_altaz(24.0, -32.0, t0)
        _, last_el = coordinates.radec_to_altaz(24.0, -32.0, t0 + TimeDelta(3600, format="sec"))
        reach = _pattern_reach(patch)
        assert float(first_el) - reach > 27.0
        assert float(last_el) + reach < 45.0

        def duration(model):
            # The readiness lead is read only for a constant-elevation patch.
            return _compute_scan_duration(
                patch,
                t0,
                end_time,
                site,
                coordinates,
                overhead,
                float(first_el),
                sun_safe=model,
                ce_ready_lead=0.0,
            )

        # A zone above 45 deg never meets the pattern: no clip.
        assert duration(self._model(t0, 45.0)) == pytest.approx(overhead.max_scan_duration, abs=1.0)
        # A zone above 27 deg holds the whole pattern once it opens at 600 s.
        assert duration(self._model(t0, 27.0)) == pytest.approx(600.0, abs=30.0)

    def test_min_duration_forward_check_follows_a_pong_field(self, coordinates):
        t0 = Time("2026-06-15T08:00:00", scale="utc")
        seen: list[tuple[float, Time]] = []

        def sun_safe(az, el, t):
            seen.append((float(el), t))
            return True

        patch = ObservingPatch(
            name="free",
            ra_center=24.0,
            dec_center=-32.0,
            width=10.0,
            height=10.0,
            scan_type="pong",
            velocity=1.0,
        )
        az, el = coordinates.radec_to_altaz(24.0, -32.0, t0)
        _, future_el = coordinates.radec_to_altaz(24.0, -32.0, t0 + TimeDelta(600, format="sec"))
        constraint = MinDurationConstraint(min_duration=600.0, sun_safe=sun_safe)

        assert constraint.score(patch, t0, float(az), float(el), coordinates) == 1.0
        assert [pose_el for pose_el, _ in seen] == [float(future_el)]
        assert (seen[0][1] - t0).sec == pytest.approx(600.0, abs=1e-6)

    def test_min_duration_forward_check_uses_a_constant_elevation(self, coordinates):
        t0 = Time("2026-06-15T08:00:00", scale="utc")
        seen: list[tuple[float, Time]] = []

        def sun_safe(az, el, t):
            seen.append((float(el), t))
            return True

        patch = ObservingPatch(
            name="ce",
            ra_center=24.0,
            dec_center=-32.0,
            width=10.0,
            height=10.0,
            scan_type="constant_el",
            velocity=1.0,
            elevation=50.0,
        )
        az, el = coordinates.radec_to_altaz(24.0, -32.0, t0)
        constraint = MinDurationConstraint(min_duration=600.0, sun_safe=sun_safe)

        assert constraint.score(patch, t0, float(az), float(el), coordinates) == 1.0
        assert [pose_el for pose_el, _ in seen] == [50.0]
        assert (seen[0][1] - t0).sec == pytest.approx(600.0, abs=1e-6)


@pytest.fixture(scope="module")
def night():
    """Return the D6 and NE daisy night of :class:`TestScienceIsRecordedInTheSlewsWrap`."""
    site = get_fyst_site()
    coordinates = Coordinates(site)
    ra_ne, dec_ne = coordinates.altaz_to_radec(40.0, 45.0, Time("2026-03-01T11:35:00", scale="utc"))
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


class TestScienceIsRecordedInTheSlewsWrap:
    """A subscan's azimuths are the wrap the slew drove the mount to.

    D6 sits at azimuth 270-280 on 2026-03-01 near 11:15 UTC, so it has two
    in-limits images, near -87 and near 273. The slew reaches it from 180
    on the 273 branch while the planner solves the daisy on the -87 branch.
    Recording the planner's branch put the mount 360 deg from where the
    slew left it, and the next slew to NE (azimuth 38) was priced and
    Sun-checked from -86.6, a short path that is clear, rather than from
    273.4, whose only path to 38 crosses the Sun (azimuth 91, elevation 16).
    """

    def test_the_subscan_holds_the_pose_the_slew_reached(self, night):
        blocks = night.blocks
        i_slew = next(i for i, b in enumerate(blocks) if b.patch_name == "slew_to_D6")
        slew = blocks[i_slew]
        science = next(b for b in blocks[i_slew:] if b.block_type == BlockType.SCIENCE)
        assert science.patch_name == "D6"
        assert slew.az_end > 180.0
        assert science.az_start <= slew.az_end <= science.az_end
        assert science.az_final == pytest.approx(273.44, abs=0.01)

    def test_the_sun_crossing_slew_from_the_recorded_pose_is_refused(self, night):
        assert not any(b.patch_name == "slew_to_NE" for b in night.blocks)
        refused = [
            b
            for b in night.blocks
            if b.block_type == BlockType.IDLE and b.metadata.get("reason") == "sun_path"
        ]
        assert refused
        assert all(b.az_start == pytest.approx(273.44, abs=0.01) for b in refused)

    def test_a_rebuild_returns_the_planners_wrap(self, night):
        (block, scan) = next(
            (b, s) for b, s in schedule_to_trajectories(night) if b.patch_name == "D6"
        )
        az = scan.trajectory.az
        assert block.az_start - float(az.min()) == pytest.approx(360.0, abs=1e-9)
        assert block.az_end - float(az.max()) == pytest.approx(360.0, abs=1e-9)
