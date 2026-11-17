"""A boresight solved outside the sky is a pose the telescope cannot take now.

For a ring bay next to the zenith the focal-plane inverse can return a
boresight elevation above 90 deg (``IM1``, from a detector elevation near
88.22 deg). Every place the offline scheduler takes a pose from the solve
treats such a pose, or one that is not finite, as it treats a failed solve:
the patch is passed over or the visit refused, and a night that meets one
runs.
"""

import numpy as np
import pytest
from astropy.time import Time, TimeDelta

from fyst_trajectories import Coordinates, get_fyst_site, get_primecam_offset
from fyst_trajectories.exceptions import OffsetInversionError
from fyst_trajectories.offsets import _solve_boresight
from fyst_trajectories.overhead import MinDurationConstraint, ObservingPatch, generate_timeline
from fyst_trajectories.overhead.constraints import _boresight_pose
from fyst_trajectories.overhead.scheduler.helpers import (
    _boresight_elevations,
    _boresight_track,
    _ce_scan_pose,
    _time_until_sun_unsafe,
)
from fyst_trajectories.overhead.scheduler.state import SchedulerContext

SITE = get_fyst_site()
COORDS = Coordinates(SITE)
IM1 = get_primecam_offset("IM1")


def _detector_el_for(boresight_el):
    """Return the detector elevation whose ``IM1`` boresight solves to ``boresight_el``."""
    lo, hi = 85.0, 89.9
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(_solve_boresight(0.0, mid, IM1, site=SITE)[1]) < boresight_el:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


#: Detector elevations whose boresight solves just under and just over the zenith.
UNDER = _detector_el_for(89.999)
OVER = _detector_el_for(90.001)


def test_the_two_poses_straddle_the_zenith():
    assert float(_solve_boresight(0.0, UNDER, IM1, site=SITE)[1]) < 90.0
    assert float(_solve_boresight(0.0, OVER, IM1, site=SITE)[1]) > 90.0


def test_boresight_pose():
    _, el = _boresight_pose(10.0, UNDER, IM1, SITE)
    assert float(el) <= 90.0
    with pytest.raises(OffsetInversionError, match="outside the sky"):
        _boresight_pose(10.0, OVER, IM1, SITE)
    with pytest.raises(OffsetInversionError) as err:
        _boresight_pose(np.array([10.0, 10.0]), np.array([UNDER, OVER]), IM1, SITE)
    assert err.value.indices == (1,)


def test_boresight_track():
    az, el = _boresight_track(np.array([10.0, 10.0]), np.array([UNDER, OVER]), IM1, SITE)
    assert np.isfinite(az[0]) and el[0] <= 90.0
    assert np.isnan(az[1]) and np.isnan(el[1])


def test_boresight_elevations():
    el = _boresight_elevations(np.array([UNDER, OVER]), IM1, SITE)
    assert el[0] <= 90.0 and np.isnan(el[1])


def _ce_patch(elevation):
    return ObservingPatch(
        name="Zen",
        ra_center=140.0,
        dec_center=-23.0,
        width=4.0,
        height=4.0,
        scan_type="constant_el",
        velocity=0.5,
        elevation=elevation,
        detector_offset=IM1,
    )


def test_ce_scan_pose():
    _, el = _ce_scan_pose(_ce_patch(UNDER), SITE)
    assert el <= 90.0
    with pytest.raises(OffsetInversionError):
        _ce_scan_pose(_ce_patch(OVER), SITE)


def test_a_constant_elevation_patch_whose_boresight_leaves_the_sky_is_refused():
    with pytest.raises(ValueError, match="no boresight places the detector there"):
        SchedulerContext.build(
            [_ce_patch(OVER)],
            SITE,
            Time("2026-03-01T01:30:00", scale="utc"),
            Time("2026-03-01T03:30:00", scale="utc"),
        )


def test_min_duration_constraint():
    """A pose outside the sky scores 0, one just inside is scored."""
    constraint = MinDurationConstraint(min_duration=60.0)
    at = Time("2026-03-01T02:00:00", scale="utc")
    assert constraint.score(_ce_patch(OVER), at, 0.0, OVER, COORDS) == 0.0
    constraint.score(_ce_patch(UNDER), at, 0.0, UNDER, COORDS)


#: The field at RA 140, Dec -23 transits next to the zenith in the hour after
#: this instant, so ``IM1``'s detector crosses both poses.
_NEAR_ZENITH = Time("2026-03-01T02:30:00", scale="utc")


@pytest.mark.parametrize("model", ["scalar", "injected"])
def test_the_sun_clip_ends_where_the_boresight_leaves_the_sky(model):
    """The window stops before the first sample whose boresight is outside the sky."""
    sun_safe = None if model == "scalar" else (lambda az, el, t: np.ones(np.shape(az), bool))
    window = _time_until_sun_unsafe(
        140.0,
        -23.0,
        _NEAR_ZENITH,
        3600.0,
        COORDS,
        SITE.sun_avoidance.exclusion_radius,
        sun_safe=sun_safe,
        reach=2.0,
        detector_offset=IM1,
    )
    assert 0.0 < window < 3600.0
    t_end = _NEAR_ZENITH + TimeDelta(window, format="sec")
    _, el = COORDS.radec_to_altaz(140.0, -23.0, t_end)
    assert float(el) < OVER


@pytest.mark.parametrize("scan_type", ["pong", "daisy"])
def test_a_night_next_to_the_zenith_runs_for_a_ring_bay(scan_type):
    """The night runs for ``IM1`` and books no more than the on-axis bay."""

    def science(bay):
        patch = ObservingPatch(
            name="ZenP",
            ra_center=140.0,
            dec_center=-23.0,
            width=4.0,
            height=4.0,
            scan_type=scan_type,
            velocity=0.5,
            detector_offset=get_primecam_offset(bay),
        )
        timeline = generate_timeline(
            patches=[patch],
            site=SITE,
            start_time=Time("2026-03-01T01:30:00", scale="utc"),
            end_time=Time("2026-03-01T03:30:00", scale="utc"),
        )
        return sum(b.duration for b in timeline.science_blocks)

    ring = science("IM1")
    assert 0.0 < ring <= science("IM0") + 1e-6
