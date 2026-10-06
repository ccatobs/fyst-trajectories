"""Value/outcome assertions for specific documentation examples.

This module holds the value-bearing checks for selected documentation
examples, the invariants, error/warning behaviours, and regression
guards that go beyond "this snippet runs". Pure execution coverage for
every code block in ``docs/*.rst`` lives in
``tests/test_doc_examples_rst.py``, which extracts and runs each block, so
nothing here is a run-only inline copy of a documented snippet.
"""

import warnings

import numpy as np
import pytest
from astropy.time import Time

from fyst_trajectories import (
    Coordinates,
    InstrumentOffset,
    get_fyst_site,
    normalize_frame,
)
from fyst_trajectories.offsets import (
    boresight_to_detector,
    compute_focal_plane_rotation,
    detector_to_boresight,
)

# NOTE: Function-level imports below mirror what the corresponding docs snippet shows
# the user. Do not hoist them to module level.

# ============================================================================
# quickstart.rst examples
# ============================================================================


def test_quickstart_get_site():
    """Test basic site retrieval from quickstart.rst."""
    site = get_fyst_site()
    print(f"FYST is at {site.latitude}, {site.longitude}")
    # FYST on Cerro Chajnantor: lat -22.9856, lon -67.7403 (site.py constants).
    assert site.latitude == pytest.approx(-22.9856, abs=1e-3)
    assert site.longitude == pytest.approx(-67.7403, abs=1e-3)


def test_quickstart_radec_to_altaz():
    """Test RA/Dec to Az/El conversion from quickstart.rst."""
    from fyst_trajectories import get_fyst_site

    site = get_fyst_site()
    coords = Coordinates(site)

    # Orion Nebula
    obstime = Time("2026-01-15T02:00:00", scale="utc")
    az, el = coords.radec_to_altaz(ra=83.82, dec=-5.39, obstime=obstime)
    print(f"Orion is at Az={az:.1f}, El={el:.1f}")
    assert isinstance(az, float)
    assert isinstance(el, float)
    # Round-trips back to the input RA/Dec (a real transform, not a stub).
    ra_back, dec_back = coords.altaz_to_radec(az, el, obstime=obstime)
    assert ra_back == pytest.approx(83.82, abs=0.01)
    assert dec_back == pytest.approx(-5.39, abs=0.01)


def test_quickstart_proper_motion():
    """Test proper motion support from quickstart.rst."""
    from fyst_trajectories import get_fyst_site

    coords = Coordinates(get_fyst_site())

    # Barnard's Star, J2000 catalogue position and proper motion
    az, el = coords.radec_to_altaz_with_pm(
        ra=269.452,
        dec=4.693,
        pm_ra=-798.58,
        pm_dec=10328.12,  # mas/yr
        ref_epoch=Time("J2000.0"),
        obstime=Time("2026-06-15T04:00:00"),
    )
    assert isinstance(az, float)
    assert isinstance(el, float)
    # 10.4"/yr proper motion over ~26.5 yr shifts the apparent position ~0.076 deg
    # from the zero-PM transform, a real correction, not a no-op.
    az0, el0 = coords.radec_to_altaz(269.452, 4.693, obstime=Time("2026-06-15T04:00:00"))
    sep = np.hypot((az - az0) * np.cos(np.radians(el)), el - el0)
    assert sep == pytest.approx(0.0761, abs=0.01)


# ============================================================================
# instrument_offsets.rst examples
# ============================================================================


def test_offsets_compute_focal_plane_rotation():
    """Test compute_focal_plane_rotation from instrument_offsets.rst.

    The +1 is the default Right-Nasmyth sign, pending FYST-team confirmation
    of the Nasmyth port ("Pending instrument verification" in
    ``docs/index.rst``); this number moves with that decision.
    """
    site = get_fyst_site()
    offset = InstrumentOffset(dx=5.0, dy=3.0, instrument_rotation=10.0)

    rotation = compute_focal_plane_rotation(
        el=45.0, site=site, offset=offset, parallactic_angle=20.0
    )
    # rotation = +1 * 45.0 + 10.0 + 20.0 = 75.0
    assert abs(rotation - 75.0) < 0.01


def test_offsets_boresight_to_detector():
    """Test boresight_to_detector from instrument_offsets.rst."""
    offset = InstrumentOffset(dx=5.0, dy=3.0)  # arcmin

    det_az, det_el = boresight_to_detector(
        az=180.0,
        el=45.0,
        offset=offset,
        focal_plane_rotation=30.0,  # degrees
    )
    assert isinstance(det_az, float)
    assert isinstance(det_el, float)
    # The detector sits offset from the boresight by the offset magnitude
    # sqrt(5^2 + 3^2) = 5.83 arcmin = 0.0972 deg on-sky.
    sep = np.hypot((det_az - 180.0) * np.cos(np.radians(45.0)), det_el - 45.0)
    assert sep == pytest.approx(0.0972, abs=0.005)


def test_offsets_detector_to_boresight():
    """Test detector_to_boresight from instrument_offsets.rst."""
    offset = InstrumentOffset(dx=5.0, dy=3.0)

    det_az, det_el = boresight_to_detector(
        az=180.0, el=45.0, offset=offset, focal_plane_rotation=30.0
    )

    bore_az, bore_el = detector_to_boresight(
        det_az=det_az, det_el=det_el, offset=offset, focal_plane_rotation=30.0
    )

    # Should get back original boresight position
    assert abs(bore_az - 180.0) < 0.001
    assert abs(bore_el - 45.0) < 0.001


# ============================================================================
# coordinate_systems.rst examples
# ============================================================================


def test_coordsys_frame_aliases():
    """Test frame alias usage from coordinate_systems.rst."""
    # Case-insensitive lookup
    astropy_frame = normalize_frame("J2000")  # Returns "icrs"
    assert astropy_frame == "icrs"
    astropy_frame = normalize_frame("b1950")  # Returns "fk4"
    assert astropy_frame == "fk4"

    # Unknown frames are lowercased for astropy compatibility
    astropy_frame = normalize_frame("MyFrame")  # Returns "myframe"
    assert astropy_frame == "myframe"


# ============================================================================
# planning.rst examples
# ============================================================================


def test_planning_field_region_cmb():
    """Test FieldRegion construction example from planning.rst."""
    from fyst_trajectories.planning import FieldRegion

    # Equatorial field: 10 deg wide x 6 deg Dec (matches the planning.rst example)
    cmb_field = FieldRegion(
        ra_center=0.0,  # deg (0h RA)
        dec_center=-2.0,  # deg
        width=10.0,  # on-sky width in degrees, not an RA span
        height=6.0,  # Dec extent in degrees
    )

    # Dec boundaries are computed automatically
    print(f"Dec range: [{cmb_field.dec_min}, {cmb_field.dec_max}]")
    assert cmb_field.dec_min == pytest.approx(-5.0)
    assert cmb_field.dec_max == pytest.approx(1.0)


def test_planning_plan_pong_scan_multiple_cycles():
    """Test multi-cycle plan_pong_scan example from planning.rst."""
    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.planning import FieldRegion, plan_pong_scan

    site = get_fyst_site()
    field = FieldRegion(ra_center=53.117, dec_center=-27.808, width=5.0, height=6.7)

    block = plan_pong_scan(
        field=field,
        velocity=0.5,
        spacing=0.08,
        num_terms=4,
        site=site,
        start_time=Time("2026-03-15T23:30:00", scale="utc"),
        timestep=0.1,
        angle=170.0,  # rotation angle (degrees)
        n_cycles=3,  # observe 3 full Pong periods
    )

    assert block.trajectory.n_points > 0
    assert block.computed_params["n_cycles"] == 3
    # Duration should equal 3 periods.
    assert block.duration == pytest.approx(block.computed_params["period"] * 3)


def test_planning_plan_source_ces():
    """Test 'Source CES' worked example from planning.rst."""
    from astropy.time import Time

    from fyst_trajectories import PRIMECAM_MODULES, get_fyst_site
    from fyst_trajectories.planning import plan_source_ces

    site = get_fyst_site()
    modules = [PRIMECAM_MODULES[k] for k in ("c", "i1", "i2", "i3", "i4", "i5", "i6")]

    block = plan_source_ces(
        body="jupiter",
        footprint=modules,
        el_bore=35.0,
        night=Time("2026-03-15T00:00:00", scale="utc"),
        mode="rising",
        site=site,
    )

    print(block.summary)
    cp = block.computed_params
    print(f"Source pass: {cp['t0_iso'][:19]} to {cp['t1_iso'][:19]}")
    print(f"Az drift:    {cp['v_az']:+.5f} deg/s")
    print(f"Az range:    [{cp['az_start']:.2f}, {cp['az_start'] + cp['az_throw']:.2f}] deg")

    assert cp["mode"] == "rising"
    assert cp["el_bore"] == pytest.approx(35.0)
    assert cp["duration"] > 0
    assert cp["n_scans"] >= 1


# ============================================================================
# Source docstring regression tests
# ============================================================================
# Source docstring regression test: a rise/set call that returns no set time.


def test_get_rise_set_times_handles_no_set_within_window():
    """Assert the outcome of the ``get_rise_set_times`` docstring example.

    Some sources rise within the search window but do not set within it, which
    is why the docstring example guards ``set_`` before dereferencing
    ``set_.iso``.  These inputs are such a case: the source rises about
    17:13 UTC and sets past the 24 h window, with roughly four hours of
    margin either way.  Asserting the outcome, rather than repeating the
    docstring's ``None`` guard, is what makes this falsifiable: a stub
    returning ``(None, None)`` must fail it.
    """
    from astropy.time import Time, TimeDelta

    from fyst_trajectories import Coordinates, get_fyst_site

    coords = Coordinates(get_fyst_site())
    start = Time("2026-03-15T00:00:00", scale="utc")
    rise, set_ = coords.get_rise_set_times(
        ra=83.633,
        dec=22.014,  # Crab Nebula / Orion neighborhood
        start_time=start,
        horizon=0.0,
        max_search_hours=24.0,
        step_hours=0.1,
    )
    assert set_ is None  # the case the docstring guard exists for
    assert rise is not None
    assert isinstance(rise.iso, str)
    assert start <= rise <= start + TimeDelta(24.0 * 3600.0, format="sec")


# ============================================================================
# Vacuum-by-default coordinates (no_refraction)
# ============================================================================


def test_no_refraction_atmosphere_pattern():
    """Test that ``Coordinates(site)`` produces vacuum coordinates without warning.

    Bare ``Coordinates(site)`` defaults to vacuum (no refraction) because
    refraction is applied downstream at execution time, by exactly one of
    the Go TCS or the ACU. No warning is emitted.
    ``AtmosphericConditions.no_refraction()`` is available as an explicit
    opt-in synonym for the same behaviour.
    """
    from fyst_trajectories import AtmosphericConditions, Coordinates, get_fyst_site

    site = get_fyst_site()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        coords_bare = Coordinates(site)  # vacuum, and no warning
    assert coords_bare.atmosphere.pressure_hpa == 0

    coords_explicit = Coordinates(site, atmosphere=AtmosphericConditions.no_refraction())
    obstime = Time("2026-01-15T02:00:00", scale="utc")
    bare = coords_bare.radec_to_altaz(83.633, 22.014, obstime=obstime)
    explicit = coords_explicit.radec_to_altaz(83.633, 22.014, obstime=obstime)
    assert bare == explicit  # identical result
