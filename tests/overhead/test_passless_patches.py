"""A constant-elevation patch that can never have a pass is refused when the schedule is built.

``generate_timeline`` refuses, before it schedules anything, a
constant-elevation patch for which the planner's own solve finds no pass
from any instant of one sidereal day: a field whose RA edges cover the
whole circle at its central declination, a field that reaches a celestial
pole, and a scan elevation the planner finds no crossing of on the side the
patch asks for (either side when it names none), or, for an
``lsa_window`` patch, none in its window. Each message names the patch and
the reason. A patch the planner plans on some day is accepted, and its
verdict equals the planner's on a grid of fields around the culminations.
"""

import dataclasses
import functools
import warnings

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time, TimeDelta

from fyst_trajectories import Coordinates, get_fyst_site, get_primecam_offset
from fyst_trajectories.exceptions import PointingError, PointingWarning
from fyst_trajectories.overhead import ObservingPatch, generate_timeline
from fyst_trajectories.overhead.scheduler.helpers import _ce_pass_refusal
from fyst_trajectories.planning import FieldRegion, plan_constant_el_scan
from fyst_trajectories.planning._ce_geometry import _field_ra_edges

SITE = get_fyst_site()

#: A night around the upper transit of the Dec -80 field below (its RA is
#: 45 deg, 3 h, past the start's local sidereal time).
_START = Time("2026-06-15T03:00:00", scale="utc")
_END = Time("2026-06-15T07:00:00", scale="utc")


def _patch(name, ra, dec, width, height, elevation):
    return ObservingPatch(
        name=name,
        ra_center=ra,
        dec_center=dec,
        width=width,
        height=height,
        scan_type="constant_el",
        velocity=0.5,
        elevation=elevation,
    )


def _schedule(patch):
    return generate_timeline(patches=[patch], site=SITE, start_time=_START, end_time=_END)


@functools.cache
def _centre_elevations(ra, dec):
    """Return the centre's elevation over one sidereal day from the start, every second."""
    coords = Coordinates(SITE)
    grid = _START + np.arange(0.0, 86164.0905, 1.0) * u.s
    _, el = coords.radec_to_altaz(ra, dec, grid)
    return np.asarray(el)


def test_a_field_covering_every_ra_is_refused():
    """A 400 deg wide band at Dec -30 has no leading and trailing RA edges."""
    band = _patch("Band", 0.0, -30.0, 400.0, 10.0, 50.0)
    with pytest.raises(ValueError, match=r"Band: .*whole turn") as err:
        _schedule(band)
    assert "RA" in str(err.value)


def test_a_field_reaching_a_pole_is_refused():
    pole = _patch("Pole", 0.0, -89.8, 2.0, 2.0, 25.0)
    with pytest.raises(ValueError, match=r"Pole: .*pole"):
        _schedule(pole)


#: A small field at Dec -80, circumpolar here: its centre culminates near
#: el 33 and passes below the pole near el 13.
_RA = float(_START.sidereal_time("mean", longitude=SITE.longitude * u.deg).deg) + 45.0
_DEC = -80.0


def _culmination():
    return float(_centre_elevations(_RA, _DEC).max())


_NO_PASS = "the planner finds no pass at the scan elevation"


def test_a_scan_elevation_above_the_culmination_is_refused():
    """The centre culminates 0.05 deg below the scan elevation: no pass, refused by name."""
    elevation = _culmination() + 0.05
    patch = _patch("High", _RA, _DEC, 2.0, 2.0, elevation)
    with pytest.raises(ValueError, match=rf"High: {_NO_PASS} .*Dec -80\.00") as err:
        _schedule(patch)
    assert f"{elevation:.2f}" in str(err.value)
    assert "on the rising or the setting side over a sidereal day" in str(err.value)


def test_a_scan_elevation_just_below_the_culmination_is_scanned():
    """The centre culminates 0.05 deg above the scan elevation: accepted and booked."""
    patch = _patch("Low", _RA, _DEC, 2.0, 2.0, _culmination() - 0.05)
    timeline = _schedule(patch)
    assert sum(b.duration for b in timeline.science_blocks) > 0.0


def test_a_scan_elevation_the_planner_cannot_time_is_refused():
    """A field whose edges barely reach the elevation, and are never seen to, is refused.

    The scan elevation sits 1e-5 deg below the centre's highest elevation
    (the RA edges, at the centre's Dec, culminate there too): the field
    reaches it for a moment the planner's crossing search does not find,
    so the planner plans no pass from any anchor of the day, on either
    side, and the patch is refused rather than selected for nothing.
    """
    elevation = _culmination() - 1e-5
    patch = _patch("Grazing", _RA, _DEC, 2.0, 2.0, elevation)
    assert not _planner_plans(patch, [{"rising": True}, {"rising": False}])
    with pytest.raises(ValueError, match=rf"Grazing: {_NO_PASS}"):
        generate_timeline(
            patches=[patch], site=SITE, start_time=_START, end_time=_START + 600.0 * u.s
        )


def test_a_scan_elevation_below_the_lower_transit_is_refused():
    """A field near the pole stays above el 20.3 all day; the scan elevation is refused."""
    lowest = float(_centre_elevations(0.0, -87.5).min())
    assert 20.3 < lowest
    patch = _patch("Under", 0.0, -87.5, 2.0, 2.0, 20.3)
    with pytest.raises(ValueError, match=rf"Under: {_NO_PASS}"):
        _schedule(patch)


def test_a_pass_longer_than_the_planners_horizon_is_refused():
    """A field 231 deg of RA wide crosses el 50 in a pass longer than 12 h.

    Its edges both cross the elevation within a day, but the planner
    searches 12 h ahead and never finds the pass whole, so the patch is
    refused.
    """
    patch = _patch("Wide", 0.0, -30.0, 200.0, 10.0, 50.0)
    assert not _planner_plans(patch, [{"rising": True}, {"rising": False}])
    with pytest.raises(ValueError, match=rf"Wide: {_NO_PASS} .*within 12\.0 hours"):
        _schedule(patch)


def test_a_patch_asking_for_a_side_is_refused_on_that_side_alone():
    """The message names the side the patch asks for."""
    patch = dataclasses.replace(
        _patch("High", _RA, _DEC, 2.0, 2.0, _culmination() + 0.05),
        scan_params={"rising": False},
    )
    with pytest.raises(ValueError, match=rf"High: {_NO_PASS} .* on the setting side "):
        _schedule(patch)


def test_the_refusals_come_in_order_and_name_each_patch():
    """The RA and pole checks come before the crossing; every refused patch is named."""
    band = _patch("Band", 0.0, -30.0, 400.0, 10.0, 89.0)
    high = dataclasses.replace(band, name="High", width=2.0, height=2.0)
    with pytest.raises(ValueError, match="whole turn") as err:
        _schedule(band)
    assert _NO_PASS not in str(err.value)
    with pytest.raises(ValueError, match=f"High: {_NO_PASS}"):
        generate_timeline(patches=[high], site=SITE, start_time=_START, end_time=_END)


# --- Patches the planner plans are booked ---------------------------------

#: A night over which the sidereal windows below open (2026-06-15 08:16).
_LSA_START = Time("2026-06-15T06:00:00", scale="utc")
_LSA_END = Time("2026-06-15T11:00:00", scale="utc")


@pytest.mark.parametrize(
    ("name", "ra", "dec", "width", "height", "elevation", "science"),
    [
        # The centre never reaches el 66 (it culminates near 63.1); the
        # field's northern border does, on the window's side.
        ("tall_north", 330.0, -50.0, 30.0, 20.0, 66.0, 4474.0),
        # The field spans a whole turn of RA, which a sidereal window's solve never asks.
        ("band", 180.0, -30.0, 360.0, 10.0, 50.0, 4501.0),
        # The centre never reaches el 80 (it culminates near 76.9).
        ("tall_south", 330.0, -10.0, 30.0, 20.0, 80.0, 4462.4),
    ],
)
def test_an_lsa_window_patch_the_planner_plans_is_booked(
    name, ra, dec, width, height, elevation, science
):
    """A sidereal-window patch is judged by its window's own solve, not by the field form.

    The window (320, 340) opens at 08:16:25.419; the night books its pass.
    """
    patch = dataclasses.replace(
        _patch(name, ra, dec, width, height, elevation),
        scan_params={"lsa_window": (320.0, 340.0)},
    )
    timeline = generate_timeline(
        patches=[patch], site=SITE, start_time=_LSA_START, end_time=_LSA_END
    )
    blocks = timeline.science_blocks
    assert blocks[0].t_start.isot == "2026-06-15T08:16:25.419"
    assert sum(b.duration for b in blocks) == pytest.approx(science, abs=0.05)


def test_a_field_whose_edges_culminate_above_its_centre_is_booked():
    """The planner times the pass by the RA edges, whose apparent Dec is not the centre's.

    A 60 x 2 deg field at RA 0, Dec -10: its centre culminates at about
    76.866 deg and its RA edges at about 76.886 and 76.887 deg, so at
    el 76.8756 the planner plans a pass the centre never reaches.
    """
    patch = _patch("Tie", 0.0, -10.0, 60.0, 2.0, 76.8756)
    timeline = generate_timeline(
        patches=[patch],
        site=SITE,
        start_time=Time("2026-06-15T08:00:00", scale="utc"),
        end_time=Time("2026-06-15T13:30:00", scale="utc"),
    )
    blocks = timeline.science_blocks
    assert len(blocks) == 5
    assert blocks[0].t_start.isot == "2026-06-15T08:53:37.032"
    assert sum(b.duration for b in blocks) == pytest.approx(13401.6, abs=0.05)


# --- The refusal agrees with the planner ----------------------------------

#: Anchors of the planner's search, every three hours through a sidereal day:
#: a pass shorter than 21 h is found from one of them, as the planner searches
#: 12 h ahead from an instant outside a pass.
_ANCHORS = [_START + TimeDelta(3600.0 * k, format="sec") for k in range(0, 25, 3)]


def _planner_plans(patch, calls):
    """Whether the planner plans the patch from some anchor with one of ``calls``' options."""
    field = FieldRegion(
        ra_center=patch.ra_center,
        dec_center=patch.dec_center,
        width=patch.width,
        height=patch.height,
    )
    for anchor in _ANCHORS:
        for options in calls:
            try:
                with warnings.catch_warnings():
                    # Scans near the culmination run high; the verdict, not the advisory, is asked.
                    warnings.filterwarnings(
                        "ignore", message="High elevation", category=PointingWarning
                    )
                    plan_constant_el_scan(
                        field,
                        elevation=patch.elevation,
                        velocity=1.0,
                        site=SITE,
                        start_time=anchor,
                        timestep=1.0,
                        detector_offset=patch.detector_offset,
                        **options,
                    )
            except PointingError:
                continue
            return True
    return False


@functools.cache
def _culmination_of(ra, dec):
    """Return the highest elevation of ``(ra, dec)`` over a sidereal day from the start."""
    el = _centre_elevations(ra, dec)
    i = int(np.argmax(el))
    coords = Coordinates(SITE)
    fine = _START + (i + np.arange(-2.0, 2.0, 0.05)) * u.s
    _, el_fine = coords.radec_to_altaz(ra, dec, fine)
    return float(np.max(el_fine))


def _variants(ra, dec, width, height, ref):
    """Return the options of each patch of one field put to the planner, and the field's patch."""
    base = ObservingPatch(
        name="Grid",
        ra_center=ra,
        dec_center=dec,
        width=width,
        height=height,
        scan_type="constant_el",
        velocity=1.0,
        elevation=50.0,
    )
    if ref == "border":
        return {
            "lsa_rising": (
                {"lsa_window": ((ra - 25.0) % 360.0, (ra - 5.0) % 360.0)},
                None,
            ),
            "lsa_setting": (
                {"lsa_window": ((ra + 5.0) % 360.0, (ra + 25.0) % 360.0)},
                None,
            ),
        }, base
    return {
        "either": ({}, None),
        "setting": ({"rising": False}, None),
        "az_window": ({"az_min": 100.0, "az_max": 120.0}, None),
        "offset": ({}, get_primecam_offset("IM3")),
    }, base


def _grid_cases():
    fields = [(0.0, -10.0, 60.0, 2.0), (30.0, -50.0, 20.0, 6.0)]
    cases = []
    for field in fields:
        for ref in ("centre", "lead", "trail", "border"):
            for delta in (-0.002, 0.002):
                variants, _ = _variants(*field, ref)
                cases.extend((field, ref, delta, variant) for variant in variants)
    return cases


def _reference_elevation(ra, dec, width, height, ref):
    lead, trail = _field_ra_edges(
        FieldRegion(ra_center=ra, dec_center=dec, width=width, height=height), 0.0
    )
    if ref == "centre":
        return _culmination_of(ra, dec)
    if ref in ("lead", "trail"):
        return _culmination_of((lead if ref == "lead" else trail) % 360.0, dec)
    near = min((dec - height / 2.0, dec + height / 2.0), key=lambda d: abs(d - SITE.latitude))
    return _culmination_of(ra, near)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("field", "ref", "delta", "variant"),
    _grid_cases(),
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_the_refusal_agrees_with_the_planner_around_each_culmination(field, ref, delta, variant):
    """Refused exactly when the planner plans no pass from any anchor of the day.

    Elevations 0.002 deg either side of the culmination of the field centre,
    of each RA edge at the central Dec (where the planner times the field
    form) and of the border point nearest the zenith (where the window
    form's range is found), for a patch on either side, one asking for the
    setting side, one with an azimuth window on the rising side, one
    observed with a ring bay, and an ``lsa_window`` on each side.
    """
    variants, base = _variants(*field, ref)
    scan_params, offset = variants[variant]
    patch = dataclasses.replace(
        base,
        elevation=_reference_elevation(*field, ref) + delta,
        scan_params=scan_params,
        detector_offset=offset,
    )
    if "lsa_window" in scan_params:
        calls = [{"lsa_window": scan_params["lsa_window"]}]
    elif variant == "setting":
        calls = [{"rising": False}]
    elif variant == "az_window":
        calls = [{"rising": True}]
    else:
        calls = [{"rising": True}, {"rising": False}]
    refused = _ce_pass_refusal(patch, Coordinates(SITE), _START) is not None
    assert refused == (not _planner_plans(patch, calls))
