"""Tests for the Prime-Cam module bays and their offsets.

Validates the bay layout (IM0 on the optical axis, IM1 .. IM6 on the inner
ring at the instrument team's angles, seen looking out at the sky with
elevation up and azimuth increasing to the right), the layout on the sky at
several boresight elevations, the name lookup and its refusals, and the
tag-resolution and scheduler-geometry helpers.
"""

import math

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time

import fyst_trajectories
from fyst_trajectories import boresight_to_detector, compute_focal_plane_rotation
from fyst_trajectories.offsets import InstrumentOffset
from fyst_trajectories.planning import plan_source_ces, resolve_footprint
from fyst_trajectories.primecam import (
    INNER_RING_RADIUS_MM,
    MODULE_FOV_RADIUS_DEG,
    PRIMECAM_IM0,
    PRIMECAM_IM1,
    PRIMECAM_IM2,
    PRIMECAM_IM3,
    PRIMECAM_IM4,
    PRIMECAM_IM5,
    PRIMECAM_IM6,
    PRIMECAM_MODULES,
    get_primecam_offset,
    primecam_geometry_dict,
    resolve_module_tag,
    resolve_offset,
)
from fyst_trajectories.site import FYST_PLATE_SCALE, get_fyst_site

# 461.3 mm * 13.89 arcsec/mm / 60 = 106.79 arcmin = 1.78 deg
EXPECTED_DISTANCE_ARCMIN = INNER_RING_RADIUS_MM * FYST_PLATE_SCALE / 60.0
EXPECTED_DISTANCE_DEG = EXPECTED_DISTANCE_ARCMIN / 60.0

BAY_NAMES = ["IM0", "IM1", "IM2", "IM3", "IM4", "IM5", "IM6"]
RING = [PRIMECAM_IM1, PRIMECAM_IM2, PRIMECAM_IM3, PRIMECAM_IM4, PRIMECAM_IM5, PRIMECAM_IM6]

# Focal-plane angle of each ring bay at boresight elevation 0, seen looking
# out at the sky: counter-clockwise from +cross-elevation (increasing azimuth).
RING_ANGLE_DEG = {
    "IM1": 180.0,
    "IM2": 120.0,
    "IM3": 60.0,
    "IM4": 0.0,
    "IM5": -60.0,
    "IM6": -120.0,
}

RETIRED_SENTENCE = (
    ". The labels i1..i6 are no longer defined: the ring is the bays IM1..IM6, "
    "at different positions, and i<n> is not IM<n>."
)
GENERIC_TAIL = "Available: IM0, IM1, IM2, IM3, IM4, IM5, IM6"


def _angle_diff_deg(a: float, b: float) -> float:
    """Return a - b wrapped into (-180, 180]."""
    d = (a - b) % 360.0
    return d - 360.0 if d > 180.0 else d


class TestRingGeometry:
    """Each ring bay sits at the ring radius and at its angle from the table."""

    def test_all_ring_bays_at_the_ring_radius(self):
        """All six ring bays sit 106.79 arcmin off-axis."""
        distances = [math.hypot(m.dx, m.dy) for m in RING]
        np.testing.assert_allclose(distances, EXPECTED_DISTANCE_ARCMIN, rtol=1e-12)

    def test_distance_is_1_78_degrees(self):
        """The inner ring sits 1.78 deg off-axis.

        Both inputs, the 461.3 mm ring radius and the 13.89 arcsec/mm plate
        scale, are listed under "Pending instrument verification" in
        ``docs/index.rst``; this pin moves when the instrument team confirms
        or changes either.
        """
        assert math.hypot(PRIMECAM_IM1.dx_deg, PRIMECAM_IM1.dy_deg) == pytest.approx(1.78, abs=0.01)

    @pytest.mark.parametrize("name", list(RING_ANGLE_DEG))
    def test_focal_plane_angle(self, name):
        """atan2(dy, dx) of each bay is its angle in the table."""
        offset = PRIMECAM_MODULES[name]
        angle = math.degrees(math.atan2(offset.dy, offset.dx))
        assert _angle_diff_deg(angle, RING_ANGLE_DEG[name]) == pytest.approx(0.0, abs=1e-9)

    def test_ring_runs_clockwise_looking_out(self):
        """From IM1 to IM6 the angle falls by 60 deg at each step."""
        angles = [math.degrees(math.atan2(m.dy, m.dx)) for m in RING]
        steps = [_angle_diff_deg(angles[i + 1], angles[i]) for i in range(5)]
        np.testing.assert_allclose(steps, -60.0, atol=1e-9)
        # And IM6 back to IM1 closes the ring.
        assert _angle_diff_deg(angles[0], angles[5]) == pytest.approx(-60.0, abs=1e-9)

    def test_adjacent_bays_one_radius_apart(self):
        """In a regular hexagon adjacent vertices are one circumradius apart."""
        separations = [
            math.hypot(
                RING[i].dx_deg - RING[(i + 1) % 6].dx_deg, RING[i].dy_deg - RING[(i + 1) % 6].dy_deg
            )
            for i in range(6)
        ]
        np.testing.assert_allclose(separations, EXPECTED_DISTANCE_DEG, rtol=1e-12)

    @pytest.mark.parametrize(
        "a,b",
        [(PRIMECAM_IM1, PRIMECAM_IM4), (PRIMECAM_IM2, PRIMECAM_IM5), (PRIMECAM_IM3, PRIMECAM_IM6)],
    )
    def test_opposite_bays_invert_through_the_axis(self, a, b):
        """IM1-IM4, IM2-IM5 and IM3-IM6 are diametrically opposite, exactly."""
        assert a.dx == -b.dx
        assert a.dy == -b.dy

    def test_no_two_bays_share_a_position(self):
        positions = {(m.dx, m.dy) for m in PRIMECAM_MODULES.values()}
        assert len(positions) == 7


class TestExactCoordinates:
    """The offsets come from exact millimetre coordinates, so the symmetries are exact."""

    def test_on_axis_bay_is_zero(self):
        assert PRIMECAM_IM0.dx == 0.0
        assert PRIMECAM_IM0.dy == 0.0

    def test_millimetre_coordinates(self):
        """Each bay's offset is its focal-surface position times the plate scale."""
        r = INNER_RING_RADIUS_MM
        half = r / 2.0
        height = r * math.sqrt(3.0) / 2.0
        expected_mm = {
            "IM0": (0.0, 0.0),
            "IM1": (-r, 0.0),
            "IM2": (-half, height),
            "IM3": (half, height),
            "IM4": (r, 0.0),
            "IM5": (half, -height),
            "IM6": (-half, -height),
        }
        for name, (x_mm, y_mm) in expected_mm.items():
            offset = PRIMECAM_MODULES[name]
            assert offset.dx == x_mm * FYST_PLATE_SCALE / 60.0, name
            assert offset.dy == y_mm * FYST_PLATE_SCALE / 60.0, name

    def test_arcminute_offsets(self):
        """The rounded offsets in arcminutes, as the instrument table gives them."""
        expected = {
            "IM0": (0.0, 0.0),
            "IM1": (-106.8, 0.0),
            "IM2": (-53.4, 92.5),
            "IM3": (53.4, 92.5),
            "IM4": (106.8, 0.0),
            "IM5": (53.4, -92.5),
            "IM6": (-53.4, -92.5),
        }
        for name, (dx, dy) in expected.items():
            offset = PRIMECAM_MODULES[name]
            assert round(offset.dx, 1) == dx, name
            assert round(offset.dy, 1) == dy, name

    def test_exact_symmetries(self):
        assert PRIMECAM_IM1.dy == 0.0 == PRIMECAM_IM4.dy
        assert PRIMECAM_IM1.dx == -PRIMECAM_IM4.dx
        assert PRIMECAM_IM2.dx == -PRIMECAM_IM3.dx == PRIMECAM_IM6.dx == -PRIMECAM_IM5.dx
        assert PRIMECAM_IM2.dy == PRIMECAM_IM3.dy == -PRIMECAM_IM5.dy == -PRIMECAM_IM6.dy

    def test_no_negative_zero(self):
        """No bay carries a -0.0, which would print in its repr."""
        for offset in PRIMECAM_MODULES.values():
            assert "-0.0'" not in repr(offset)
            for value in (offset.dx, offset.dy):
                assert not (value == 0.0 and math.copysign(1.0, value) < 0.0)

    def test_no_instrument_rotation(self):
        for offset in PRIMECAM_MODULES.values():
            assert offset.instrument_rotation == 0.0


class TestLayoutOnSky:
    """The layout projected on the sky through the public az/el projection.

    The bearing of each bay's detector from the boresight is measured on the
    sphere, counter-clockwise from the direction of increasing azimuth as
    seen looking out at the sky: astropy's ``position_angle`` runs from
    increasing elevation toward increasing azimuth, so the bearing is 90 deg
    minus it. The expected angles are the instrument side's table, written
    out here rather than computed from the constants.
    """

    # Bearing (deg) of IM1 .. IM6 at each boresight elevation.
    EXPECTED_BEARING_DEG = {
        0.0: [180.0, 120.0, 60.0, 0.0, -60.0, -120.0],
        30.0: [210.0, 150.0, 90.0, 30.0, -30.0, -90.0],
        60.0: [240.0, 180.0, 120.0, 60.0, 0.0, -60.0],
        70.0: [250.0, 190.0, 130.0, 70.0, 10.0, -50.0],
        90.0: [270.0, 210.0, 150.0, 90.0, 30.0, -30.0],
    }
    BORESIGHT_AZ = 120.0

    @staticmethod
    def _detector(az: float, el: float, offset: InstrumentOffset) -> tuple[float, float]:
        site = get_fyst_site()
        rotation = compute_focal_plane_rotation(el, site=site, offset=offset)
        det_az, det_el = boresight_to_detector(az, el, offset, rotation)
        return float(det_az), float(det_el)

    @pytest.mark.parametrize("el", sorted(EXPECTED_BEARING_DEG))
    def test_bearing_is_the_bay_angle_plus_elevation(self, el):
        """At el 0 the turn's sign cannot be seen; at 30, 60, 70 and 90 deg it can.

        A clockwise turn would give the same bearings at el 0 and differ by
        2 * el at the others.
        """
        boresight = SkyCoord(self.BORESIGHT_AZ * u.deg, el * u.deg)
        for name, expected in zip(
            ["IM1", "IM2", "IM3", "IM4", "IM5", "IM6"], self.EXPECTED_BEARING_DEG[el], strict=True
        ):
            det_az, det_el = self._detector(self.BORESIGHT_AZ, el, PRIMECAM_MODULES[name])
            detector = SkyCoord(det_az * u.deg, det_el * u.deg)
            bearing = 90.0 - boresight.position_angle(detector).deg
            assert _angle_diff_deg(bearing, expected) == pytest.approx(0.0, abs=1e-6), name
            assert boresight.separation(detector).deg == pytest.approx(
                EXPECTED_DISTANCE_DEG, abs=1e-9
            ), name

    def test_on_axis_bay_sits_on_the_boresight(self):
        for el in self.EXPECTED_BEARING_DEG:
            det_az, det_el = self._detector(self.BORESIGHT_AZ, el, PRIMECAM_IM0)
            assert det_az == pytest.approx(self.BORESIGHT_AZ, abs=1e-12)
            assert det_el == pytest.approx(el, abs=1e-12)

    def test_im6_prediction_at_az_120_el_45(self):
        """A prediction for an on-sky check: IM6 with the boresight at az 120, el 45."""
        det_az, det_el = self._detector(120.0, 45.0, PRIMECAM_IM6)
        cross_el = (det_az - 120.0) * math.cos(math.radians(det_el))
        assert cross_el == pytest.approx(0.461, abs=0.005)
        assert det_el - 45.0 == pytest.approx(-1.721, abs=0.005)


class TestModuleNames:
    """The names: seven bay keys, case-insensitive lookup, ``c`` and ``center`` for IM0."""

    def test_keys_are_the_seven_bays_in_order(self):
        assert list(PRIMECAM_MODULES) == BAY_NAMES

    def test_offset_names(self):
        assert [o.name for o in PRIMECAM_MODULES.values()] == [f"PrimeCam-{n}" for n in BAY_NAMES]

    def test_constants_are_the_mapping_entries(self):
        constants = [PRIMECAM_IM0, *RING]
        for name, constant in zip(BAY_NAMES, constants, strict=True):
            assert PRIMECAM_MODULES[name] is constant

    @pytest.mark.parametrize("name", BAY_NAMES)
    def test_lookup_any_case(self, name):
        offset = PRIMECAM_MODULES[name]
        for spelling in (name, name.lower(), name.capitalize(), "i" + name[1:].upper()):
            assert get_primecam_offset(spelling) is offset

    @pytest.mark.parametrize("name", ["c", "C", "center", "Center", "CENTER"])
    def test_centre_spellings_are_im0(self, name):
        assert get_primecam_offset(name) is PRIMECAM_IM0

    def test_is_read_only(self):
        """No caller can add, replace or remove a bay for every other caller."""
        with pytest.raises(TypeError):
            PRIMECAM_MODULES["IM7"] = PRIMECAM_IM0
        with pytest.raises(TypeError):
            PRIMECAM_MODULES["IM0"] = PRIMECAM_IM1
        with pytest.raises(TypeError):
            del PRIMECAM_MODULES["IM1"]

    def test_no_alias_keys(self):
        for alias in ("c", "center", "i1", "im0"):
            assert alias not in PRIMECAM_MODULES


class TestUnknownNames:
    """A miss is a KeyError; a retired ring label says the labels are gone."""

    def test_full_message_for_i1(self):
        with pytest.raises(KeyError) as excinfo:
            get_primecam_offset("i1")
        assert excinfo.value.args == (
            "Unknown PrimeCam module 'i1'. Available: IM0, IM1, IM2, IM3, IM4, IM5, IM6. "
            "The labels i1..i6 are no longer defined: the ring is the bays IM1..IM6, "
            "at different positions, and i<n> is not IM<n>.",
        )

    @pytest.mark.parametrize("name", ["i1", "i2", "i3", "i4", "i5", "i6", "I4", "I6"])
    def test_every_retired_label_carries_the_sentence(self, name):
        with pytest.raises(KeyError) as excinfo:
            get_primecam_offset(name)
        assert excinfo.value.args == (
            f"Unknown PrimeCam module '{name}'. {GENERIC_TAIL}{RETIRED_SENTENCE}",
        )

    @pytest.mark.parametrize("name", ["IM7", "zzz", "i7", "i0", "nonexistent"])
    def test_other_unknown_names_get_the_generic_text(self, name):
        with pytest.raises(KeyError) as excinfo:
            get_primecam_offset(name)
        assert excinfo.value.args == (f"Unknown PrimeCam module '{name}'. {GENERIC_TAIL}",)

    @pytest.mark.parametrize("name", [None, 5])
    def test_a_name_that_is_not_a_str_is_a_type_error(self, name):
        """A value that is not a str is refused by type, not reported as a key miss."""
        with pytest.raises(TypeError, match="module_name must be a str, got"):
            get_primecam_offset(name)

    def test_a_module_that_is_not_a_str_is_refused_through_resolve_offset(self):
        with pytest.raises(TypeError, match="module_name must be a str, got int"):
            resolve_offset(module=5)

    def test_refused_through_resolve_module_tag(self):
        with pytest.raises(KeyError, match=r"i<n> is not IM<n>"):
            resolve_module_tag("i1")
        with pytest.raises(KeyError, match=r"i<n> is not IM<n>"):
            resolve_module_tag("IM1,i2")

    def test_refused_through_resolve_offset(self):
        with pytest.raises(KeyError, match=r"i<n> is not IM<n>"):
            resolve_offset(module="i1")

    def test_refused_through_resolve_footprint(self):
        with pytest.raises(KeyError, match=r"i<n> is not IM<n>"):
            resolve_footprint("i1")

    def test_refused_through_plan_source_ces(self):
        with pytest.raises(KeyError, match=r"i<n> is not IM<n>"):
            plan_source_ces(
                body="jupiter",
                footprint="i1",
                el_bore=35.0,
                night=Time("2026-03-15T00:00:00", scale="utc"),
                mode="rising",
                site=get_fyst_site(),
            )


class TestRemovedNames:
    """The old position constants are gone from the package, with no alias."""

    @pytest.mark.parametrize(
        "name",
        [
            "PRIMECAM_CENTER",
            "PRIMECAM_I1",
            "PRIMECAM_I2",
            "PRIMECAM_I3",
            "PRIMECAM_I4",
            "PRIMECAM_I5",
            "PRIMECAM_I6",
        ],
    )
    def test_removed(self, name):
        assert not hasattr(fyst_trajectories, name)
        assert name not in fyst_trajectories.__all__
        assert not hasattr(fyst_trajectories.primecam, name)

    def test_new_constants_exported(self):
        for name in BAY_NAMES:
            assert f"PRIMECAM_{name}" in fyst_trajectories.__all__
            assert getattr(fyst_trajectories, f"PRIMECAM_{name}") is PRIMECAM_MODULES[name]


class TestResolveOffset:
    """Module name, custom dx/dy, or boresight - and a refusal when both are given."""

    def test_module_returns_the_bay(self):
        assert resolve_offset(module="IM3") is PRIMECAM_IM3
        assert resolve_offset(module="im6") is PRIMECAM_IM6

    def test_centre_spelling_returns_im0(self):
        assert resolve_offset(module="center") is PRIMECAM_IM0

    def test_custom_dx_dy_returns_instrument_offset(self):
        result = resolve_offset(dx=10.0, dy=20.0)
        assert isinstance(result, InstrumentOffset)
        assert result.dx == pytest.approx(10.0)
        assert result.dy == pytest.approx(20.0)

    def test_custom_name_is_preserved(self):
        result = resolve_offset(dx=10.0, dy=20.0, name="my-offset")
        assert result.name == "my-offset"

    def test_dx_only_defaults_dy_to_zero(self):
        result = resolve_offset(dx=10.0)
        assert result.dx == pytest.approx(10.0)
        assert result.dy == pytest.approx(0.0)

    def test_no_args_returns_none(self):
        """resolve_offset() with no arguments should return None (boresight)."""
        assert resolve_offset() is None

    def test_dy_only_returns_instrument_offset(self):
        """resolve_offset(dy=10.0) should return InstrumentOffset with dx=0.0."""
        result = resolve_offset(dy=10.0)
        assert isinstance(result, InstrumentOffset)
        assert result.dx == pytest.approx(0.0)
        assert result.dy == pytest.approx(10.0)

    def test_module_and_dx_raises_value_error(self):
        """resolve_offset(module='IM1', dx=10.0) should raise ValueError (ambiguous)."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            resolve_offset(module="IM1", dx=10.0)


class TestPrimecamGeometryDict:
    """Geometry adapter: seven bay slots, degree centers, radius, boresight shift."""

    def test_seven_bay_slots(self):
        geom = primecam_geometry_dict()
        assert list(geom) == BAY_NAMES

    def test_centre_bay_at_origin(self):
        """The IM0 slot sits on the optical axis (0, 0)."""
        geom = primecam_geometry_dict()
        assert geom["IM0"]["center"] == [0.0, 0.0]

    def test_default_radius_on_every_slot(self):
        geom = primecam_geometry_dict()
        for slot in geom.values():
            assert slot["radius"] == pytest.approx(MODULE_FOV_RADIUS_DEG)

    def test_radius_override(self):
        """radius_deg overrides the per-slot radius."""
        geom = primecam_geometry_dict(radius_deg=0.5)
        assert all(slot["radius"] == pytest.approx(0.5) for slot in geom.values())

    def test_centers_match_module_offsets_in_degrees(self):
        """Each center is the bay's (dx_deg, dy_deg) -> (xi, eta)."""
        geom = primecam_geometry_dict()
        for name, off in PRIMECAM_MODULES.items():
            assert geom[name]["center"] == pytest.approx([off.dx_deg, off.dy_deg])

    def test_boresight_offsets_shift_all_centers(self):
        """Global xi/eta offsets shift every bay center by the same amount."""
        base = primecam_geometry_dict()
        shifted = primecam_geometry_dict(xi_offset_deg=1.0, eta_offset_deg=-2.0)
        for name in base:
            assert shifted[name]["center"][0] == pytest.approx(base[name]["center"][0] + 1.0)
            assert shifted[name]["center"][1] == pytest.approx(base[name]["center"][1] - 2.0)

    def test_negative_radius_raises(self):
        """A negative radius is refused rather than passed to the scheduler schema."""
        with pytest.raises(ValueError, match="radius_deg must be positive"):
            primecam_geometry_dict(radius_deg=-1)

    def test_nan_offset_raises(self):
        """A NaN boresight offset is refused rather than passed to the scheduler schema."""
        with pytest.raises(ValueError, match="xi_offset_deg must be a finite number"):
            primecam_geometry_dict(xi_offset_deg=float("nan"))


class TestResolveModuleTag:
    """String and sequence tags: expansion, dedup, case folding, and the refusals."""

    def test_comma_tag(self):
        """A comma tag resolves to the named bays, in order."""
        offsets = resolve_module_tag("IM5,IM6")
        assert [o.name for o in offsets] == ["PrimeCam-IM5", "PrimeCam-IM6"]

    def test_sequence_equals_comma_tag(self):
        assert resolve_module_tag(["IM5", "IM6"]) == resolve_module_tag("IM5,IM6")

    def test_all_expands_to_the_seven_bays_in_order(self):
        offsets = resolve_module_tag("all")
        assert [o.name for o in offsets] == [f"PrimeCam-{n}" for n in BAY_NAMES]
        assert offsets == list(PRIMECAM_MODULES.values())

    def test_all_any_case(self):
        assert resolve_module_tag(" ALL ") == resolve_module_tag("all")

    def test_case_and_whitespace_insensitive(self):
        assert resolve_module_tag(" im5 , Im6 ") == resolve_module_tag("IM5,IM6")

    def test_centre_spellings_dedup_to_im0(self):
        offsets = resolve_module_tag("im0,c,center")
        assert offsets == [PRIMECAM_IM0]

    def test_repeated_slot_is_deduplicated(self):
        """A literally repeated slot collapses to one entry (distinct-module centroid)."""
        assert [o.name for o in resolve_module_tag("IM5,IM6,IM5")] == [
            "PrimeCam-IM5",
            "PrimeCam-IM6",
        ]

    def test_unknown_token_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown PrimeCam module 'bogus'"):
            resolve_module_tag("IM1,bogus")

    def test_concatenated_names_raise_key_error(self):
        """Concatenated names without a comma are not a recognised module."""
        with pytest.raises(KeyError, match="Unknown PrimeCam module"):
            resolve_module_tag("IM1IM2")

    @pytest.mark.parametrize("tag", ["", "   ", []])
    def test_empty_input_raises_value_error(self, tag):
        """An empty tag resolves to no modules and raises ValueError."""
        with pytest.raises(ValueError, match="resolved to no modules"):
            resolve_module_tag(tag)

    @pytest.mark.parametrize("tag", [123, None, b"IM1,IM2", bytearray(b"IM1")])
    def test_non_sequence_type_raises_type_error(self, tag):
        """A non-str, non-sequence tag (incl. bytes) raises a clear TypeError."""
        with pytest.raises(TypeError, match="str or sequence of str"):
            resolve_module_tag(tag)

    @pytest.mark.parametrize(
        "tag,kind",
        [
            (["IM1", 5], "int at index 1 of list"),
            (["IM1", None], "NoneType at index 1"),
            ((b"IM1",), "bytes at index 0 of tuple"),
        ],
    )
    def test_non_string_element_raises_type_error(self, tag, kind):
        """An element that is not a str is refused by type, as a bare non-str name is."""
        with pytest.raises(TypeError, match=f"str or sequence of str, got {kind}"):
            resolve_module_tag(tag)

    @pytest.mark.parametrize(
        "tag,names",
        [("IM5,IM6", ["IM5", "IM6"]), ("all", BAY_NAMES)],
    )
    def test_footprint_matches_hand_built_list(self, tag, names):
        from_tag = resolve_footprint(resolve_module_tag(tag))
        from_list = resolve_footprint([get_primecam_offset(n) for n in names])
        assert from_tag.center_xi_deg == pytest.approx(from_list.center_xi_deg)
        assert from_tag.center_eta_deg == pytest.approx(from_list.center_eta_deg)

    def test_the_whole_array_is_centred_exactly_on_the_boresight(self):
        """The seven bays average to exactly zero, with no rounding residue.

        The ring is symmetric about the boresight, so its centre is the
        boresight itself. A plain floating-point mean of the seven centres
        leaves about -3e-17 deg in cross-elevation, which a pass summary
        prints as ``-0.000``.
        """
        whole = resolve_footprint(resolve_module_tag("all"))
        assert (whole.center_xi_deg, whole.center_eta_deg) == (0.0, 0.0)
        assert math.copysign(1.0, whole.center_xi_deg) == 1.0
        assert math.copysign(1.0, whole.center_eta_deg) == 1.0
