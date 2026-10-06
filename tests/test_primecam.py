"""Tests for PrimeCam module positions and offsets.

Validates the hexagonal geometry of the PrimeCam inner-ring modules (standard
polar convention, x = r*cos(theta), y = r*sin(theta)), the module-name lookup
and tag-resolution helpers, the scheduler geometry dict, and the IM position
designations.
"""

import numpy as np
import pytest

from fyst_trajectories.offsets import InstrumentOffset
from fyst_trajectories.primecam import (
    INNER_RING_RADIUS_MM,
    MODULE_FOV_RADIUS_DEG,
    PRIMECAM_CENTER,
    PRIMECAM_I1,
    PRIMECAM_I2,
    PRIMECAM_I3,
    PRIMECAM_I4,
    PRIMECAM_I5,
    PRIMECAM_I6,
    PRIMECAM_MODULES,
    get_primecam_offset,
    primecam_geometry_dict,
    resolve_module_tag,
    resolve_offset,
)
from fyst_trajectories.site import get_fyst_site

# Expected angular distance of inner ring from center in arcminutes.
# 461.3 mm * 13.89 arcsec/mm / 60 = 106.79 arcmin = 1.78 deg
_PLATE_SCALE = get_fyst_site().plate_scale
EXPECTED_DISTANCE_ARCMIN = INNER_RING_RADIUS_MM * _PLATE_SCALE / 60.0
EXPECTED_DISTANCE_DEG = EXPECTED_DISTANCE_ARCMIN / 60.0


class TestHexagonalSymmetry:
    """All inner ring modules should be at the same radial distance from center."""

    def test_all_inner_ring_equidistant(self):
        """All 6 inner ring modules at same distance (~1.78 deg = ~106.8 arcmin)."""
        modules = [PRIMECAM_I1, PRIMECAM_I2, PRIMECAM_I3, PRIMECAM_I4, PRIMECAM_I5, PRIMECAM_I6]
        distances = [np.sqrt(m.dx**2 + m.dy**2) for m in modules]

        np.testing.assert_allclose(distances, EXPECTED_DISTANCE_ARCMIN, rtol=1e-6)

    def test_distance_is_1_78_degrees(self):
        """The inner ring sits 1.78 deg off-axis.

        Both inputs, the 461.3 mm ring radius and the 13.89 arcsec/mm plate
        scale, are listed under "Pending instrument verification" in
        ``docs/index.rst``; this pin moves when the instrument team confirms
        or changes either.
        """
        dist_deg = np.sqrt(PRIMECAM_I1.dx_deg**2 + PRIMECAM_I1.dy_deg**2)
        assert dist_deg == pytest.approx(1.78, abs=0.01)


class TestAxisAlignedModules:
    """I1 and I4 should be along the y-axis (dx approximately 0)."""

    def test_i1_on_y_axis(self):
        """I1 (theta=-90) should have dx=0, dy<0."""
        assert PRIMECAM_I1.dx == pytest.approx(0.0, abs=1e-10)
        assert PRIMECAM_I1.dy < 0

    def test_i4_on_y_axis(self):
        """I4 (theta=+90) should have dx=0, dy>0."""
        assert PRIMECAM_I4.dx == pytest.approx(0.0, abs=1e-10)
        assert PRIMECAM_I4.dy > 0

    def test_i1_i4_diametrically_opposite(self):
        assert PRIMECAM_I1.dx == pytest.approx(-PRIMECAM_I4.dx, abs=1e-10)
        assert PRIMECAM_I1.dy == pytest.approx(-PRIMECAM_I4.dy, abs=1e-10)


class TestMirrorSymmetry:
    """Adjacent pairs mirror about the x-axis; opposite pairs invert through the origin."""

    def test_i2_i3_mirror_symmetry(self):
        """I2 and I3 should be mirror images across the x-axis."""
        assert PRIMECAM_I2.dx == pytest.approx(PRIMECAM_I3.dx, abs=1e-10)
        assert PRIMECAM_I2.dy == pytest.approx(-PRIMECAM_I3.dy, abs=1e-10)

    def test_i5_i6_mirror_symmetry(self):
        """I5 and I6 should be mirror images across the x-axis."""
        assert PRIMECAM_I5.dx == pytest.approx(PRIMECAM_I6.dx, abs=1e-10)
        assert PRIMECAM_I5.dy == pytest.approx(-PRIMECAM_I6.dy, abs=1e-10)

    def test_i2_i5_diametrically_opposite(self):
        """I2 and I5 are diametrically opposite (180 deg apart on the ring)."""
        assert PRIMECAM_I2.dx == pytest.approx(-PRIMECAM_I5.dx, abs=1e-10)
        assert PRIMECAM_I2.dy == pytest.approx(-PRIMECAM_I5.dy, abs=1e-10)


class TestAdjacentModuleSeparation:
    """Adjacent modules are separated by the ring radius (hexagonal geometry).

    In a regular hexagon with circumradius R, adjacent vertices are separated
    by exactly R. So adjacent modules should be separated by the same distance
    as the ring radius from center.
    """

    def _angular_separation(self, m1, m2):
        """Compute angular separation between two modules in degrees."""
        ddx = m1.dx_deg - m2.dx_deg
        ddy = m1.dy_deg - m2.dy_deg
        return np.sqrt(ddx**2 + ddy**2)

    def test_all_adjacent_separations(self):
        ordered = [PRIMECAM_I1, PRIMECAM_I2, PRIMECAM_I3, PRIMECAM_I4, PRIMECAM_I5, PRIMECAM_I6]
        separations = []
        for i in range(6):
            sep = self._angular_separation(ordered[i], ordered[(i + 1) % 6])
            separations.append(sep)

        np.testing.assert_allclose(separations, EXPECTED_DISTANCE_DEG, rtol=0.01)

    def test_diametrically_opposite_separation(self):
        """Opposite modules (I1-I4, I2-I5, I3-I6) separated by 2*R."""
        for m1, m2 in [
            (PRIMECAM_I1, PRIMECAM_I4),
            (PRIMECAM_I2, PRIMECAM_I5),
            (PRIMECAM_I3, PRIMECAM_I6),
        ]:
            sep = self._angular_separation(m1, m2)
            assert sep == pytest.approx(2 * EXPECTED_DISTANCE_DEG, rel=0.01)


class TestCartesianPositions:
    """Verify each module's arcminute offset against its millimetre focal-plane position."""

    def test_i1_position_mm(self):
        """I1: (0, -461.3) mm (the ring radius is pending instrument verification)."""
        assert PRIMECAM_I1.dx == pytest.approx(0.0 * _PLATE_SCALE / 60.0, abs=1e-10)
        assert PRIMECAM_I1.dy == pytest.approx(-461.3 * _PLATE_SCALE / 60.0, rel=1e-6)

    def test_i2_position_mm(self):
        """I2: (399.5, -230.65) mm."""
        expected_x_mm = INNER_RING_RADIUS_MM * np.cos(np.deg2rad(-30))
        expected_y_mm = INNER_RING_RADIUS_MM * np.sin(np.deg2rad(-30))
        assert PRIMECAM_I2.dx == pytest.approx(expected_x_mm * _PLATE_SCALE / 60.0, abs=0.01)
        assert PRIMECAM_I2.dy == pytest.approx(expected_y_mm * _PLATE_SCALE / 60.0, abs=0.01)

    def test_i3_position_mm(self):
        """I3: (399.5, 230.65) mm."""
        expected_x_mm = INNER_RING_RADIUS_MM * np.cos(np.deg2rad(30))
        expected_y_mm = INNER_RING_RADIUS_MM * np.sin(np.deg2rad(30))
        assert PRIMECAM_I3.dx == pytest.approx(expected_x_mm * _PLATE_SCALE / 60.0, abs=0.01)
        assert PRIMECAM_I3.dy == pytest.approx(expected_y_mm * _PLATE_SCALE / 60.0, abs=0.01)

    def test_i5_position_mm(self):
        """I5: (-399.5, 230.65) mm."""
        expected_x_mm = INNER_RING_RADIUS_MM * np.cos(np.deg2rad(150))
        expected_y_mm = INNER_RING_RADIUS_MM * np.sin(np.deg2rad(150))
        assert PRIMECAM_I5.dx == pytest.approx(expected_x_mm * _PLATE_SCALE / 60.0, abs=0.01)
        assert PRIMECAM_I5.dy == pytest.approx(expected_y_mm * _PLATE_SCALE / 60.0, abs=0.01)

    def test_i6_position_mm(self):
        """I6: (-399.5, -230.65) mm."""
        expected_x_mm = INNER_RING_RADIUS_MM * np.cos(np.deg2rad(-150))
        expected_y_mm = INNER_RING_RADIUS_MM * np.sin(np.deg2rad(-150))
        assert PRIMECAM_I6.dx == pytest.approx(expected_x_mm * _PLATE_SCALE / 60.0, abs=0.01)
        assert PRIMECAM_I6.dy == pytest.approx(expected_y_mm * _PLATE_SCALE / 60.0, abs=0.01)


class TestAllModulesDistinct:
    """All 7 modules should occupy unique positions."""

    def test_no_duplicate_positions(self):
        """No two non-alias modules share the same (dx, dy) position."""
        # Use one key per physical module (exclude 'center' alias for 'c')
        unique_keys = ["c", "i1", "i2", "i3", "i4", "i5", "i6"]
        positions = [(PRIMECAM_MODULES[k].dx, PRIMECAM_MODULES[k].dy) for k in unique_keys]
        assert len(set(positions)) == len(positions)


class TestGetPrimecamOffset:
    """Name lookup: every module key, case-insensitively, with a KeyError on a miss."""

    def test_returns_correct_module(self):
        assert get_primecam_offset("c") is PRIMECAM_CENTER
        assert get_primecam_offset("center") is PRIMECAM_CENTER
        assert get_primecam_offset("i1") is PRIMECAM_I1
        assert get_primecam_offset("i2") is PRIMECAM_I2
        assert get_primecam_offset("i3") is PRIMECAM_I3
        assert get_primecam_offset("i4") is PRIMECAM_I4
        assert get_primecam_offset("i5") is PRIMECAM_I5
        assert get_primecam_offset("i6") is PRIMECAM_I6

    def test_case_insensitive(self):
        assert get_primecam_offset("I1") is PRIMECAM_I1
        assert get_primecam_offset("CENTER") is PRIMECAM_CENTER

    def test_unknown_module_raises(self):
        """Unknown module name should raise KeyError."""
        with pytest.raises(KeyError, match="Unknown PrimeCam module"):
            get_primecam_offset("nonexistent")


class TestCenterModule:
    """The center module is the zero offset on the optical axis."""

    def test_center_is_zero(self):
        assert PRIMECAM_CENTER.dx == 0.0
        assert PRIMECAM_CENTER.dy == 0.0


class TestModulesDict:
    """``PRIMECAM_MODULES`` holds eight keys, with ``center`` an alias of ``c``."""

    def test_contains_all_modules(self):
        expected_keys = {"c", "center", "i1", "i2", "i3", "i4", "i5", "i6"}
        assert set(PRIMECAM_MODULES.keys()) == expected_keys

    def test_center_aliases(self):
        """Both 'c' and 'center' should reference the same object."""
        assert PRIMECAM_MODULES["c"] is PRIMECAM_MODULES["center"]

    def test_is_read_only(self):
        """No caller can add, replace or remove a module for every other caller."""
        with pytest.raises(TypeError):
            PRIMECAM_MODULES["i7"] = PRIMECAM_CENTER
        with pytest.raises(TypeError):
            PRIMECAM_MODULES["c"] = PRIMECAM_I1
        with pytest.raises(TypeError):
            del PRIMECAM_MODULES["i1"]


class TestResolveOffset:
    """Module name, custom dx/dy, or boresight - and a refusal when both are given."""

    def test_module_i1_returns_primecam_i1(self):
        assert resolve_offset(module="i1") is PRIMECAM_I1

    def test_module_i3_returns_primecam_i3(self):
        assert resolve_offset(module="i3") is PRIMECAM_I3

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
        """resolve_offset(module='i1', dx=10.0) should raise ValueError (ambiguous)."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            resolve_offset(module="i1", dx=10.0)


class TestPrimecamGeometryDict:
    """Geometry adapter: seven deduped slots, degree centers, radius, boresight shift."""

    def test_seven_slots_and_center_alias_deduped(self):
        """Returns 7 slots ('c' + i1..i6); the duplicate 'center' alias is dropped."""
        geom = primecam_geometry_dict()
        assert set(geom) == {"c", "i1", "i2", "i3", "i4", "i5", "i6"}
        assert "center" not in geom

    def test_center_module_at_origin(self):
        """The 'c' module sits on the optical axis (0, 0)."""
        geom = primecam_geometry_dict()
        assert geom["c"]["center"] == pytest.approx([0.0, 0.0])

    def test_default_radius_on_every_slot(self):
        geom = primecam_geometry_dict()
        for slot in geom.values():
            assert slot["radius"] == pytest.approx(MODULE_FOV_RADIUS_DEG)

    def test_radius_override(self):
        """radius_deg overrides the per-slot radius."""
        geom = primecam_geometry_dict(radius_deg=0.5)
        assert all(slot["radius"] == pytest.approx(0.5) for slot in geom.values())

    def test_centers_match_module_offsets_in_degrees(self):
        """Each center is the module's (dx_deg, dy_deg) -> (xi, eta)."""
        geom = primecam_geometry_dict()
        for name in ("c", "i1", "i2", "i3", "i4", "i5", "i6"):
            off = PRIMECAM_MODULES[name]
            assert geom[name]["center"] == pytest.approx([off.dx_deg, off.dy_deg])

    def test_boresight_offsets_shift_all_centers(self):
        """Global xi/eta offsets shift every module center by the same amount."""
        base = primecam_geometry_dict()
        shifted = primecam_geometry_dict(xi_offset_deg=1.0, eta_offset_deg=-2.0)
        for name in base:
            assert shifted[name]["center"][0] == pytest.approx(base[name]["center"][0] + 1.0)
            assert shifted[name]["center"][1] == pytest.approx(base[name]["center"][1] - 2.0)


class TestResolveModuleTag:
    """String and sequence tags: expansion, dedup, case folding, and the refusals."""

    def test_comma_tag(self):
        """A comma tag resolves to the named modules, in order."""
        offsets = resolve_module_tag("i1,i2")
        assert [o.name for o in offsets] == ["PrimeCam-I1", "PrimeCam-I2"]

    def test_sequence_equals_comma_tag(self):
        assert resolve_module_tag(["i1", "i2"]) == resolve_module_tag("i1,i2")

    def test_all_expands_to_seven_modules(self):
        """'all' expands to c, i1..i6 (the 'center' alias is dropped)."""
        names = [o.name for o in resolve_module_tag("all")]
        assert names == [
            "PrimeCam-Center",
            "PrimeCam-I1",
            "PrimeCam-I2",
            "PrimeCam-I3",
            "PrimeCam-I4",
            "PrimeCam-I5",
            "PrimeCam-I6",
        ]

    def test_case_and_whitespace_insensitive(self):
        assert resolve_module_tag(" I1 , I2 ") == resolve_module_tag("i1,i2")

    def test_c_and_center_dedup_to_single_module(self):
        offsets = resolve_module_tag("c,center")
        assert len(offsets) == 1
        assert offsets[0] is get_primecam_offset("c")

    def test_repeated_slot_is_deduplicated(self):
        """A literally repeated slot collapses to one entry (distinct-module centroid)."""
        assert [o.name for o in resolve_module_tag("i1,i2,i1")] == ["PrimeCam-I1", "PrimeCam-I2"]

    def test_unknown_token_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown PrimeCam module"):
            resolve_module_tag("i1,bogus")

    def test_concatenated_names_raise_key_error(self):
        """Concatenated names without a comma are not a recognised module."""
        with pytest.raises(KeyError, match="Unknown PrimeCam module"):
            resolve_module_tag("i1i2")

    @pytest.mark.parametrize("tag", ["", "   ", []])
    def test_empty_input_raises_value_error(self, tag):
        """An empty tag resolves to no modules and raises ValueError."""
        with pytest.raises(ValueError, match="resolved to no modules"):
            resolve_module_tag(tag)

    @pytest.mark.parametrize("tag", [123, None, b"i1,i2", bytearray(b"i1")])
    def test_non_sequence_type_raises_type_error(self, tag):
        """A non-str, non-sequence tag (incl. bytes) raises a clear TypeError."""
        with pytest.raises(TypeError, match="str or sequence of str"):
            resolve_module_tag(tag)

    def test_non_string_element_raises_key_error(self):
        """A non-string element is stringified and rejected as an unknown module."""
        with pytest.raises(KeyError, match="Unknown PrimeCam module"):
            resolve_module_tag(["i1", 5])

    @pytest.mark.parametrize(
        "tag,names",
        [("i1,i2", ["i1", "i2"]), ("all", ["c", "i1", "i2", "i3", "i4", "i5", "i6"])],
    )
    def test_footprint_matches_hand_built_list(self, tag, names):
        from fyst_trajectories.planning.footprints import resolve_footprint

        from_tag = resolve_footprint(resolve_module_tag(tag))
        from_list = resolve_footprint([get_primecam_offset(n) for n in names])
        assert from_tag.center_xi_deg == pytest.approx(from_list.center_xi_deg)
        assert from_tag.center_eta_deg == pytest.approx(from_list.center_eta_deg)


class TestIMDesignations:
    """The instrument team's IM position designations.

    IM0 is the on-axis position and mirror-invariant, so it is aliased to
    ``c`` today. The ring designations IM1..IM6 do NOT correspond
    index-for-index to i1..i6 (the schemes number the ring in opposite
    senses on sky), so they are rejected until the as-built correspondence
    is confirmed; these tests are the guard against anyone assuming
    ``i(n) == IM(n)``.
    """

    @pytest.mark.parametrize("name", ["IM0", "im0", "Im0"])
    def test_im0_is_the_center_module(self, name):
        """IM0 resolves to the identical center offset object, any case."""
        assert get_primecam_offset(name) is PRIMECAM_CENTER

    def test_im0_through_resolve_offset(self):
        assert resolve_offset(module="IM0") is PRIMECAM_CENTER

    def test_im0_through_resolve_module_tag_dedups_with_c(self):
        offsets = resolve_module_tag("im0,c,center")
        assert len(offsets) == 1
        assert offsets[0] is PRIMECAM_CENTER

    @pytest.mark.parametrize("name", ["IM1", "IM2", "IM3", "IM4", "IM5", "IM6"])
    def test_im_ring_designations_rejected_until_confirmed(self, name):
        """IM1..IM6 fail loud: the i<->IM ring correspondence is unconfirmed."""
        with pytest.raises(KeyError, match="Unknown PrimeCam module"):
            get_primecam_offset(name)

    def test_error_message_advertises_im0(self):
        """The unknown-module error lists im0 among the available names."""
        with pytest.raises(KeyError, match="im0"):
            get_primecam_offset("nope")
