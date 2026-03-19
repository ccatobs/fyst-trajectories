"""Tests for PrimeCam module positions and offsets.

Validates the hexagonal geometry of the PrimeCam inner ring modules,
ensuring correct trigonometric convention (standard polar: x = r*cos(theta),
y = r*sin(theta)) and cross-validates angular separations against scan_patterns.
"""

import numpy as np
import pytest

from fyst_trajectories.offsets import InstrumentOffset
from fyst_trajectories.primecam import (
    INNER_RING_RADIUS_MM,
    PRIMECAM_CENTER,
    PRIMECAM_I1,
    PRIMECAM_I2,
    PRIMECAM_I3,
    PRIMECAM_I4,
    PRIMECAM_I5,
    PRIMECAM_I6,
    PRIMECAM_MODULES,
    get_primecam_offset,
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
        """Inner ring distance should be approximately 1.78 degrees."""
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
        """I1 and I4 should be diametrically opposite."""
        assert PRIMECAM_I1.dx == pytest.approx(-PRIMECAM_I4.dx, abs=1e-10)
        assert PRIMECAM_I1.dy == pytest.approx(-PRIMECAM_I4.dy, abs=1e-10)


class TestMirrorSymmetry:
    """Adjacent module pairs should exhibit mirror symmetry about x-axis."""

    def test_i2_i3_mirror_symmetry(self):
        """I2 and I3 should be mirror images across the x-axis."""
        assert PRIMECAM_I2.dx == pytest.approx(PRIMECAM_I3.dx, abs=1e-10)
        assert PRIMECAM_I2.dy == pytest.approx(-PRIMECAM_I3.dy, abs=1e-10)

    def test_i5_i6_mirror_symmetry(self):
        """I5 and I6 should be mirror images across the x-axis."""
        assert PRIMECAM_I5.dx == pytest.approx(PRIMECAM_I6.dx, abs=1e-10)
        assert PRIMECAM_I5.dy == pytest.approx(-PRIMECAM_I6.dy, abs=1e-10)

    def test_i2_i5_mirror_symmetry(self):
        """I2 and I5 should be mirror images across the y-axis."""
        assert PRIMECAM_I2.dx == pytest.approx(-PRIMECAM_I5.dx, abs=1e-10)
        assert PRIMECAM_I2.dy == pytest.approx(-PRIMECAM_I5.dy, abs=1e-10)


class TestAdjacentModuleSeparation:
    """Adjacent modules should be separated by ~1.78 deg (hexagonal geometry).

    In a regular hexagon with circumradius R, adjacent vertices are separated
    by exactly R. So adjacent modules should be separated by the same distance
    as the ring radius from center.
    """

    def _angular_separation(self, m1, m2):
        """Compute angular separation between two modules in degrees."""
        ddx = m1.dx_deg - m2.dx_deg
        ddy = m1.dy_deg - m2.dy_deg
        return np.sqrt(ddx**2 + ddy**2)

    def test_i1_i2_separation(self):
        """I1-I2 separation should be ~1.78 deg (not ~0.92 deg from the old bug)."""
        sep = self._angular_separation(PRIMECAM_I1, PRIMECAM_I2)
        assert sep == pytest.approx(EXPECTED_DISTANCE_DEG, rel=0.01)
        # The buggy code gave ~0.92 deg; verify we are NOT close to that
        assert sep > 1.5, f"Separation {sep:.2f} deg is too small (old sin/cos bug?)"

    def test_i1_i6_separation(self):
        """I1-I6 separation should be ~1.78 deg."""
        sep = self._angular_separation(PRIMECAM_I1, PRIMECAM_I6)
        assert sep == pytest.approx(EXPECTED_DISTANCE_DEG, rel=0.01)

    def test_all_adjacent_separations(self):
        """All adjacent module pairs should have the same separation."""
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
    """Verify expected Cartesian positions in mm (before plate-scale conversion)."""

    def test_i1_position_mm(self):
        """I1: (0, -461.3) mm."""
        assert PRIMECAM_I1.dx == pytest.approx(0.0 * _PLATE_SCALE / 60.0, abs=1e-10)
        assert PRIMECAM_I1.dy == pytest.approx(-461.3 * _PLATE_SCALE / 60.0, abs=0.01)

    def test_i2_position_mm(self):
        """I2: (399.6, -230.65) mm."""
        expected_x_mm = INNER_RING_RADIUS_MM * np.cos(np.deg2rad(-30))
        expected_y_mm = INNER_RING_RADIUS_MM * np.sin(np.deg2rad(-30))
        assert PRIMECAM_I2.dx == pytest.approx(expected_x_mm * _PLATE_SCALE / 60.0, abs=0.01)
        assert PRIMECAM_I2.dy == pytest.approx(expected_y_mm * _PLATE_SCALE / 60.0, abs=0.01)

    def test_i3_position_mm(self):
        """I3: (399.6, 230.65) mm."""
        expected_x_mm = INNER_RING_RADIUS_MM * np.cos(np.deg2rad(30))
        expected_y_mm = INNER_RING_RADIUS_MM * np.sin(np.deg2rad(30))
        assert PRIMECAM_I3.dx == pytest.approx(expected_x_mm * _PLATE_SCALE / 60.0, abs=0.01)
        assert PRIMECAM_I3.dy == pytest.approx(expected_y_mm * _PLATE_SCALE / 60.0, abs=0.01)

    def test_i5_position_mm(self):
        """I5: (-399.6, 230.65) mm."""
        expected_x_mm = INNER_RING_RADIUS_MM * np.cos(np.deg2rad(150))
        expected_y_mm = INNER_RING_RADIUS_MM * np.sin(np.deg2rad(150))
        assert PRIMECAM_I5.dx == pytest.approx(expected_x_mm * _PLATE_SCALE / 60.0, abs=0.01)
        assert PRIMECAM_I5.dy == pytest.approx(expected_y_mm * _PLATE_SCALE / 60.0, abs=0.01)

    def test_i6_position_mm(self):
        """I6: (-399.6, -230.65) mm."""
        expected_x_mm = INNER_RING_RADIUS_MM * np.cos(np.deg2rad(-150))
        expected_y_mm = INNER_RING_RADIUS_MM * np.sin(np.deg2rad(-150))
        assert PRIMECAM_I6.dx == pytest.approx(expected_x_mm * _PLATE_SCALE / 60.0, abs=0.01)
        assert PRIMECAM_I6.dy == pytest.approx(expected_y_mm * _PLATE_SCALE / 60.0, abs=0.01)


class TestGetPrimecamOffset:
    """Tests for get_primecam_offset function."""

    def test_returns_correct_module(self):
        """get_primecam_offset returns the correct module for each name."""
        assert get_primecam_offset("c") is PRIMECAM_CENTER
        assert get_primecam_offset("center") is PRIMECAM_CENTER
        assert get_primecam_offset("i1") is PRIMECAM_I1
        assert get_primecam_offset("i2") is PRIMECAM_I2
        assert get_primecam_offset("i3") is PRIMECAM_I3
        assert get_primecam_offset("i4") is PRIMECAM_I4
        assert get_primecam_offset("i5") is PRIMECAM_I5
        assert get_primecam_offset("i6") is PRIMECAM_I6

    def test_case_insensitive(self):
        """Module names should be case-insensitive."""
        assert get_primecam_offset("I1") is PRIMECAM_I1
        assert get_primecam_offset("CENTER") is PRIMECAM_CENTER

    def test_unknown_module_raises(self):
        """Unknown module name should raise KeyError."""
        with pytest.raises(KeyError, match="Unknown PrimeCam module"):
            get_primecam_offset("nonexistent")


class TestCenterModule:
    """Tests for the center module."""

    def test_center_is_zero(self):
        """Center module should have zero offset."""
        assert PRIMECAM_CENTER.dx == 0.0
        assert PRIMECAM_CENTER.dy == 0.0

    def test_center_name(self):
        """Center module should have correct name."""
        assert PRIMECAM_CENTER.name == "PrimeCam-Center"


class TestModulesDict:
    """Tests for the PRIMECAM_MODULES dictionary."""

    def test_contains_all_modules(self):
        """Dictionary should contain center and all 6 inner ring modules."""
        expected_keys = {"c", "center", "i1", "i2", "i3", "i4", "i5", "i6"}
        assert set(PRIMECAM_MODULES.keys()) == expected_keys

    def test_center_aliases(self):
        """Both 'c' and 'center' should reference the same object."""
        assert PRIMECAM_MODULES["c"] is PRIMECAM_MODULES["center"]


class TestResolveOffset:
    """Tests for the resolve_offset function."""

    def test_module_i1_returns_primecam_i1(self):
        """resolve_offset(module='i1') should return the same object as PRIMECAM_I1."""
        assert resolve_offset(module="i1") is PRIMECAM_I1

    def test_module_i3_returns_primecam_i3(self):
        """resolve_offset(module='i3') should return the same object as PRIMECAM_I3."""
        assert resolve_offset(module="i3") is PRIMECAM_I3

    def test_custom_dx_dy_returns_instrument_offset(self):
        """resolve_offset(dx=10.0, dy=20.0) should return InstrumentOffset with those values."""
        result = resolve_offset(dx=10.0, dy=20.0)
        assert isinstance(result, InstrumentOffset)
        assert result.dx == pytest.approx(10.0)
        assert result.dy == pytest.approx(20.0)

    def test_custom_name_is_preserved(self):
        """resolve_offset(dx=10.0, dy=20.0, name='my-offset') should set the name."""
        result = resolve_offset(dx=10.0, dy=20.0, name="my-offset")
        assert result.name == "my-offset"

    def test_dx_only_defaults_dy_to_zero(self):
        """resolve_offset(dx=10.0) should default dy to 0.0."""
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

    def test_dy_only_with_dx_none(self):
        """resolve_offset(dx=None, dy=5.0) should return InstrumentOffset with dx=0.0."""
        result = resolve_offset(dx=None, dy=5.0)
        assert isinstance(result, InstrumentOffset)
        assert result.dx == pytest.approx(0.0)
        assert result.dy == pytest.approx(5.0)

    def test_module_and_dx_raises_value_error(self):
        """resolve_offset(module='i1', dx=10.0) should raise ValueError (ambiguous)."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            resolve_offset(module="i1", dx=10.0)
