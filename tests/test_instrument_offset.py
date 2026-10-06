"""Tests for the InstrumentOffset value type: its repr and from_focal_plane."""

import pytest

from fyst_trajectories.offsets import (
    InstrumentOffset,
)


class TestInstrumentRotationRepr:
    """``repr`` shows ``instrument_rotation`` only when it is non-zero."""

    def test_repr_without_instrument_rotation(self):
        offset = InstrumentOffset(dx=5.0, dy=3.0, name="Test")
        r = repr(offset)
        assert "instrument_rotation" not in r
        assert "dx=5.0'" in r
        assert "dy=3.0'" in r
        assert "name='Test'" in r

    def test_repr_with_instrument_rotation(self):
        offset = InstrumentOffset(dx=5.0, dy=3.0, instrument_rotation=15.0)
        r = repr(offset)
        assert "instrument_rotation=15.0" in r


class TestFromFocalPlane:
    """The mm-to-arcmin factory: the plate-scale conversion and the pass-through fields."""

    def test_basic_conversion(self):
        """The factory converts mm to arcmin through the plate scale."""
        offset = InstrumentOffset.from_focal_plane(x_mm=0.0, y_mm=-461.3, plate_scale=13.89)
        # 461.3 mm * 13.89 arcsec/mm / 60 = 106.79 arcmin
        assert offset.dx == pytest.approx(0.0, abs=1e-10)
        assert offset.dy == pytest.approx(-106.79, abs=0.01)

    def test_name_passed_through(self):
        offset = InstrumentOffset.from_focal_plane(
            x_mm=0.0, y_mm=-461.3, plate_scale=13.89, name="TestModule"
        )
        assert offset.name == "TestModule"

    def test_instrument_rotation_passed_through(self):
        offset = InstrumentOffset.from_focal_plane(
            x_mm=0.0, y_mm=-461.3, plate_scale=13.89, instrument_rotation=15.0
        )
        assert offset.instrument_rotation == pytest.approx(15.0)

    def test_zero_position_returns_zero_offset(self):
        offset = InstrumentOffset.from_focal_plane(x_mm=0.0, y_mm=0.0, plate_scale=13.89)
        assert offset.dx == pytest.approx(0.0, abs=1e-10)
        assert offset.dy == pytest.approx(0.0, abs=1e-10)

    def test_consistency_with_manual_calculation(self):
        x_mm, y_mm, plate_scale = 100.0, 200.0, 13.89

        # Manual calculation
        dx_arcmin_manual = x_mm * plate_scale / 60.0
        dy_arcmin_manual = y_mm * plate_scale / 60.0

        # Via factory
        offset = InstrumentOffset.from_focal_plane(x_mm=x_mm, y_mm=y_mm, plate_scale=plate_scale)

        assert offset.dx == pytest.approx(dx_arcmin_manual, abs=1e-10)
        assert offset.dy == pytest.approx(dy_arcmin_manual, abs=1e-10)

    def test_symmetric_positions(self):
        plate_scale = 13.89

        offset_pos = InstrumentOffset.from_focal_plane(
            x_mm=100.0, y_mm=100.0, plate_scale=plate_scale
        )
        offset_neg = InstrumentOffset.from_focal_plane(
            x_mm=-100.0, y_mm=-100.0, plate_scale=plate_scale
        )

        assert offset_neg.dx == pytest.approx(-offset_pos.dx, abs=1e-10)
        assert offset_neg.dy == pytest.approx(-offset_pos.dy, abs=1e-10)
