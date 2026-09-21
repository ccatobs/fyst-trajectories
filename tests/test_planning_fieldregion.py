"""Tests for the FieldRegion dataclass."""

import pytest

from fyst_trajectories.planning import FieldRegion


class TestFieldRegion:
    """Derived dec bounds, and the refusal of non-positive width or height."""

    def test_dec_boundaries(self):
        field = FieldRegion(ra_center=180.0, dec_center=-30.0, width=2.0, height=4.0)
        assert field.dec_min == pytest.approx(-32.0)
        assert field.dec_max == pytest.approx(-28.0)

    @pytest.mark.parametrize("width", [0.0, -1.0])
    def test_non_positive_width_raises(self, width):
        with pytest.raises(ValueError, match="width must be positive"):
            FieldRegion(ra_center=0.0, dec_center=0.0, width=width, height=1.0)

    @pytest.mark.parametrize("height", [0.0, -2.0])
    def test_non_positive_height_raises(self, height):
        with pytest.raises(ValueError, match="height must be positive"):
            FieldRegion(ra_center=0.0, dec_center=0.0, width=1.0, height=height)
