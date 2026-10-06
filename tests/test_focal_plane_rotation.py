"""Tests for compute_focal_plane_rotation."""

import numpy as np
import pytest
from _site_ports import _site_with_port

from fyst_trajectories.offsets import (
    InstrumentOffset,
    compute_focal_plane_rotation,
)


class TestComputeFocalPlaneRotation:
    """The rotation is ``nasmyth_sign * el + instrument_rotation + pa``.

    Covers the right-Nasmyth, left-Nasmyth and Cassegrain sign cases, all three
    terms together, and the array path, where a per-sample trajectory's
    elevation and parallactic angle broadcast elementwise.
    """

    def test_right_nasmyth_positive(self):
        site = _site_with_port("right")
        offset = InstrumentOffset(dx=0.0, dy=0.0)
        rot = compute_focal_plane_rotation(45.0, site=site, offset=offset)
        # site.nasmyth_sign = +1, so rotation = +1 * 45 + 0 + 0 = 45
        assert rot == pytest.approx(45.0)

    def test_with_parallactic_angle(self):
        site = _site_with_port("right")
        offset = InstrumentOffset(dx=0.0, dy=0.0)
        rot = compute_focal_plane_rotation(45.0, site=site, offset=offset, parallactic_angle=10.0)
        assert rot == pytest.approx(55.0)

    def test_with_instrument_rotation(self):
        site = _site_with_port("right")
        offset = InstrumentOffset(dx=0.0, dy=0.0, instrument_rotation=15.0)
        rot = compute_focal_plane_rotation(45.0, site=site, offset=offset)
        # +1 * 45 + 15 + 0 = 60
        assert rot == pytest.approx(60.0)

    def test_array_input(self):
        site = _site_with_port("right")
        offset = InstrumentOffset(dx=0.0, dy=0.0)
        el = np.array([30.0, 45.0, 60.0])
        rot = compute_focal_plane_rotation(el, site=site, offset=offset)
        np.testing.assert_allclose(rot, el)

    def test_cassegrain_elevation_does_not_contribute(self):
        """Cassegrain (``nasmyth_sign=0``) drops the elevation term entirely."""
        cass_site = _site_with_port("cassegrain")
        assert cass_site.nasmyth_sign == 0

        offset = InstrumentOffset(dx=5.0, dy=3.0)
        # At various elevations, rotation should be the same (0*el + 0 + 0 = 0)
        rot_30 = compute_focal_plane_rotation(30.0, site=cass_site, offset=offset)
        rot_60 = compute_focal_plane_rotation(60.0, site=cass_site, offset=offset)
        rot_85 = compute_focal_plane_rotation(85.0, site=cass_site, offset=offset)

        assert rot_30 == pytest.approx(0.0)
        assert rot_60 == pytest.approx(0.0)
        assert rot_85 == pytest.approx(0.0)

    def test_cassegrain_with_parallactic_angle(self):
        """Cassegrain still ignores elevation when a parallactic angle is supplied."""
        cass_site = _site_with_port("cassegrain")
        offset = InstrumentOffset(dx=5.0, dy=3.0)
        rot = compute_focal_plane_rotation(
            45.0,
            site=cass_site,
            offset=offset,
            parallactic_angle=25.0,
        )
        # 0 * 45 + 0 + 25 = 25
        assert rot == pytest.approx(25.0)

    def test_all_three_components(self, site):
        """The three terms combine as ``nasmyth_sign * el + instrument_rotation + pa``."""
        # site is FYST with nasmyth_sign = +1
        offset = InstrumentOffset(dx=5.0, dy=3.0, instrument_rotation=15.0)
        el = 45.0
        pa = 20.0

        rot = compute_focal_plane_rotation(el, site=site, offset=offset, parallactic_angle=pa)
        # +1 * 45 + 15 + 20 = 80
        assert rot == pytest.approx(80.0)

    def test_left_nasmyth_all_components(self):
        left_site = _site_with_port("left")
        offset = InstrumentOffset(dx=5.0, dy=3.0, instrument_rotation=10.0)
        rot = compute_focal_plane_rotation(
            45.0,
            site=left_site,
            offset=offset,
            parallactic_angle=20.0,
        )
        # -1 * 45 + 10 + 20 = -15
        assert rot == pytest.approx(-15.0)

    def test_array_el_and_pa_broadcast_elementwise(self):
        site = _site_with_port("right")  # nasmyth_sign = +1
        offset = InstrumentOffset(dx=0.0, dy=0.0, instrument_rotation=10.0)
        el = np.array([20.0, 45.0, 70.0])
        pa = np.array([5.0, -3.0, 12.0])

        rot = compute_focal_plane_rotation(el, site=site, offset=offset, parallactic_angle=pa)

        assert isinstance(rot, np.ndarray)
        assert rot.shape == (3,)
        np.testing.assert_allclose(rot, site.nasmyth_sign * el + 10.0 + pa)
        # Concrete spot-check: +1*45 + 10 + (-3) = 52.
        assert rot[1] == pytest.approx(52.0)

    def test_site_and_offset_are_keyword_only(self, site):
        """Only ``el`` is positional, so ``site`` and ``offset`` cannot be swapped."""
        offset = InstrumentOffset(dx=0.0, dy=0.0)
        with pytest.raises(TypeError):
            compute_focal_plane_rotation(45.0, site, offset)
