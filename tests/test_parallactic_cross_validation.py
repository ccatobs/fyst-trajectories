"""Cross-validation of ``Coordinates.get_parallactic_angle`` against ERFA.

The library's parallactic-angle formula is unit-tested only against
itself. This file adds an independent reference: the IAU SOFA / ERFA
``hd2pa(ha, dec, phi)`` primitive (bundled with astropy as the ``erfa``
package). Agreement to better than 0.01° validates the
``arctan2(sin H, cos δ tan φ − sin δ cos H)`` implementation in
``coordinates.py``.
"""

import erfa
import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time, TimeDelta


class TestParallacticAngleCrossValidation:
    """Compare ``get_parallactic_angle`` against ``erfa.hd2pa``."""

    @pytest.fixture
    def grid(self, coordinates):
        """RA/Dec/time grid spanning typical FYST observing geometry."""
        ras = np.array([0.0, 90.0, 180.0, 270.0, 359.5])
        decs = np.array([-60.0, -30.0, 0.0, 30.0, 60.0])
        times = Time(
            [
                "2026-01-15T04:00:00",
                "2026-03-15T04:00:00",
                "2026-06-15T08:00:00",
                "2026-09-15T16:00:00",
            ],
            scale="utc",
        )
        return ras, decs, times

    def test_scalar_agreement_with_erfa(self, coordinates, grid):
        """At every grid point, the library result agrees with ``erfa.hd2pa``."""
        ras, decs, times = grid
        lat_rad = np.deg2rad(coordinates.site.latitude)
        max_pa = 0.0
        for ra in ras:
            for dec in decs:
                for t in times:
                    pa_lib = coordinates.get_parallactic_angle(ra, dec, obstime=t)
                    ha = coordinates.get_hour_angle(ra, t)
                    pa_ref = float(np.rad2deg(erfa.hd2pa(np.deg2rad(ha), np.deg2rad(dec), lat_rad)))
                    diff = abs(((pa_lib - pa_ref) + 180.0) % 360.0 - 180.0)
                    max_pa = max(max_pa, abs(pa_lib))
                    assert diff < 0.01, (
                        f"PA mismatch at RA={ra}, dec={dec}, t={t.iso}: "
                        f"lib={pa_lib:.4f}, erfa={pa_ref:.4f}, diff={diff:.4f}"
                    )
        # Sanity: at least some grid points have nonzero PA so we know
        # the test isn't comparing zero against zero.
        assert max_pa > 0.0

    def test_vectorised_agreement_with_erfa(self, coordinates):
        """Vectorised library call matches a vectorised ERFA reference."""
        n = 50
        rng = np.random.default_rng(seed=42)
        ras = rng.uniform(0.0, 360.0, size=n)
        decs = rng.uniform(-80.0, 80.0, size=n)
        times = Time("2026-06-15T08:00:00", scale="utc") + TimeDelta(np.arange(n) * 60.0 * u.s)

        pa_lib = coordinates.get_parallactic_angle(ras, decs, obstime=times)
        ha_arr = coordinates.get_hour_angle(ras, times)
        lat_rad = np.deg2rad(coordinates.site.latitude)
        pa_ref = np.rad2deg(erfa.hd2pa(np.deg2rad(ha_arr), np.deg2rad(decs), lat_rad))

        diff = np.abs(((pa_lib - pa_ref) + 180.0) % 360.0 - 180.0)
        assert diff.max() < 0.01, (
            f"Vectorised PA mismatch: max diff = {diff.max():.4f}° at index {diff.argmax()}"
        )
