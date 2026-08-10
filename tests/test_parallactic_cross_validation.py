"""Cross-validate ``Coordinates.get_parallactic_angle`` against an apparent place.

The parallactic-angle *formula* is the IAU SOFA / ERFA ``hd2pa(ha, dec, phi)``
primitive (bundled with astropy as the ``erfa`` package). The subtlety this
file guards against is the input *frame*: the hour angle must be
referenced to the **apparent** equinox of date, the same equinox as the
local apparent sidereal time, not the catalogue ICRS/J2000 RA. Feeding
``erfa.hd2pa`` the library's own ``get_hour_angle`` (which forms
``apparent LST - ICRS RA``) would cancel the bug on both sides and mask it, so
the reference here transforms ICRS -> TETE(obstime) to obtain the apparent
RA/Dec *before* forming the hour angle. Agreement to < 0.01 deg, including at
epochs well past J2000 where the uncorrected precession term grows, validates
that ``get_parallactic_angle`` is referenced to the apparent celestial pole.

Points within 10 deg of the zenith are excluded: the parallactic angle is
genuinely ill-conditioned there (it swings through 180 deg at transit), so the
two finite transform chains can disagree by more than 0.01 deg for purely
numerical reasons unrelated to the frame question under test.
"""

import erfa
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import TETE, SkyCoord
from astropy.time import Time, TimeDelta


def _apparent_pa(coordinates, ra, dec, t):
    """Independent parallactic angle via apparent-place HA + ``erfa.hd2pa``.

    Brings the ICRS RA/Dec to the apparent equinox of date (TETE) before
    forming ``HA = LAST - RA_apparent``, then applies the IAU SOFA primitive.
    Works for scalar or array ``ra``/``dec``.
    """
    loc = coordinates.location
    lat_rad = np.deg2rad(coordinates.site.latitude)
    app = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").transform_to(
        TETE(obstime=t, location=loc)
    )
    last = t.sidereal_time("apparent", longitude=loc.lon).to_value(u.deg)
    ha = np.deg2rad(((last - app.ra.deg + 180.0) % 360.0) - 180.0)
    return np.rad2deg(erfa.hd2pa(ha, np.deg2rad(app.dec.deg), lat_rad))


def _pa_diff(a, b):
    """Smallest signed angular difference of two PAs, in degrees (abs)."""
    return np.abs(((a - b) + 180.0) % 360.0 - 180.0)


class TestParallacticAngleApparentPlace:
    """Compare ``get_parallactic_angle`` against the apparent-place reference."""

    @pytest.fixture
    def grid(self):
        """RA/Dec/time grid spanning typical FYST (southern-sky) geometry."""
        ras = np.array([0.0, 90.0, 180.0, 270.0, 359.5])
        decs = np.array([-65.0, -50.0, -35.0, -10.0, 5.0])
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

    def test_scalar_agreement_apparent_place(self, coordinates, grid):
        """Library PA matches the apparent-place reference at every observable point."""
        ras, decs, times = grid
        max_pa = 0.0
        n_checked = 0
        for ra in ras:
            for dec in decs:
                for t in times:
                    _, el = coordinates.radec_to_altaz(ra, dec, obstime=t)
                    if el < 20.0 or el > 80.0:  # horizon / near-zenith singularity
                        continue
                    pa_lib = coordinates.get_parallactic_angle(ra, dec, obstime=t)
                    pa_ref = _apparent_pa(coordinates, ra, dec, t)
                    diff = _pa_diff(pa_lib, pa_ref)
                    max_pa = max(max_pa, abs(pa_lib))
                    n_checked += 1
                    assert diff < 0.01, (
                        f"PA mismatch at RA={ra}, dec={dec}, t={t.iso}: "
                        f"lib={pa_lib:.4f}, ref={pa_ref:.4f}, diff={diff:.4f}"
                    )
        assert n_checked > 10  # the grid really exercised the comparison
        assert max_pa > 0.0  # not comparing zero against zero

    def test_vectorised_agreement_apparent_place(self, coordinates):
        """Vectorised library call matches the vectorised apparent-place reference."""
        n = 60
        rng = np.random.default_rng(seed=42)
        ras = rng.uniform(0.0, 360.0, size=n)
        decs = rng.uniform(-75.0, -5.0, size=n)
        times = Time("2026-06-15T08:00:00", scale="utc") + TimeDelta(np.arange(n) * 60.0 * u.s)

        _, el = coordinates.radec_to_altaz(ras, decs, obstime=times)
        mask = (el >= 20.0) & (el <= 80.0)
        assert mask.sum() > 10

        pa_lib = coordinates.get_parallactic_angle(ras, decs, obstime=times)
        pa_ref = _apparent_pa(coordinates, ras, decs, times)
        diff = _pa_diff(pa_lib, pa_ref)[mask]
        assert diff.max() < 0.01, f"max diff = {diff.max():.4f} deg"

    def test_agreement_across_epochs(self, coordinates):
        """Regression: agreement holds across 2026-2035 as precession accumulates.

        Forming the hour angle as ``apparent LST - ICRS RA`` mixes frames, leaving an
        uncorrected precession term that is ~0 at J2000 and grows ~0.018 deg/yr.
        Referencing the PA to the apparent pole keeps the disagreement < 0.01 deg
        at every epoch; the buggy form misses by several tenths of a degree and
        worsens with epoch.
        """
        rng = np.random.default_rng(seed=7)
        epochs = Time(
            [
                "2026-06-15T08:00:00",
                "2029-06-15T08:00:00",
                "2032-06-15T08:00:00",
                "2035-06-15T08:00:00",
            ],
            scale="utc",
        )
        n_epochs_checked = 0
        for t in epochs:
            ras = rng.uniform(0.0, 360.0, size=80)
            decs = rng.uniform(-75.0, -5.0, size=80)
            _, el = coordinates.radec_to_altaz(ras, decs, obstime=t)
            mask = (el >= 20.0) & (el <= 80.0)
            if mask.sum() < 5:
                continue
            n_epochs_checked += 1
            pa_lib = coordinates.get_parallactic_angle(ras, decs, obstime=t)
            pa_ref = _apparent_pa(coordinates, ras, decs, t)
            diff = _pa_diff(pa_lib, pa_ref)[mask]
            assert diff.max() < 0.01, (
                f"epoch {t.iso}: max PA diff = {diff.max():.4f} deg (frame-mix regression)"
            )
        assert n_epochs_checked >= 1  # guard against a vacuous pass
