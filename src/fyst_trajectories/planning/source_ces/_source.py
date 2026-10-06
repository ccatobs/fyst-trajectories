"""The source a source-CES pass tracks.

``_SourceSpec`` carries the six source keywords the public planners take: a
solar-system ``body``, or a fixed ``ra``/``dec`` with an optional proper motion
from ``ref_epoch``. It holds them as the caller gave them and places the source
on the sky; the kernel's input stage validates them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.time import Time

from ...coordinates import Coordinates


@dataclass(frozen=True)
class _SourceSpec:
    """A source-CES target: a solar-system body, or a fixed RA/Dec with optional proper motion.

    One object stands for the planners' six source keywords in every private
    signature. Strictly private; no public API guarantees on this class.

    Attributes
    ----------
    body : str or None
        Solar-system body name; ``None`` for an RA/Dec source.
    ra, dec : float or None
        Fixed source position in degrees.
    pm_ra, pm_dec : float
        Proper motion in mas/yr (``pm_ra`` includes the cos(dec) factor).
    ref_epoch : Time or None
        Reference epoch of ``ra``/``dec``, needed with a non-zero proper motion.
    """

    body: str | None = None
    ra: float | None = None
    dec: float | None = None
    pm_ra: float = 0.0
    pm_dec: float = 0.0
    ref_epoch: Time | None = None

    @property
    def label(self) -> str:
        """Human-readable source label for error messages and summaries."""
        if self.body is not None:
            return self.body.capitalize()
        return f"RA={self.ra:.3f}, Dec={self.dec:.3f}"

    def sample_altaz(self, coords: Coordinates, times: Time) -> tuple[np.ndarray, np.ndarray]:
        """Sample the source's (az, el) at the given times."""
        if self.body is not None:
            return coords.get_body_altaz(self.body, times)
        if self.pm_ra != 0.0 or self.pm_dec != 0.0:
            az, el = coords.radec_to_altaz_with_pm(
                ra=self.ra,
                dec=self.dec,
                pm_ra=self.pm_ra,
                pm_dec=self.pm_dec,
                ref_epoch=self.ref_epoch,
                obstime=times,
            )
            return np.asarray(az, dtype=float), np.asarray(el, dtype=float)
        return coords.radec_to_altaz(self.ra, self.dec, times)

    def radec_at(self, coords: Coordinates, obstime: Time) -> tuple[float, float]:
        """Return source RA/Dec at a single instant (for trajectory metadata)."""
        if self.body is not None:
            return coords.get_body_radec(self.body, obstime)
        return float(self.ra), float(self.dec)
