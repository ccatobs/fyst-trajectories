"""Independent parallactic-angle oracle shared by the offset and PA tests.

The reference brings the ICRS position to the apparent equinox of date (TETE)
before forming the hour angle, and takes nothing from
``Coordinates.get_parallactic_angle``, so it can serve as ground truth for it.
"""

import erfa
import numpy as np
from astropy import units as u
from astropy.coordinates import TETE, SkyCoord


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
