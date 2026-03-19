"""Cross-validation tests between fyst-trajectories and KOSMA focal plane model.

This module compares the fyst-trajectories spherical offset model against the
KOSMA flat-projection focal plane model extracted from
``tests_with_focalplane.py``. The KOSMA model uses flat-plane trigonometry
(2D rotation + linear mm-to-arcsec conversion), while fyst-trajectories uses
exact spherical trigonometry (great-circle offsets).

The two models should agree closely for small offsets (< 1 degree) and
diverge for larger offsets where the flat-earth approximation breaks down.

Key KOSMA constants extracted from tests_with_focalplane.py:
- plate_scale: 13.89 arcsec/mm
- focal_length: (180 * 3600) / (plate_scale * pi) + correction
- Port rotation: +el (Right Nasmyth), -el (Left), flange_rot (Cassegrain)
- Total rotation: rho = port_rotation + instr_focal_plane_rotation
- Offset projection: flat 2D rotation by rho, then mm_to_arcsec conversion
"""

import math

import pytest
from astropy.time import Time

from fyst_trajectories.coordinates import Coordinates
from fyst_trajectories.offsets import (
    InstrumentOffset,
    boresight_to_detector,
    compute_focal_plane_rotation,
)
from fyst_trajectories.site import (
    AxisLimits,
    Site,
    SunAvoidanceConfig,
    TelescopeLimits,
)

# ---------------------------------------------------------------------------
# KOSMA model constants (from tests_with_focalplane.py)
# ---------------------------------------------------------------------------

KOSMA_PLATE_SCALE = 13.89  # arcsec/mm — intentionally independent of site.plate_scale
"""FYST plate scale as hardcoded in the KOSMA control system.

This is defined independently from ``site.plate_scale`` so these cross-validation
tests compare against the KOSMA model's own constants, not our config values.
"""

KOSMA_FOCAL_LENGTH_CORRECTION = 0.0  # mm (default)
"""Focal length correction for the instrument (default = 0)."""

KOSMA_FOCAL_LENGTH = (180.0 * 3600.0) / (
    KOSMA_PLATE_SCALE * math.pi
) + KOSMA_FOCAL_LENGTH_CORRECTION
"""Effective focal length in mm, derived from plate scale."""


def kosma_mm_to_arcsec(mm: float) -> float:
    """Convert mm in focal plane to arcsec on sky (KOSMA formula).

    This is the exact formula from tests_with_focalplane.py:
        arcsec = mm / (focal_length * pi) * 180 * 3600

    Parameters
    ----------
    mm : float
        Position in focal plane in millimeters.

    Returns
    -------
    float
        Angular offset in arcseconds.
    """
    return mm / (KOSMA_FOCAL_LENGTH * math.pi) * 180.0 * 3600.0


def kosma_focal_plane_offset(
    ref_x_mm: float,
    ref_y_mm: float,
    elevation: float,
    port: str = "Right",
    instr_focal_plane_rotation: float = 0.0,
) -> tuple[float, float]:
    """Compute focal plane offset using the KOSMA flat-projection model.

    This implements the core rotation logic from compute_focal_plane() in
    tests_with_focalplane.py, simplified to the reference position rotation
    case (zero boresight/elevation axis offsets).

    Parameters
    ----------
    ref_x_mm : float
        Reference position X in mm (focal plane).
    ref_y_mm : float
        Reference position Y in mm (focal plane).
    elevation : float
        Telescope elevation in degrees.
    port : str
        Nasmyth port: "Right", "Left", or "Cassegrain".
    instr_focal_plane_rotation : float
        Instrument focal plane rotation in degrees (alpha_p in KOSMA).

    Returns
    -------
    fp_x_arcsec : float
        Focal plane X offset in arcseconds.
    fp_y_arcsec : float
        Focal plane Y offset in arcseconds.
    """
    # Port-dependent rotation (angle_if in KOSMA code)
    if port.startswith("Left"):
        angle_if = -elevation
    elif port.startswith("Right"):
        angle_if = elevation
    elif port.startswith("Cass"):
        angle_if = 0.0  # Cassegrain uses flange_rotation, default 0
    else:
        angle_if = 0.0

    # Total geometric rotation rho
    rho = angle_if + instr_focal_plane_rotation

    # Convert reference positions mm -> arcsec
    ref_x_arcsec = kosma_mm_to_arcsec(ref_x_mm)
    ref_y_arcsec = kosma_mm_to_arcsec(ref_y_mm)

    # 2D rotation by rho (flat projection)
    rho_rad = math.radians(rho)
    fp_x = ref_x_arcsec * math.cos(rho_rad) - ref_y_arcsec * math.sin(rho_rad)
    fp_y = ref_x_arcsec * math.sin(rho_rad) + ref_y_arcsec * math.cos(rho_rad)

    return fp_x, fp_y


# ---------------------------------------------------------------------------
# Helper to create test sites
# ---------------------------------------------------------------------------


def _make_site(nasmyth_port: str = "right") -> Site:
    """Create a minimal test site."""
    return Site(
        name="TestFYST",
        description="",
        latitude=-22.985639,
        longitude=-67.740278,
        elevation=5611.8,
        atmosphere=None,
        telescope_limits=TelescopeLimits(
            azimuth=AxisLimits(
                min=-270,
                max=270,
                max_velocity=3,
                max_acceleration=1,
            ),
            elevation=AxisLimits(
                min=20,
                max=90,
                max_velocity=1,
                max_acceleration=0.5,
            ),
        ),
        sun_avoidance=SunAvoidanceConfig(
            enabled=True,
            exclusion_radius=45,
            warning_radius=50,
        ),
        nasmyth_port=nasmyth_port,
    )


# ---------------------------------------------------------------------------
# Cross-validation tests
# ---------------------------------------------------------------------------


class TestKOSMACrossValidationRotation:
    """Cross-validate rotation angle computation between models.

    Both models should produce identical rotation angles since neither
    involves projection approximations, it is purely additive.
    """

    @pytest.mark.parametrize("el", [20.0, 45.0, 60.0, 85.0])
    def test_right_nasmyth_rotation_matches(self, el):
        """Test that rotation = +1*el + inst_rot matches KOSMA rho."""
        inst_rot = 10.0
        site = _make_site("right")
        offset = InstrumentOffset(dx=0.0, dy=0.0, instrument_rotation=inst_rot)

        ccat_rot = compute_focal_plane_rotation(el, site, offset)

        # KOSMA: rho = angle_if + instr_focal_plane_rotation
        # For Right Nasmyth: angle_if = +el
        kosma_rho = el + inst_rot

        assert ccat_rot == pytest.approx(kosma_rho)

    @pytest.mark.parametrize("el", [20.0, 45.0, 60.0, 85.0])
    def test_left_nasmyth_rotation_matches(self, el):
        """Test that rotation = -1*el + inst_rot matches KOSMA rho."""
        inst_rot = 10.0
        site = _make_site("left")
        offset = InstrumentOffset(dx=0.0, dy=0.0, instrument_rotation=inst_rot)

        ccat_rot = compute_focal_plane_rotation(el, site, offset)

        # KOSMA: angle_if = -el for Left Nasmyth
        kosma_rho = -el + inst_rot

        assert ccat_rot == pytest.approx(kosma_rho)

    @pytest.mark.parametrize("el", [20.0, 45.0, 60.0, 85.0])
    def test_cassegrain_rotation_matches(self, el):
        """Test that cassegrain rotation = inst_rot matches KOSMA rho."""
        inst_rot = 10.0
        site = _make_site("cassegrain")
        offset = InstrumentOffset(dx=0.0, dy=0.0, instrument_rotation=inst_rot)

        ccat_rot = compute_focal_plane_rotation(el, site, offset)

        # KOSMA: angle_if = 0 for Cassegrain (flange_rotation=0)
        kosma_rho = 0.0 + inst_rot

        assert ccat_rot == pytest.approx(kosma_rho)


class TestKOSMAParallacticAngleRotation:
    """Cross-validate rotation with non-zero parallactic angle.

    KOSMA's full rotation: rho = nasmyth_sign * el + instrument_rotation + tel_angle_focal_plane
    fyst-trajectories's:       rho = nasmyth_sign * el + instrument_rotation + parallactic_angle

    Where tel_angle_focal_plane in KOSMA corresponds to the parallactic angle.
    Since the rotation is pure addition, the two should match exactly.
    """

    # (RA, Dec) test cases spanning different hour angles and declinations.
    # Chosen to produce a range of parallactic angles at the FYST site.
    _CELESTIAL_CASES = [
        # (ra, dec, description)
        (180.0, -23.0, "near transit at FYST latitude"),
        (90.0, -23.0, "far east of meridian"),
        (270.0, -23.0, "far west of meridian"),
        (180.0, -60.0, "high-dec source near transit"),
        (120.0, 10.0, "northern source east of meridian"),
    ]

    _OBSTIME = Time("2026-03-15T04:00:00", scale="utc")

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "ra,dec,description",
        _CELESTIAL_CASES,
        ids=[c[2] for c in _CELESTIAL_CASES],
    )
    def test_rotation_formula_with_parallactic_angle(self, ra, dec, description):
        """Total rotation with PA matches nasmyth_sign*el + inst_rot + pa."""
        inst_rot = 10.0
        site = _make_site("right")
        coords = Coordinates(site)

        _, el = coords.radec_to_altaz(ra, dec, self._OBSTIME)
        pa = coords.get_parallactic_angle(ra, dec, self._OBSTIME)

        offset = InstrumentOffset(dx=0.0, dy=0.0, instrument_rotation=inst_rot)
        ccat_rot = compute_focal_plane_rotation(el, site, offset, parallactic_angle=pa)

        # KOSMA: rho = angle_if + instr_focal_plane_rotation + tel_angle_focal_plane
        # For Right Nasmyth: angle_if = +el, tel_angle_focal_plane = pa
        kosma_rho = el + inst_rot + pa

        assert ccat_rot == pytest.approx(kosma_rho)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "ra,dec,description",
        _CELESTIAL_CASES,
        ids=[c[2] for c in _CELESTIAL_CASES],
    )
    def test_rotation_with_nonzero_offset(self, ra, dec, description):
        """Rotation matches KOSMA formula even with non-zero focal plane offset."""
        inst_rot = 5.0
        site = _make_site("right")
        coords = Coordinates(site)

        _, el = coords.radec_to_altaz(ra, dec, self._OBSTIME)
        pa = coords.get_parallactic_angle(ra, dec, self._OBSTIME)

        offset = InstrumentOffset.from_focal_plane(
            x_mm=5.0,
            y_mm=3.0,
            plate_scale=KOSMA_PLATE_SCALE,
            instrument_rotation=inst_rot,
        )

        ccat_rot = compute_focal_plane_rotation(
            el,
            site,
            offset,
            parallactic_angle=pa,
        )
        kosma_rot = site.nasmyth_sign * el + inst_rot + pa

        assert ccat_rot == pytest.approx(kosma_rot)

    @pytest.mark.slow
    def test_parallactic_angle_near_zero_at_transit(self):
        """Parallactic angle is near zero at transit (HA ~ 0)."""
        site = _make_site("right")
        coords = Coordinates(site)

        # Use a Dec near the site latitude so it transits near zenith.
        # Find the RA that transits at the observation time.
        lst = coords.get_lst(self._OBSTIME)
        ra_at_transit = lst  # HA = LST - RA = 0 when RA = LST
        dec = site.latitude  # transits near zenith

        pa = coords.get_parallactic_angle(ra_at_transit, dec, self._OBSTIME)

        # At transit, parallactic angle should be very close to zero
        assert pa == pytest.approx(0.0, abs=1.0)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "port,expected_sign",
        [
            ("right", +1),
            ("left", -1),
        ],
    )
    def test_parallactic_angle_with_both_ports(self, port, expected_sign):
        """PA contribution is port-independent; port only affects el sign."""
        inst_rot = 7.0
        site = _make_site(port)
        coords = Coordinates(site)

        ra, dec = 120.0, -30.0
        _, el = coords.radec_to_altaz(ra, dec, self._OBSTIME)
        pa = coords.get_parallactic_angle(ra, dec, self._OBSTIME)

        offset = InstrumentOffset(dx=0.0, dy=0.0, instrument_rotation=inst_rot)
        ccat_rot = compute_focal_plane_rotation(el, site, offset, parallactic_angle=pa)

        kosma_rho = expected_sign * el + inst_rot + pa

        assert ccat_rot == pytest.approx(kosma_rho)


class TestKOSMAElevationDependentOffsets:
    """Cross-validate small offset projection across elevations 20-85 deg.

    At high elevations (el > 80), 1/cos(el) amplifies azimuth residuals,
    so we compare cross-elevation offsets (dAz * cos(el)) rather than dAz.
    """

    @pytest.mark.parametrize("el", [20.0, 30.0, 45.0, 60.0, 75.0, 85.0])
    @pytest.mark.parametrize(
        "ref_x_mm,ref_y_mm",
        [
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (5.0, 3.0),
            (10.0, -5.0),
        ],
    )
    def test_small_offset_at_elevation(self, el, ref_x_mm, ref_y_mm):
        """Small offsets agree between KOSMA flat and ccat spherical."""
        site = _make_site("right")

        kosma_x, kosma_y = kosma_focal_plane_offset(
            ref_x_mm,
            ref_y_mm,
            el,
            port="Right",
        )

        offset = InstrumentOffset.from_focal_plane(
            x_mm=ref_x_mm,
            y_mm=ref_y_mm,
            plate_scale=KOSMA_PLATE_SCALE,
        )
        field_rotation = compute_focal_plane_rotation(el, site, offset)
        det_az, det_el = boresight_to_detector(
            180.0,
            el,
            offset,
            field_rotation=field_rotation,
        )

        cos_el = math.cos(math.radians(el))
        ccat_xel_arcsec = (det_az - 180.0) * cos_el * 3600.0
        ccat_del_arcsec = (det_el - el) * 3600.0

        # Flat-projection error scales as ~offset^3 and grows with
        # 1/cos(el); at ~155 arcsec and el=85 it can reach ~0.6 arcsec.
        offset_arcsec = math.sqrt(
            kosma_mm_to_arcsec(ref_x_mm) ** 2 + kosma_mm_to_arcsec(ref_y_mm) ** 2
        )
        if el >= 80.0:
            tol = max(0.2, offset_arcsec * 0.005)
        elif el >= 70.0:
            tol = max(0.1, offset_arcsec * 0.002)
        else:
            tol = 0.1

        assert ccat_xel_arcsec == pytest.approx(kosma_x, abs=tol)
        assert ccat_del_arcsec == pytest.approx(kosma_y, abs=tol)


class TestKOSMACrossValidationSmallOffsets:
    """Cross-validate offset projection for small offsets.

    For small offsets (< ~0.5 degrees), the KOSMA flat-projection and
    the fyst-trajectories spherical model should agree to within a few
    arcseconds. The flat-plane error scales as offset^3.
    """

    @pytest.mark.parametrize(
        "ref_x_mm,ref_y_mm",
        [
            (1.0, 0.0),  # ~14 arcsec along x
            (0.0, 1.0),  # ~14 arcsec along y
            (1.0, 1.0),  # ~20 arcsec diagonal
            (5.0, 3.0),  # ~80 arcsec, still small
            (10.0, -5.0),  # ~155 arcsec
        ],
    )
    def test_small_offset_agreement(self, ref_x_mm, ref_y_mm):
        """Test that small offsets agree between KOSMA flat and ccat spherical.

        For offsets under ~3 arcmin, the flat-projection error is
        negligible (< 0.01 arcsec).
        """
        elevation = 45.0
        site = _make_site("right")

        # KOSMA model: flat projection
        kosma_x, kosma_y = kosma_focal_plane_offset(
            ref_x_mm,
            ref_y_mm,
            elevation,
            port="Right",
        )

        # fyst-trajectories model: spherical projection
        # Use from_focal_plane to convert mm -> arcmin via plate scale
        offset = InstrumentOffset.from_focal_plane(
            x_mm=ref_x_mm,
            y_mm=ref_y_mm,
            plate_scale=KOSMA_PLATE_SCALE,
        )

        # Compute field rotation (mechanical only, no parallactic angle)
        field_rotation = compute_focal_plane_rotation(
            elevation,
            site,
            offset,
        )

        # Apply offset using spherical model
        det_az, det_el = boresight_to_detector(
            180.0,
            elevation,
            offset,
            field_rotation=field_rotation,
        )

        # Convert spherical result to offsets in arcsec for comparison.
        # KOSMA fp_x is cross-elevation = dAz * cos(el), fp_y is dEl.
        # The spherical model returns (dAz, dEl) in degrees.
        cos_el = math.cos(math.radians(elevation))
        ccat_xel_arcsec = (det_az - 180.0) * cos_el * 3600.0
        ccat_del_arcsec = (det_el - elevation) * 3600.0

        # For these small offsets, agreement should be within 0.1 arcsec
        assert ccat_xel_arcsec == pytest.approx(kosma_x, abs=0.1)
        assert ccat_del_arcsec == pytest.approx(kosma_y, abs=0.1)


class TestKOSMACrossValidationLargeOffsets:
    """Cross-validate offset projection for large offsets.

    For PrimeCam-scale offsets (~1.78 degrees = 106.8 arcmin), the flat
    projection error becomes significant. This test documents where the
    two models diverge and by how much.

    The flat-plane error for a great-circle offset of angular distance
    rho scales as ~rho^3/6 for the leading-order term. At 1.78 degrees
    (0.031 rad), the error is ~5e-6 rad = ~1 arcsec. At 5 degrees
    (0.087 rad), the error is ~1.1e-4 rad = ~23 arcsec.
    """

    @pytest.mark.parametrize(
        "offset_mm,max_diff_arcsec,description",
        [
            # Offset in mm on the focal plane. Converted to arcmin via plate scale.
            # Measured differences at el=45: 0.98, 35.6, 113.9, 937.6 arcsec
            # Bounds are set at ~2x measured to allow margin
            (10.0 * 60.0 / KOSMA_PLATE_SCALE, 2.0, "small module offset (~10 arcmin)"),
            (60.0 * 60.0 / KOSMA_PLATE_SCALE, 72.0, "1-degree offset"),
            (461.3, 230.0, "PrimeCam inner ring (~1.78 deg, 461.3 mm)"),
            (300.0 * 60.0 / KOSMA_PLATE_SCALE, 1900.0, "5-degree offset (extreme)"),
        ],
    )
    def test_flat_vs_spherical_divergence(
        self,
        offset_mm,
        max_diff_arcsec,
        description,
    ):
        """Verify that flat-vs-spherical difference scales with offset size.

        Parameters
        ----------
        offset_mm : float
            Offset magnitude in millimeters on the focal plane.
        max_diff_arcsec : float
            Maximum acceptable difference in arcseconds. This is an upper
            bound on the flat-projection error for the given offset size.
        description : str
            Human-readable description of the test case.
        """
        elevation = 45.0
        site = _make_site("right")

        # KOSMA flat-projection result
        kosma_x, kosma_y = kosma_focal_plane_offset(
            offset_mm,
            0.0,
            elevation,
            port="Right",
        )

        # fyst-trajectories spherical result using from_focal_plane
        offset = InstrumentOffset.from_focal_plane(
            x_mm=offset_mm,
            y_mm=0.0,
            plate_scale=KOSMA_PLATE_SCALE,
        )
        field_rotation = compute_focal_plane_rotation(elevation, site, offset)
        det_az, det_el = boresight_to_detector(
            180.0,
            elevation,
            offset,
            field_rotation=field_rotation,
        )

        # KOSMA fp_x is cross-elevation = dAz * cos(el)
        cos_el = math.cos(math.radians(elevation))
        ccat_xel_arcsec = (det_az - 180.0) * cos_el * 3600.0
        ccat_del_arcsec = (det_el - elevation) * 3600.0

        # Compute difference between the two models
        diff_x = abs(ccat_xel_arcsec - kosma_x)
        diff_y = abs(ccat_del_arcsec - kosma_y)
        diff_total = math.sqrt(diff_x**2 + diff_y**2)

        # The difference should be within the expected bound
        assert diff_total < max_diff_arcsec, (
            f"Flat-vs-spherical difference for {description}: "
            f"{diff_total:.2f} arcsec exceeds bound {max_diff_arcsec} arcsec"
        )

    def test_divergence_increases_with_offset(self):
        """Verify that the flat-vs-spherical difference increases with offset.

        The error from flat-plane approximation grows with offset^3,
        so larger offsets should produce larger discrepancies.
        """
        elevation = 45.0
        site = _make_site("right")

        diffs = []
        # Focal plane positions in mm, increasing in size
        offset_mm_sizes = [
            10.0 * 60.0 / KOSMA_PLATE_SCALE,  # ~10 arcmin
            60.0 * 60.0 / KOSMA_PLATE_SCALE,  # ~60 arcmin
            461.3,  # ~106.8 arcmin (inner ring)
            180.0 * 60.0 / KOSMA_PLATE_SCALE,  # ~180 arcmin
            300.0 * 60.0 / KOSMA_PLATE_SCALE,  # ~300 arcmin
        ]

        for offset_mm in offset_mm_sizes:
            kosma_x, kosma_y = kosma_focal_plane_offset(
                offset_mm,
                0.0,
                elevation,
                port="Right",
            )

            offset = InstrumentOffset.from_focal_plane(
                x_mm=offset_mm,
                y_mm=0.0,
                plate_scale=KOSMA_PLATE_SCALE,
            )
            field_rotation = compute_focal_plane_rotation(
                elevation,
                site,
                offset,
            )
            det_az, det_el = boresight_to_detector(
                180.0,
                elevation,
                offset,
                field_rotation=field_rotation,
            )

            # KOSMA fp_x is cross-elevation = dAz * cos(el)
            cos_el = math.cos(math.radians(elevation))
            ccat_xel = (det_az - 180.0) * cos_el * 3600.0
            ccat_del = (det_el - elevation) * 3600.0

            diff = math.sqrt((ccat_xel - kosma_x) ** 2 + (ccat_del - kosma_y) ** 2)
            diffs.append(diff)

        # Each subsequent offset should produce a larger difference
        for i in range(len(diffs) - 1):
            assert diffs[i + 1] > diffs[i], (
                f"Expected divergence to increase: "
                f"offset_mm={offset_mm_sizes[i + 1]:.1f} diff={diffs[i + 1]:.4f} <= "
                f"offset_mm={offset_mm_sizes[i]:.1f} diff={diffs[i]:.4f}"
            )


class TestKOSMAPlateScaleConsistency:
    """Verify plate scale and focal length relationships."""

    def test_plate_scale_to_focal_length(self):
        """Test the KOSMA plate_scale -> focal_length conversion."""
        expected_fl = (180.0 * 3600.0) / (KOSMA_PLATE_SCALE * math.pi)
        assert KOSMA_FOCAL_LENGTH == pytest.approx(expected_fl)

    def test_mm_to_arcsec_roundtrip(self):
        """Test that mm_to_arcsec is consistent with plate_scale."""
        # 1mm at the focal plane should be plate_scale arcsec on sky
        one_mm_arcsec = kosma_mm_to_arcsec(1.0)
        assert one_mm_arcsec == pytest.approx(KOSMA_PLATE_SCALE, rel=1e-10)

    def test_focal_length_reasonable(self):
        """Test that derived focal length is physically reasonable for FYST."""
        # FYST is a 6m telescope with f/0.6 primary + reimaging.
        # Effective focal length depends on optical design.
        # The plate scale of 13.89"/mm implies f ~ 14.8m effective.
        assert 10_000 < KOSMA_FOCAL_LENGTH < 20_000  # mm
