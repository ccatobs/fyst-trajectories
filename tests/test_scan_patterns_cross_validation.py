"""Cross-validation tests comparing fyst-trajectories against scan_patterns.

This module validates that scan pattern implementations in fyst-trajectories
produce the same results as the legacy ``scan_patterns`` package (imported
as ``scanning``).

Two levels of comparison are tested:

**Level 1 -- Sky pattern offsets (pure math):**
Both packages implement the same Fourier expansion (Pong) and CV Daisy
algorithms. For Pong (a closed-form Fourier series), results match to
machine precision (~1e-10 deg). For Daisy (an iterative simulation),
the unit-scale difference (scan_patterns uses arcseconds internally,
fyst-trajectories uses degrees) causes floating-point divergence that
compounds over thousands of steps. The standard Daisy case agrees
within ~0.5 milliarcsec; the Ra=0 case is numerically degenerate and
shows larger divergence (see class docstring for details).

**Level 2 -- AltAz trajectories:**
After coordinate conversion, results should agree within ~0.2 deg. The
tolerance is larger because the packages use different astronomy backends
(manual spherical trig vs astropy with refraction and IERS corrections).

Key parameter mapping between the two packages:

=================  ======================
scan_patterns       fyst-trajectories
=================  ======================
``num_term``        ``num_terms``
``start_acc``       ``start_acceleration``
``R0``              ``radius``
``Rt``              ``turn_radius``
``Ra``              ``avoidance_radius``
``sample_interval`` ``timestep``
=================  ======================
"""

import math

import numpy as np
import pytest
from astropy.time import Time

try:
    from scanning.coordinates import Daisy, Pong, TelescopePattern

    SCANNING_AVAILABLE = True
except ImportError:
    SCANNING_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not SCANNING_AVAILABLE,
    reason="scan_patterns not installed. Install from scan_patterns/ directory.",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pong_offsets_scanning(
    num_term: int,
    width: float,
    height: float,
    spacing: float,
    velocity: float,
    sample_interval: float,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Pong x/y offsets using the scan_patterns algorithm at specified times.

    Replicates the vertex computation from scan_patterns' Pong class and
    evaluates the Fourier expansion at the given *times* so that both
    packages are sampled at identical instants.

    Parameters
    ----------
    num_term : int
        Number of Fourier terms.
    width : float
        Scan region width in degrees.
    height : float
        Scan region height in degrees.
    spacing : float
        Space between scan lines in degrees.
    velocity : float
        Scan velocity in degrees/second.
    sample_interval : float
        Time between samples in seconds.
    times : np.ndarray
        Time values at which to evaluate.

    Returns
    -------
    x : np.ndarray
        X offsets in degrees.
    y : np.ndarray
        Y offsets in degrees.
    """
    # Compute internal parameters for direct Fourier evaluation
    vert_spacing = math.sqrt(2) * spacing
    vavg = velocity / math.sqrt(2)

    x_numvert = math.ceil(width / vert_spacing)
    y_numvert = math.ceil(height / vert_spacing)

    if x_numvert % 2 == y_numvert % 2:
        if x_numvert >= y_numvert:
            y_numvert += 1
        else:
            x_numvert += 1

    num_vert = [x_numvert, y_numvert]
    most_i = num_vert.index(max(x_numvert, y_numvert))
    while math.gcd(num_vert[0], num_vert[1]) != 1:
        num_vert[most_i] += 2
    x_numvert, y_numvert = num_vert

    peri_x = x_numvert * vert_spacing * 2 / vavg
    peri_y = y_numvert * vert_spacing * 2 / vavg
    amp_x = x_numvert * vert_spacing / 2
    amp_y = y_numvert * vert_spacing / 2

    # Evaluate the Fourier expansion at each requested time.
    # This is the triangle-wave Fourier series from the SCUBA paper,
    # previously accessed via Pong._fourier_expansion (removed in Phase 6).
    def _fourier_expansion(n_term, amp, t, peri):
        N = n_term * 2 - 1
        a = (8 * amp) / (math.pi**2)
        b = 2 * math.pi / peri
        pos = 0.0
        for n in range(1, N + 1, 2):
            c = math.pow(-1, (n - 1) / 2) / n**2
            pos += c * math.sin(b * n * t)
        return pos * a

    x = np.array([_fourier_expansion(num_term, amp_x, t, peri_x) for t in times])
    y = np.array([_fourier_expansion(num_term, amp_y, t, peri_y) for t in times])
    return x, y


def _pong_offsets_ccat(
    num_terms: int,
    width: float,
    height: float,
    spacing: float,
    velocity: float,
    timestep: float,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Pong x/y offsets using fyst-trajectories at specified times.

    Instantiates a PongScanPattern to access the internal Fourier method
    and vertex computation, then evaluates at the given *times*.

    Parameters
    ----------
    num_terms : int
        Number of Fourier terms.
    width : float
        Scan region width in degrees.
    height : float
        Scan region height in degrees.
    spacing : float
        Space between scan lines in degrees.
    velocity : float
        Scan velocity in degrees/second.
    timestep : float
        Time between samples in seconds.
    times : np.ndarray
        Time values at which to evaluate.

    Returns
    -------
    x : np.ndarray
        X offsets in degrees.
    y : np.ndarray
        Y offsets in degrees.
    """
    from fyst_trajectories.patterns.configs import PongScanConfig
    from fyst_trajectories.patterns.pong import PongScanPattern

    config = PongScanConfig(
        timestep=timestep,
        width=width,
        height=height,
        spacing=spacing,
        velocity=velocity,
        num_terms=num_terms,
        angle=0.0,
    )
    pattern = PongScanPattern(ra=0.0, dec=0.0, config=config)

    x_numvert, y_numvert, amp_x, amp_y = pattern._compute_vertices()
    vert_spacing = math.sqrt(2) * spacing
    vavg = velocity / math.sqrt(2)
    peri_x = x_numvert * vert_spacing * 2 / vavg
    peri_y = y_numvert * vert_spacing * 2 / vavg

    x = pattern._fourier_triangle_wave(num_terms, amp_x, times, peri_x)
    y = pattern._fourier_triangle_wave(num_terms, amp_y, times, peri_y)
    return x, y


def _daisy_offsets_scanning(
    velocity: float,
    start_acc: float,
    R0: float,
    Rt: float,
    Ra: float,
    T: float,
    sample_interval: float,
    y_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate Daisy offsets using scan_patterns.

    Parameters
    ----------
    velocity : float
        Scan velocity in degrees/second.
    start_acc : float
        Start acceleration in degrees/second^2.
    R0 : float
        Characteristic radius in degrees.
    Rt : float
        Turn radius in degrees.
    Ra : float
        Avoidance radius in degrees.
    T : float
        Total duration in seconds.
    sample_interval : float
        Time between samples in seconds.
    y_offset : float, optional
        Initial y offset in degrees. Default is 0.0.

    Returns
    -------
    times : np.ndarray
        Time offsets in seconds.
    x : np.ndarray
        X offsets in degrees.
    y : np.ndarray
        Y offsets in degrees.
    """
    daisy = Daisy(
        velocity=velocity,
        start_acc=start_acc,
        R0=R0,
        Rt=Rt,
        Ra=Ra,
        T=T,
        sample_interval=sample_interval,
        y_offset=y_offset,
    )
    data = daisy.save_data()
    times = np.array(data["time_offset"])
    x = np.array(data["x_coord"])
    y = np.array(data["y_coord"])
    return times, x, y


def _daisy_offsets_ccat(
    velocity: float,
    start_acceleration: float,
    radius: float,
    turn_radius: float,
    avoidance_radius: float,
    duration: float,
    timestep: float,
    y_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Daisy offsets using fyst-trajectories.

    Uses the internal ``_generate_daisy_pattern`` method to get raw x/y
    offsets (before coordinate conversion), matching what scan_patterns
    produces.

    Parameters
    ----------
    velocity : float
        Scan velocity in degrees/second.
    start_acceleration : float
        Start acceleration in degrees/second^2.
    radius : float
        Characteristic radius R0 in degrees.
    turn_radius : float
        Turn radius in degrees.
    avoidance_radius : float
        Avoidance radius in degrees.
    duration : float
        Total duration in seconds.
    timestep : float
        Time between samples in seconds.
    y_offset : float, optional
        Initial y offset in degrees. Default is 0.0.

    Returns
    -------
    x : np.ndarray
        X offsets in degrees.
    y : np.ndarray
        Y offsets in degrees.
    """
    from fyst_trajectories.patterns.configs import DaisyScanConfig
    from fyst_trajectories.patterns.daisy import _DAISY_INTERNAL_TIMESTEP, DaisyScanPattern

    config = DaisyScanConfig(
        timestep=timestep,
        radius=radius,
        velocity=velocity,
        turn_radius=turn_radius,
        avoidance_radius=avoidance_radius,
        start_acceleration=start_acceleration,
        y_offset=y_offset,
    )
    pattern = DaisyScanPattern(ra=0.0, dec=0.0, config=config)

    if timestep > _DAISY_INTERNAL_TIMESTEP:
        sample_every = math.ceil(timestep / _DAISY_INTERNAL_TIMESTEP)
        dt = timestep / sample_every
    else:
        sample_every = 1
        dt = timestep

    x, y = pattern._generate_daisy_pattern(
        duration=duration,
        dt=dt,
        r0=radius,
        rt=turn_radius,
        ra_avoid=avoidance_radius,
        target_speed=velocity,
        start_acc=start_acceleration,
        y_offset=y_offset,
    )

    if sample_every > 1:
        x = x[::sample_every]
        y = y[::sample_every]

    return x, y


# ---------------------------------------------------------------------------
# Level 1 -- Sky pattern offset tests
# ---------------------------------------------------------------------------


class TestPongSkyOffsets:
    """Cross-validate Pong Fourier expansion between the two packages.

    Both packages implement the identical Fourier triangle-wave formula.
    When evaluated at the same time points the results should agree to
    floating-point precision.
    """

    # Pure-math agreement: the only difference is scalar vs vectorized
    # evaluation of the same sin() series, so machine-epsilon tolerance.
    TOLERANCE = 1e-10  # degrees

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "num_terms,width,height,spacing,velocity",
        [
            (4, 2.0, 2.0, 0.1, 0.5),
            (8, 1.0, 3.0, 0.05, 1.0),
            (2, 5.0, 5.0, 0.2, 0.3),
        ],
        ids=["square-4term", "rect-8term", "wide-2term"],
    )
    def test_pong_offsets_match(self, num_terms, width, height, spacing, velocity):
        """Pong x/y offsets agree to machine precision for identical time points."""
        sample_interval = 1.0 / 400.0
        duration = 50.0
        n_points = int(round(duration / sample_interval)) + 1
        times = np.linspace(0, duration, n_points)

        x_sp, y_sp = _pong_offsets_scanning(
            num_term=num_terms,
            width=width,
            height=height,
            spacing=spacing,
            velocity=velocity,
            sample_interval=sample_interval,
            times=times,
        )
        x_cc, y_cc = _pong_offsets_ccat(
            num_terms=num_terms,
            width=width,
            height=height,
            spacing=spacing,
            velocity=velocity,
            timestep=sample_interval,
            times=times,
        )

        np.testing.assert_allclose(
            x_sp,
            x_cc,
            atol=self.TOLERANCE,
            err_msg="Pong x offsets diverge between scan_patterns and fyst-trajectories",
        )
        np.testing.assert_allclose(
            y_sp,
            y_cc,
            atol=self.TOLERANCE,
            err_msg="Pong y offsets diverge between scan_patterns and fyst-trajectories",
        )

        # Velocity comparison via numerical gradient.
        # Gradient of machine-precision-identical positions: tolerance is
        # dominated by np.gradient's finite-difference accuracy (~dt^2).
        vx_sp = np.gradient(x_sp, times)
        vy_sp = np.gradient(y_sp, times)
        vx_cc = np.gradient(x_cc, times)
        vy_cc = np.gradient(y_cc, times)
        np.testing.assert_allclose(
            vx_sp,
            vx_cc,
            atol=1e-7,
            err_msg="Pong x velocities diverge",
        )
        np.testing.assert_allclose(
            vy_sp,
            vy_cc,
            atol=1e-7,
            err_msg="Pong y velocities diverge",
        )

    @pytest.mark.slow
    def test_pong_vertex_counts_agree(self):
        """Both packages compute identical vertex counts for the same inputs."""
        from fyst_trajectories.patterns.configs import PongScanConfig
        from fyst_trajectories.patterns.pong import PongScanPattern

        width, height, spacing = 2.0, 2.0, 0.1
        vert_spacing = math.sqrt(2) * spacing

        # -- scan_patterns vertex computation (inline, mirrors Pong._generate_scan) --
        x_nv_sp = math.ceil(width / vert_spacing)
        y_nv_sp = math.ceil(height / vert_spacing)
        if x_nv_sp % 2 == y_nv_sp % 2:
            if x_nv_sp >= y_nv_sp:
                y_nv_sp += 1
            else:
                x_nv_sp += 1
        nv = [x_nv_sp, y_nv_sp]
        most = nv.index(max(nv))
        while math.gcd(nv[0], nv[1]) != 1:
            nv[most] += 2
        x_nv_sp, y_nv_sp = nv

        # -- fyst-trajectories --
        config = PongScanConfig(
            timestep=0.01,
            width=width,
            height=height,
            spacing=spacing,
            velocity=0.5,
            num_terms=4,
            angle=0.0,
        )
        pattern = PongScanPattern(ra=0.0, dec=0.0, config=config)
        x_nv_cc, y_nv_cc, _, _ = pattern._compute_vertices()

        assert x_nv_sp == x_nv_cc, "x vertex counts differ"
        assert y_nv_sp == y_nv_cc, "y vertex counts differ"

    @pytest.mark.slow
    def test_pong_rotation_end_to_end(self):
        """Rotated Pong data from scan_patterns matches fyst-trajectories.

        Both packages apply rotation internally during data generation.
        This test compares the fully-rotated output arrays at matching
        time points to verify the rotation code paths agree.
        """
        num_terms = 4
        width, height, spacing, velocity = 2.0, 2.0, 0.1, 0.5
        angle = 30.0
        sample_interval = 1.0 / 400.0

        # -- scan_patterns: rotation is applied inside _generate_scan --
        pong = Pong(
            num_term=num_terms,
            width=width,
            height=height,
            spacing=spacing,
            velocity=velocity,
            angle=angle,
            sample_interval=sample_interval,
        )
        sp_data = pong.save_data()
        times_sp = np.array(sp_data["time_offset"])
        x_sp = np.array(sp_data["x_coord"])
        y_sp = np.array(sp_data["y_coord"])

        # -- fyst-trajectories: evaluate unrotated offsets, apply rotation --
        from fyst_trajectories.patterns.configs import PongScanConfig
        from fyst_trajectories.patterns.pong import PongScanPattern

        config = PongScanConfig(
            timestep=sample_interval,
            width=width,
            height=height,
            spacing=spacing,
            velocity=velocity,
            num_terms=num_terms,
            angle=angle,
        )
        pattern = PongScanPattern(ra=0.0, dec=0.0, config=config)

        x_nv, y_nv, amp_x, amp_y = pattern._compute_vertices()
        vert_spacing = math.sqrt(2) * spacing
        vavg = velocity / math.sqrt(2)
        peri_x = x_nv * vert_spacing * 2 / vavg
        peri_y = y_nv * vert_spacing * 2 / vavg

        x_cc = pattern._fourier_triangle_wave(
            num_terms,
            amp_x,
            times_sp,
            peri_x,
        )
        y_cc = pattern._fourier_triangle_wave(
            num_terms,
            amp_y,
            times_sp,
            peri_y,
        )

        angle_rad = math.radians(angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        x_cc_rot = x_cc * cos_a - y_cc * sin_a
        y_cc_rot = x_cc * sin_a + y_cc * cos_a

        # Align array lengths (scan_patterns uses cumulative addition
        # of sample_interval, which may produce one fewer point)
        min_len = min(len(x_sp), len(x_cc_rot))

        np.testing.assert_allclose(
            x_sp[:min_len],
            x_cc_rot[:min_len],
            atol=1e-10,
            err_msg="Rotated Pong x offsets diverge",
        )
        np.testing.assert_allclose(
            y_sp[:min_len],
            y_cc_rot[:min_len],
            atol=1e-10,
            err_msg="Rotated Pong y offsets diverge",
        )


class TestDaisySkyOffsets:
    """Cross-validate Daisy CV scan offsets between the two packages.

    The Daisy algorithm is an iterative physics simulation. Both packages
    implement the same equations but differ in two ways that affect
    numerical results:

    1. **Unit scale**: scan_patterns runs the simulation in arcseconds
       (converting back to degrees at the end), while fyst-trajectories runs
       in degrees throughout. Floating-point arithmetic at different
       magnitudes produces slightly different results that compound over
       the ~24000 iterative steps.

    2. **Epsilon guard**: fyst-trajectories adds a small-distance epsilon
       check (r < 1e-10 deg) at the origin that scan_patterns lacks.

    These differences are purely numerical -- the underlying equations
    are mathematically identical.
    """

    # The unit-scale difference (arcsec vs degrees) causes floating-point
    # divergence that compounds across ~24000 iterative steps. For the
    # standard case (Ra > 0), the turning threshold has margin, so the
    # two implementations make the same turning decisions and diverge
    # only through accumulated rounding. Empirically this stays under
    # ~1e-4 degrees (~0.4 arcsec) over 60 seconds.
    TOLERANCE = 5e-4  # degrees (~1.8 arcsec)

    # When Ra=0 the turning threshold is exactly 1.0 (the maximum
    # possible dot product), making the turn/straight decision
    # infinitely sensitive to floating-point differences. A tiny
    # position difference can cause one implementation to turn while
    # the other goes straight, leading to large subsequent divergence.
    # This is a fundamental numerical sensitivity, not a bug.
    TOLERANCE_NO_AVOIDANCE = 0.6  # degrees

    @pytest.mark.slow
    def test_daisy_offsets_converge_standard(self):
        """Daisy x/y offsets agree for the standard case (Ra > 0)."""
        velocity = 1.0 / 3.0
        R0 = 0.47
        Rt = 800.0 / 3600.0
        Ra = 600.0 / 3600.0
        start_acc = 0.2
        T = 60.0
        sample_interval = 1.0 / 400.0

        _times_sp, x_sp, y_sp = _daisy_offsets_scanning(
            velocity=velocity,
            start_acc=start_acc,
            R0=R0,
            Rt=Rt,
            Ra=Ra,
            T=T,
            sample_interval=sample_interval,
        )

        x_cc, y_cc = _daisy_offsets_ccat(
            velocity=velocity,
            start_acceleration=start_acc,
            radius=R0,
            turn_radius=Rt,
            avoidance_radius=Ra,
            duration=T,
            timestep=sample_interval,
        )

        min_len = min(len(x_sp), len(x_cc))
        x_sp = x_sp[:min_len]
        y_sp = y_sp[:min_len]
        x_cc = x_cc[:min_len]
        y_cc = y_cc[:min_len]

        # Skip the very first samples where the epsilon guard differs
        skip = 10
        np.testing.assert_allclose(
            x_sp[skip:],
            x_cc[skip:],
            atol=self.TOLERANCE,
            err_msg="Daisy x offsets diverge between scan_patterns and fyst-trajectories",
        )
        np.testing.assert_allclose(
            y_sp[skip:],
            y_cc[skip:],
            atol=self.TOLERANCE,
            err_msg="Daisy y offsets diverge between scan_patterns and fyst-trajectories",
        )

    @pytest.mark.slow
    def test_daisy_offsets_converge_no_avoidance(self):
        """Daisy offsets agree within generous tolerance when Ra=0.

        With Ra=0 the avoidance threshold becomes exactly 1.0 (the
        theoretical maximum of the dot product), so the turn/straight
        decision is degenerate. Tiny floating-point differences from
        the arcsec-vs-degree unit scale cause the two implementations
        to diverge at turning boundaries.

        We verify the overall scan shape is consistent (offsets stay
        within the same spatial envelope) rather than demanding
        point-by-point agreement.
        """
        velocity = 0.5
        R0 = 0.3
        Rt = 0.15
        Ra = 0.0
        start_acc = 0.5
        T = 60.0
        sample_interval = 1.0 / 400.0

        _times_sp, x_sp, y_sp = _daisy_offsets_scanning(
            velocity=velocity,
            start_acc=start_acc,
            R0=R0,
            Rt=Rt,
            Ra=Ra,
            T=T,
            sample_interval=sample_interval,
        )

        x_cc, y_cc = _daisy_offsets_ccat(
            velocity=velocity,
            start_acceleration=start_acc,
            radius=R0,
            turn_radius=Rt,
            avoidance_radius=Ra,
            duration=T,
            timestep=sample_interval,
        )

        min_len = min(len(x_sp), len(x_cc))
        x_sp = x_sp[:min_len]
        y_sp = y_sp[:min_len]
        x_cc = x_cc[:min_len]
        y_cc = y_cc[:min_len]

        skip = 10
        np.testing.assert_allclose(
            x_sp[skip:],
            x_cc[skip:],
            atol=self.TOLERANCE_NO_AVOIDANCE,
            err_msg="Daisy x offsets diverge (no-avoidance case)",
        )
        np.testing.assert_allclose(
            y_sp[skip:],
            y_cc[skip:],
            atol=self.TOLERANCE_NO_AVOIDANCE,
            err_msg="Daisy y offsets diverge (no-avoidance case)",
        )

        # Verify both implementations produce the same spatial envelope:
        # the max radius from origin should agree within 10%
        r_sp = np.sqrt(x_sp[skip:] ** 2 + y_sp[skip:] ** 2)
        r_cc = np.sqrt(x_cc[skip:] ** 2 + y_cc[skip:] ** 2)
        assert abs(r_sp.max() - r_cc.max()) / r_sp.max() < 0.1, (
            f"Scan envelopes differ: scan_patterns max_r={r_sp.max():.4f}, "
            f"fyst-trajectories max_r={r_cc.max():.4f}"
        )


# ---------------------------------------------------------------------------
# Level 2 -- AltAz trajectory tests
# ---------------------------------------------------------------------------


def _make_harmonized_site():
    """Create a Site using scan_patterns' exact FYST_LOC coordinates.

    Uses the same coordinates as scan_patterns so site-location
    differences don't contribute to cross-validation error.
    Refraction is disabled so both backends compute geometric
    (vacuum) coordinates.
    """
    from fyst_trajectories.site import (
        AxisLimits,
        Site,
        SunAvoidanceConfig,
        TelescopeLimits,
    )

    # -22d59m08.30s = -22.985639 deg, -67d44m25.00s = -67.740278 deg
    return Site(
        name="FYST-test",
        description="FYST with scan_patterns coordinates",
        latitude=-(22 + 59 / 60 + 8.30 / 3600),
        longitude=-(67 + 44 / 60 + 25.00 / 3600),
        elevation=5611.8,
        atmosphere=None,
        telescope_limits=TelescopeLimits(
            azimuth=AxisLimits(
                min=-270,
                max=270,
                max_velocity=3.0,
                max_acceleration=1.0,
            ),
            elevation=AxisLimits(
                min=20,
                max=90,
                max_velocity=1.0,
                max_acceleration=0.5,
            ),
        ),
        sun_avoidance=SunAvoidanceConfig(
            enabled=True,
            exclusion_radius=45.0,
            warning_radius=50.0,
        ),
    )


def _assert_altaz_agree(az_cc, el_cc, az_sp, el_sp, tolerance, skip=0):
    """Assert Az/El arrays agree within tolerance, handling wrapping."""
    min_len = min(len(az_cc), len(az_sp))
    az_cc = az_cc[skip:min_len]
    el_cc = el_cc[skip:min_len]
    az_sp = az_sp[skip:min_len]
    el_sp = el_sp[skip:min_len]

    el_diff = np.abs(el_cc - el_sp)
    assert np.max(el_diff) < tolerance, (
        f"Elevation mismatch: max diff = {np.max(el_diff):.4f} deg (tolerance = {tolerance} deg)"
    )

    az_diff = np.abs(az_cc - az_sp)
    az_diff = np.minimum(az_diff, 360.0 - az_diff)
    assert np.max(az_diff) < tolerance, (
        f"Azimuth mismatch: max diff = {np.max(az_diff):.4f} deg (tolerance = {tolerance} deg)"
    )


class TestPongAltAzTrajectory:
    """Cross-validate full AltAz Pong trajectories.

    The two packages use different astronomy backends for sky-to-AltAz
    conversion (manual spherical trig in scan_patterns vs astropy with
    IERS in fyst-trajectories). Refraction is disabled so both backends
    compute geometric (vacuum) coordinates, isolating the remaining
    differences:

    1. IERS/precession/nutation corrections (astropy only).
    2. Different sky-offset handling: flat-sky approximation vs
       SkyCoord.spherical_offsets_by.
    """

    # Dominated by the sky-offset method difference: scan_patterns uses
    # a flat-sky approximation, fyst-trajectories uses spherical_offsets_by.
    POSITION_TOLERANCE = 0.22  # degrees

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "ra,dec",
        [
            (180.0, -60.0),  # ~53 deg elevation (moderate)
            (150.0, -40.0),  # ~71 deg elevation (high)
        ],
        ids=["moderate-el", "high-el"],
    )
    def test_pong_altaz_agreement(self, ra, dec):
        """Pong AltAz trajectories agree within tolerance."""
        from fyst_trajectories.patterns.configs import PongScanConfig
        from fyst_trajectories.patterns.pong import PongScanPattern

        harmonized_site = _make_harmonized_site()
        sp_lat = harmonized_site.latitude
        sp_lon = harmonized_site.longitude

        start_time = Time("2026-03-15T04:00:00", scale="utc")
        num_terms = 4
        width, height = 2.0, 2.0
        spacing = 0.1
        velocity = 0.5
        sample_interval = 0.1
        duration = 60.0

        config = PongScanConfig(
            timestep=sample_interval,
            width=width,
            height=height,
            spacing=spacing,
            velocity=velocity,
            num_terms=num_terms,
            angle=0.0,
        )
        pattern = PongScanPattern(ra=ra, dec=dec, config=config)
        traj = pattern.generate(
            site=harmonized_site,
            duration=duration,
            start_time=start_time,
        )

        pong = Pong(
            num_term=num_terms,
            width=width,
            height=height,
            spacing=spacing,
            velocity=velocity,
            sample_interval=sample_interval,
        )
        tp = TelescopePattern(
            pong,
            start_ra=ra,
            start_dec=dec,
            start_datetime=start_time.datetime,
            lat=sp_lat,
            lon=sp_lon,
        )

        sp_data = tp.save_data(columns=["az_coord", "alt_coord"])
        az_sp = np.array(sp_data["az_coord"])
        el_sp = np.array(sp_data["alt_coord"])

        _assert_altaz_agree(
            traj.az,
            traj.el,
            az_sp,
            el_sp,
            self.POSITION_TOLERANCE,
        )


class TestDaisyAltAzTrajectory:
    """Cross-validate full AltAz Daisy trajectories.

    Same approach as TestPongAltAzTrajectory. Tolerance is slightly
    larger because the Daisy Level 1 offsets carry accumulated
    divergence from the iterative simulation (arcsec vs degree scale).
    """

    # Pong AltAz tolerance (0.22 deg) plus Daisy Level 1 divergence
    POSITION_TOLERANCE = 0.25  # degrees

    @pytest.mark.slow
    def test_daisy_altaz_agreement(self):
        """Daisy AltAz trajectories agree within tolerance."""
        from fyst_trajectories.patterns.configs import DaisyScanConfig
        from fyst_trajectories.patterns.daisy import DaisyScanPattern

        harmonized_site = _make_harmonized_site()
        sp_lat = harmonized_site.latitude
        sp_lon = harmonized_site.longitude

        ra, dec = 180.0, -60.0
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        velocity = 1.0 / 3.0
        R0 = 0.47
        Rt = 800.0 / 3600.0
        Ra = 600.0 / 3600.0
        start_acc = 0.2
        duration = 60.0
        sample_interval = 0.1

        config = DaisyScanConfig(
            timestep=sample_interval,
            radius=R0,
            velocity=velocity,
            turn_radius=Rt,
            avoidance_radius=Ra,
            start_acceleration=start_acc,
            y_offset=0.0,
        )
        pattern = DaisyScanPattern(ra=ra, dec=dec, config=config)
        traj = pattern.generate(
            site=harmonized_site,
            duration=duration,
            start_time=start_time,
        )

        daisy = Daisy(
            velocity=velocity,
            start_acc=start_acc,
            R0=R0,
            Rt=Rt,
            Ra=Ra,
            T=duration,
            sample_interval=sample_interval,
        )
        tp = TelescopePattern(
            daisy,
            start_ra=ra,
            start_dec=dec,
            start_datetime=start_time.datetime,
            lat=sp_lat,
            lon=sp_lon,
        )

        sp_data = tp.save_data(columns=["az_coord", "alt_coord"])
        az_sp = np.array(sp_data["az_coord"])
        el_sp = np.array(sp_data["alt_coord"])

        # Skip first 10 points where the epsilon guard differs
        _assert_altaz_agree(
            traj.az,
            traj.el,
            az_sp,
            el_sp,
            self.POSITION_TOLERANCE,
            skip=10,
        )
