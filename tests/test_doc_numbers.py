"""Pin the falsifiable numbers the documentation states, per the project rule.

A falsifiable number in a comment or docstring needs a test. These tests anchor
the numeric claims corrected in the documentation accuracy pass:

- the pong velocity-overshoot band (``PongScanConfig.velocity`` docstring):
  roughly 9 to 18 percent for ``num_terms >= 4``, about 27 percent at
  ``num_terms=1``, oscillating rather than converging;
- the staggered-retune arithmetic (``inject_retune`` docstring): the
  per-module cost is ``retune_duration / retune_interval``, about 1.7 percent
  at the shipped defaults of 5 s every 300 s;
- the slew-row ``azmin`` / ``azmax`` semantics (``write_timeline`` and the
  ECSV schema page): from/to azimuths, preserved unordered.
"""

import inspect
import math

import numpy as np
import pytest
from astropy.time import Time, TimeDelta

from fyst_trajectories import get_fyst_site
from fyst_trajectories.patterns.configs import PongScanConfig
from fyst_trajectories.patterns.pong import PongScanPattern
from fyst_trajectories.trajectory_utils import DEFAULT_RETUNE_DURATION_SEC, inject_retune


def _pong_overshoot(num_terms, width=3.0, height=3.0, spacing=0.1, velocity=0.5, n_pts=400_001):
    """Peak diagonal speed of the shipped truncated series, relative to ``velocity``."""
    cfg = PongScanConfig(
        timestep=0.01,
        width=width,
        height=height,
        spacing=spacing,
        velocity=velocity,
        num_terms=num_terms,
        angle=0.0,
    )
    pattern = PongScanPattern(ra=180.0, dec=-30.0, config=cfg)
    x_nv, y_nv, amp_x, amp_y = pattern._compute_vertices()
    vert = math.sqrt(2) * spacing
    vavg = velocity / math.sqrt(2)
    period_x = x_nv * vert * 2 / vavg
    period_y = y_nv * vert * 2 / vavg
    t = np.linspace(0.0, 2.0 * max(period_x, period_y), n_pts)
    dt = t[1] - t[0]
    x = pattern._fourier_triangle_wave(num_terms, amp_x, t, period_x)
    y = pattern._fourier_triangle_wave(num_terms, amp_y, t, period_y)
    speed = np.hypot(np.gradient(x, dt), np.gradient(y, dt))
    return float(speed.max()) / velocity - 1.0


class TestPongOvershootBand:
    """The docstring's overshoot claims, measured from the shipped series."""

    def test_num_terms_one_is_about_27_percent(self):
        assert 0.26 <= _pong_overshoot(1) <= 0.285

    def test_band_and_ceiling_for_practical_num_terms(self):
        overshoots = {n: _pong_overshoot(n) for n in (4, 10, 16, 64)}
        for n, value in overshoots.items():
            assert 0.085 <= value <= 0.185, f"num_terms={n}: overshoot {value:.3f} out of band"

    def test_overshoot_does_not_converge_monotonically(self):
        values = [_pong_overshoot(n) for n in (4, 10, 16, 64)]
        diffs = np.diff(values)
        assert (diffs > 0).any() and (diffs < 0).any(), (
            f"overshoot sequence {values} looks monotonic; the docstring says it oscillates"
        )


class TestRetuneArithmetic:
    """The inject_retune docstring's 'about 1.7% at the defaults, 5 s every 300 s'."""

    def test_defaults_match_the_stated_numbers(self):
        assert DEFAULT_RETUNE_DURATION_SEC == 5.0
        default_interval = inspect.signature(inject_retune).parameters["retune_interval"].default
        assert default_interval == 300.0
        fraction = DEFAULT_RETUNE_DURATION_SEC / default_interval
        assert abs(fraction - 0.017) < 0.001


class TestSlewRowAzimuthOrder:
    """Slew rows keep from/to azimuths, unordered, through the ECSV round trip."""

    def test_negative_direction_slew_row_is_unordered(self, tmp_path):
        from astropy.table import Table

        from fyst_trajectories.overhead.io import write_timeline
        from fyst_trajectories.overhead.models import (
            CalibrationPolicy,
            ObservingTimeline,
            OverheadModel,
            TimelineBlock,
        )

        t0 = Time("2026-06-15T02:00:00", scale="utc")
        slew = TimelineBlock(
            t_start=t0,
            t_stop=t0 + TimeDelta(30, format="sec"),
            block_type="slew",
            patch_name="slew",
            az_start=180.0,
            az_end=120.0,
            elevation=45.0,
            scan_index=0,
            scan_type="none",
            metadata={},
        )
        timeline = ObservingTimeline(
            blocks=[slew],
            site=get_fyst_site(),
            start_time=t0,
            end_time=t0 + TimeDelta(60, format="sec"),
            overhead_model=OverheadModel(),
            calibration_policy=CalibrationPolicy(),
        )
        path = tmp_path / "slew.ecsv"
        write_timeline(timeline, path)
        table = Table.read(path)
        assert float(table["azmin"][0]) == 180.0
        assert float(table["azmax"][0]) == 120.0


class TestTransitRotationScaling:
    """coordinate_systems.rst / get_parallactic_angle: ~820 s per degree.

    The docs state the time for a 180 deg parallactic-angle swing at
    transit scales with the transit zenith distance at roughly 820 s per
    degree. Analytically the swing rate integrates to
    t_180 = 180 deg * sin(z) / (omega_sidereal * cos(latitude)); pin the
    stated coefficient at z = 1 deg against the closed form.
    """

    def test_820_seconds_per_degree_at_one_degree(self):
        import math

        omega = 360.0 / 86164.0905  # sidereal rate, deg per SI second
        lat = -22.985639
        z = 1.0
        t_180 = 180.0 * math.sin(math.radians(z)) / (omega * math.cos(math.radians(lat)))
        # The docs say "roughly 820 s per degree".
        assert t_180 == pytest.approx(820.0, rel=0.02)


class TestSummerSunCap:
    """sun_avoidance.rst: the midsummer Sun transits nearly overhead.

    The docs state FYST's latitude sits within half a degree of the
    solstice solar declination, so around midsummer ``sun_el`` reaches
    about 90 deg and the 45 deg scalar radius caps safe elevations near
    45 deg (cap = 180 - radius - sun_el).
    """

    def test_solstice_sun_peaks_within_half_degree_of_zenith(self):
        from astropy.time import Time

        from fyst_trajectories import Coordinates, get_fyst_site

        coords = Coordinates(get_fyst_site())
        # Solar noon near the December solstice: sample transit hours over
        # a few days around it.
        peaks = []
        for day in ("2026-12-20", "2026-12-21", "2026-12-22", "2026-12-23"):
            for hh in ("16:30", "16:45", "17:00", "17:15", "17:30"):
                _az, el = coords.get_sun_altaz(Time(f"{day}T{hh}:00", scale="utc"))
                peaks.append(float(el))
        sun_peak = max(peaks)
        assert sun_peak > 89.4  # "sun_el up to about 90 deg"
        cap = 180.0 - 45.0 - sun_peak
        assert cap == pytest.approx(45.0, abs=0.6)  # "falls to about 45 deg"
