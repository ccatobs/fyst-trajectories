"""Pin the falsifiable numbers the documentation states, per the project rule.

A falsifiable number in a comment or docstring needs a test. These tests anchor
the numeric claims the documentation states:

- the pong velocity-overshoot band (``PongScanConfig.velocity`` docstring):
  roughly 9 to 18 percent for ``num_terms >= 4``, about 27 percent at
  ``num_terms=1``, oscillating rather than converging;
- the staggered-retune arithmetic (``inject_retune`` docstring): the
  per-module cost is ``retune_duration / retune_interval``, about 1.7 percent
  at the shipped defaults of 5 s every 300 s;
- the slew-row ``azmin`` / ``azmax`` semantics (``write_timeline`` and the
  ECSV schema page): from/to azimuths, preserved unordered;
- the two rendered tables of code-derived numbers (the PrimeCam inner-ring
  offsets on the offsets page, the pending-verification defaults on the index
  page), cell by cell against the constants they are printed from;
- the LSA-window duration claim (``planning.rst`` and the
  ``plan_constant_el_scan`` docstring): the window is solar hours, and
  ``ScanBlock.duration`` is that window quantised to whole azimuth legs.
"""

import inspect
import math
import re
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time, TimeDelta

from fyst_trajectories import get_fyst_site
from fyst_trajectories.patterns.configs import PongScanConfig
from fyst_trajectories.patterns.pong import PongScanPattern
from fyst_trajectories.trajectory_utils import DEFAULT_RETUNE_DURATION_SEC, inject_retune

DOCS = Path(__file__).resolve().parents[1] / "docs"


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


def _list_table_cells(page, heading):
    """Map first cell -> second cell for the list-table under ``heading`` on an rst page."""
    body = (DOCS / page).read_text(encoding="utf-8").split(heading, 1)[1]
    body = body.split(".. list-table::", 1)[1]
    rows = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not line.startswith("   "):  # the first dedent ends the table
            break
        if stripped.startswith(":"):  # directive options
            continue
        if stripped.startswith("* - "):
            rows.append([stripped[4:]])
        elif stripped.startswith("- "):
            rows[-1].append(stripped[2:])
        else:  # continuation of the cell above
            rows[-1][-1] += " " + stripped
    return {row[0]: row[1] for row in rows[1:]}  # row 0 is the header


def _numbers(cell):
    """Every number in a rendered table cell, digit-group spaces removed."""
    ungrouped = re.sub(r"(?<=\d) (?=\d)", "", cell)
    return [float(text) for text in re.findall(r"-?\d+(?:\.\d+)?", ungrouped)]


class TestRenderedTables:
    """The docs' two tables of code-derived numbers, cell by cell.

    Both tables print values derived from module constants, and the pages
    themselves say a plate-scale or radius revision should be expected. A
    revision that leaves either table behind fails here.
    """

    def test_inner_ring_offsets_match_the_module_constants(self):
        from fyst_trajectories.primecam import INNER_RING_RADIUS_MM, PRIMECAM_MODULES
        from fyst_trajectories.site import FYST_PLATE_SCALE

        text = (DOCS / "instrument_offsets.rst").read_text(encoding="utf-8")
        radius_arcmin = INNER_RING_RADIUS_MM * FYST_PLATE_SCALE / 60.0
        assert f"({radius_arcmin / 60.0:.2f} deg = {radius_arcmin:.1f} arcmin from center)" in text
        printed = {}
        for line in text.splitlines():
            if not line.startswith("| i"):
                continue
            name, dx, dy = (cell.strip() for cell in line.strip("|").split("|"))
            printed[name] = (float(dx), float(dy))
        assert set(printed) == {f"i{index}" for index in range(1, 7)}
        for name, (dx, dy) in printed.items():
            offset = PRIMECAM_MODULES[name]
            assert (dx, dy) == (round(offset.dx, 1), round(offset.dy, 1)), name

    def test_index_defaults_match_the_library_constants(self):
        from fyst_trajectories import primecam, site

        cells = _list_table_cells("index.rst", "Pending instrument verification")
        interval = inspect.signature(inject_retune).parameters["retune_interval"].default
        expected = {
            "Sun exclusion / warning radii": [
                site.FYST_SUN_EXCLUSION_RADIUS,
                site.FYST_SUN_WARNING_RADIUS,
            ],
            "Az/El velocity limits": [site.FYST_AZ_MAX_VELOCITY, site.FYST_EL_MAX_VELOCITY],
            "Az/El acceleration limits": [
                site.FYST_AZ_MAX_ACCELERATION,
                site.FYST_EL_MAX_ACCELERATION,
            ],
            "Plate scale": [site.FYST_PLATE_SCALE],
            "PrimeCam inner ring radius": [primecam.INNER_RING_RADIUS_MM],
            "Per-module FOV radius (PrimeCam)": [primecam.MODULE_FOV_RADIUS_DEG],
            "Retune interval (in-scan)": [interval],
        }
        assert set(expected) <= set(cells)
        for parameter, values in expected.items():
            assert _numbers(cells[parameter])[: len(values)] == values, parameter


@pytest.mark.offline
class TestRenderedSimulatorDefaults:
    """The index table's simulator-tier rows, against the policy dataclasses."""

    def test_index_defaults_match_the_simulator_policies(self):
        from fyst_trajectories.overhead import CalibrationPolicy, OverheadModel
        from fyst_trajectories.overhead.calibration_night import CalibrationNightPolicy

        cells = _list_table_cells("index.rst", "Pending instrument verification")
        policy = CalibrationPolicy()
        night = CalibrationNightPolicy()
        expected = {
            "Whole-array retune duration": [OverheadModel().retune_duration],
            "Skydip cadence": [policy.skydip_cadence],
            "Calibration cadences (offline simulator)": [
                policy.pointing_cadence,
                policy.focus_cadence,
                policy.planet_cal_cadence,
            ],
            "Planet-calibration scan geometry": [float(policy.planet_cal_passes)],
            "Calibration-night scan tables and slew rates": [night.az_speed, night.az_accel],
        }
        assert set(expected) <= set(cells)
        for parameter, values in expected.items():
            assert _numbers(cells[parameter])[: len(values)] == values, parameter
        assert policy.planet_cal_footprint == "c"
        assert policy.planet_cal_scan is False


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


class TestLsaWindowDuration:
    """planning.rst / plan_constant_el_scan: solar hours, quantised to whole legs.

    The page's own example asks for a 20 deg LSA window and the docs now say
    two things about it: the scan spans ``(delta_lsa) / 15`` hours of UTC,
    about 0.3 percent longer than the sidereal window it names, and
    ``ScanBlock.duration`` is that span quantised to whole azimuth legs
    (4793.46 s against the 4800 s window, re-run, not derived).
    """

    def test_recorded_window_is_solar_and_duration_is_quantised(self):
        from fyst_trajectories.planning import FieldRegion, plan_constant_el_scan

        field = FieldRegion(ra_center=30.0, dec_center=-47.0, width=10.0, height=6.0)
        block = plan_constant_el_scan(
            field=field,
            elevation=45.0,
            velocity=0.5,
            site=get_fyst_site(),
            start_time=Time("2026-09-15T00:00:00", scale="utc"),
            lsa_window=(310.0, 330.0),
        )
        params = block.computed_params
        window = (
            Time(params["end_time_iso"], scale="utc") - Time(params["start_time_iso"], scale="utc")
        ).sec
        assert window == pytest.approx(20.0 / 15.0 * 3600.0, abs=1e-3)
        assert block.duration == pytest.approx(4793.46, abs=0.01)
        assert block.duration < window

    def test_a_solar_hour_window_overruns_the_sidereal_one_by_about_0_3_percent(self):
        # astropy's own sidereal day, not a re-typed ratio.
        excess = 86400.0 / (1.0 * u.sday).to_value(u.s) - 1.0
        assert 0.0025 <= excess <= 0.0030


@pytest.mark.offline
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
    """get_parallactic_angle Notes: ~820 s per degree.

    The docs state the time for a 180 deg parallactic-angle swing at
    transit scales with the transit zenith distance at roughly 820 s per
    degree. Analytically the swing rate integrates to
    t_180 = 180 deg * sin(z) / (omega_sidereal * cos(latitude)); pin the
    stated coefficient at z = 1 deg against the closed form.
    """

    def test_820_seconds_per_degree_at_one_degree(self):
        from fyst_trajectories.site import FYST_LATITUDE

        # The sidereal rate is astropy's, not a re-typed literal; the latitude
        # is the site constant, so a change to either reaches this test.
        omega = 360.0 / (1.0 * u.sday).to_value(u.s)  # deg per SI second
        lat = FYST_LATITUDE
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


@pytest.mark.offline
class TestCalibrationNightNumbers:
    """overhead_calibration_night.rst: the turnaround peak, the duty cycle, the table width."""

    def _leg(self):
        from fyst_trajectories import ConstantElScanConfig, TrajectoryBuilder, get_fyst_site

        config = ConstantElScanConfig(
            timestep=0.1, az_start=0.0, az_stop=2.44, elevation=45.0, az_speed=1.5, az_accel=1.5
        )
        return (
            TrajectoryBuilder(get_fyst_site())
            .with_config(config)
            .duration(120.0)
            .build(validate_dynamics=False)
        )

    def test_quintic_turnaround_peaks_at_one_and_a_half_times_nominal(self):
        """'the default 1.5 deg/s^2 reaches 2.25 deg/s^2' (measured on the sampled leg)."""
        traj = self._leg()
        az_vel = np.gradient(np.unwrap(traj.az, period=360.0), traj.times)
        peak = float(np.abs(np.gradient(az_vel, traj.times)).max())
        assert peak == pytest.approx(2.25, abs=0.05)

    def test_science_fraction_at_the_instrument_defaults(self):
        """'a 2.44 deg leg spends about 45 percent of its samples in science'."""
        assert self._leg().science_mask.mean() == pytest.approx(0.45, abs=0.02)

    def test_table_widths_are_one_point_six_module_widths_on_sky(self):
        """The shipped widths are a constant on-sky extent of 1.6 module widths (2.08 deg)."""
        from fyst_trajectories.overhead import DEFAULT_SCAN_TABLES
        from fyst_trajectories.planning.footprints import resolve_footprint

        fp = resolve_footprint("c")
        module = float(fp.cover_xi_deg.max() - fp.cover_xi_deg.min())
        bins = DEFAULT_SCAN_TABLES["default"].bins
        sky = [b.az_throw * math.cos(math.radians(b.el_centre)) for b in bins]
        assert module == pytest.approx(1.30, abs=0.01)
        assert sum(sky) / len(sky) == pytest.approx(2.08, abs=0.02)
        assert sum(sky) / len(sky) / module == pytest.approx(1.60, abs=0.02)
