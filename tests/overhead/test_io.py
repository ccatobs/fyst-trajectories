"""Tests for TOAST-compatible ECSV I/O."""

import math

import numpy as np
import pytest
from astropy.table import Table
from astropy.time import Time, TimeDelta

from fyst_trajectories import Coordinates, get_fyst_site
from fyst_trajectories.overhead import (
    ObservingPatch,
    generate_timeline,
    schedule_to_trajectories,
)
from fyst_trajectories.overhead.io import (
    read_timeline,
    write_timeline,
)
from fyst_trajectories.overhead.models import (
    BlockType,
    CalibrationPolicy,
    ObservingTimeline,
    OverheadModel,
    TimelineBlock,
)
from fyst_trajectories.overhead.utils import compute_nasmyth_rotation


def _make_test_timeline():
    """Create a minimal test timeline."""
    site = get_fyst_site()
    t0 = Time("2026-06-15T02:00:00", scale="utc")
    t1 = t0 + TimeDelta(3600, format="sec")

    blocks = [
        TimelineBlock(
            t_start=t0,
            t_stop=t0 + TimeDelta(300, format="sec"),
            block_type="calibration",
            patch_name="retune",
            az_start=180.0,
            az_end=180.0,
            elevation=50.0,
            scan_index=0,
            scan_type="retune",
        ),
        TimelineBlock(
            t_start=t0 + TimeDelta(300, format="sec"),
            t_stop=t0 + TimeDelta(2100, format="sec"),
            block_type="science",
            patch_name="deep_field",
            az_start=120.0,
            az_end=240.0,
            elevation=50.0,
            scan_index=1,
            rising=True,
            scan_type="pong",
            metadata={
                "ra_center": 180.0,
                "dec_center": -30.0,
                "width": 4.0,
                "height": 4.0,
                "velocity": 0.5,
                "scan_params": {"spacing": 0.1, "num_terms": 4},
            },
        ),
        TimelineBlock(
            t_start=t0 + TimeDelta(2100, format="sec"),
            t_stop=t0 + TimeDelta(2105, format="sec"),
            block_type="calibration",
            patch_name="retune",
            az_start=240.0,
            az_end=240.0,
            elevation=50.0,
            scan_index=2,
            scan_type="retune",
        ),
    ]

    return ObservingTimeline(
        blocks=blocks,
        site=site,
        start_time=t0,
        end_time=t1,
        overhead_model=OverheadModel(),
        calibration_policy=CalibrationPolicy(),
        metadata={"test_key": "test_value"},
    )


class TestWriteTimeline:
    """Tests for write_timeline()."""

    def test_writes_file(self, tmp_path):
        timeline = _make_test_timeline()
        path = tmp_path / "test_timeline.ecsv"
        write_timeline(timeline, path)
        assert path.exists()
        content = path.read_text()
        assert "deep_field" in content
        assert "retune" in content

    def test_empty_timeline(self, tmp_path):
        site = get_fyst_site()
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        timeline = ObservingTimeline(
            blocks=[],
            site=site,
            start_time=t0,
            end_time=t0,
            overhead_model=OverheadModel(),
            calibration_policy=CalibrationPolicy(),
        )
        path = tmp_path / "empty.ecsv"
        write_timeline(timeline, path)
        assert path.exists()


class TestReadTimeline:
    """Tests for read_timeline()."""

    def test_round_trip(self, tmp_path):
        timeline = _make_test_timeline()
        path = tmp_path / "rt_timeline.ecsv"
        write_timeline(timeline, path)

        loaded = read_timeline(path)
        assert len(loaded.blocks) == len(timeline.blocks)
        assert loaded.blocks[0].block_type == "calibration"
        assert loaded.blocks[1].block_type == "science"
        assert loaded.blocks[1].patch_name == "deep_field"

    def test_preserves_times(self, tmp_path):
        timeline = _make_test_timeline()
        path = tmp_path / "times.ecsv"
        write_timeline(timeline, path)

        loaded = read_timeline(path)
        for orig, loaded_b in zip(timeline.blocks, loaded.blocks):
            assert abs(orig.t_start.unix - loaded_b.t_start.unix) < 1.0
            assert abs(orig.t_stop.unix - loaded_b.t_stop.unix) < 1.0

    def test_preserves_metadata(self, tmp_path):
        timeline = _make_test_timeline()
        path = tmp_path / "meta.ecsv"
        write_timeline(timeline, path)

        loaded = read_timeline(path)
        assert loaded.metadata.get("test_key") == "test_value"

    def test_preserves_block_types(self, tmp_path):
        timeline = _make_test_timeline()
        path = tmp_path / "types.ecsv"
        write_timeline(timeline, path)

        loaded = read_timeline(path)
        types = [b.block_type for b in loaded.blocks]
        assert types == ["calibration", "science", "calibration"]

    def test_preserves_scan_types(self, tmp_path):
        timeline = _make_test_timeline()
        path = tmp_path / "scan_types.ecsv"
        write_timeline(timeline, path)

        loaded = read_timeline(path)
        scan_types = [b.scan_type for b in loaded.blocks]
        assert scan_types == ["retune", "pong", "retune"]


class TestCanonicalColumnNames:
    """F-2: verify write_timeline produces TOAST canonical column names."""

    def test_uses_toast_column_names(self, tmp_path):
        """Written ECSV must use start_time/stop_time ISO + scan_index names."""
        timeline = _make_test_timeline()
        path = tmp_path / "canonical.ecsv"
        write_timeline(timeline, path)

        table = Table.read(str(path), format="ascii.ecsv")
        assert "start_time" in table.colnames
        assert "stop_time" in table.colnames
        assert "scan_index" in table.colnames
        assert "subscan_index" in table.colnames
        # Legacy MJD/scan column names must NOT be present in new writes.
        assert "start_timestamp" not in table.colnames
        assert "stop_timestamp" not in table.colnames
        assert "scan" not in table.colnames
        assert "subscan" not in table.colnames

    def test_times_written_as_iso_strings(self, tmp_path):
        """start_time/stop_time must be ISO strings, not MJD floats."""
        timeline = _make_test_timeline()
        path = tmp_path / "iso_times.ecsv"
        write_timeline(timeline, path)

        table = Table.read(str(path), format="ascii.ecsv")
        first = str(table["start_time"][0])
        # ISO format looks like "2026-06-15 02:00:00.000"
        assert "2026-06-15" in first


class TestMetadataPersistence:
    """F-1: per-block science metadata must survive an ECSV round-trip."""

    def test_science_metadata_round_trip(self, tmp_path):
        timeline = _make_test_timeline()
        path = tmp_path / "meta_rt.ecsv"
        write_timeline(timeline, path)

        loaded = read_timeline(path)
        sci = [b for b in loaded.blocks if b.block_type == BlockType.SCIENCE]
        assert len(sci) == 1
        meta = sci[0].metadata
        assert meta["ra_center"] == 180.0
        assert meta["dec_center"] == -30.0
        assert meta["width"] == 4.0
        assert meta["height"] == 4.0
        assert meta["velocity"] == 0.5
        assert meta["scan_params"] == {"spacing": 0.1, "num_terms": 4}

    def test_non_science_metadata_is_empty(self, tmp_path):
        timeline = _make_test_timeline()
        path = tmp_path / "meta_empty.ecsv"
        write_timeline(timeline, path)

        loaded = read_timeline(path)
        non_sci = [b for b in loaded.blocks if b.block_type != BlockType.SCIENCE]
        for b in non_sci:
            assert b.metadata == {}

    def test_metadata_roundtrip_through_simulation_bridge(self, tmp_path):
        """F-1: ensure schedule_to_trajectories works after ECSV roundtrip."""
        site = get_fyst_site()
        patches = [
            ObservingPatch(
                name="rt_field",
                ra_center=181.25,
                dec_center=-28.5,
                width=3.0,
                height=2.0,
                scan_type="pong",
                velocity=0.75,
                scan_params={"spacing": 0.12, "num_terms": 5},
            ),
        ]
        timeline = generate_timeline(
            patches=patches,
            site=site,
            start_time="2026-06-15T02:00:00",
            end_time="2026-06-15T03:00:00",
        )

        path = tmp_path / "bridge.ecsv"
        write_timeline(timeline, path)
        loaded = read_timeline(path)

        # The loaded science blocks must carry metadata matching the patch.
        assert loaded.n_science_scans > 0
        for b in loaded.science_blocks:
            assert b.metadata["ra_center"] == pytest.approx(181.25)
            assert b.metadata["dec_center"] == pytest.approx(-28.5)
            assert b.metadata["width"] == pytest.approx(3.0)
            assert b.metadata["height"] == pytest.approx(2.0)
            assert b.metadata["velocity"] == pytest.approx(0.75)
            assert b.metadata["scan_params"] == {"spacing": 0.12, "num_terms": 5}

        # The simulation bridge should succeed end-to-end on the loaded
        # timeline and reproduce the same pong geometry (width/height) as
        # the in-memory timeline (not the former 180/-30/4x4 hardcoded
        # fallback).
        in_mem = schedule_to_trajectories(timeline)
        loaded_pairs = schedule_to_trajectories(loaded)
        assert len(in_mem) == len(loaded_pairs)
        for (_, in_mem_sb), (_, loaded_sb) in zip(in_mem, loaded_pairs):
            assert in_mem_sb.config.width == pytest.approx(loaded_sb.config.width)
            assert in_mem_sb.config.height == pytest.approx(loaded_sb.config.height)
            assert in_mem_sb.config.velocity == pytest.approx(loaded_sb.config.velocity)
            assert loaded_sb.config.width == pytest.approx(3.0)
            assert loaded_sb.config.height == pytest.approx(2.0)


class TestSiteReconstruction:
    """F-3 (Arch-9): read_timeline reconstructs Site from table metadata."""

    def test_fyst_default_site(self, tmp_path):
        """A timeline written with the FYST default site must round-trip."""
        timeline = _make_test_timeline()
        path = tmp_path / "fyst_site.ecsv"
        write_timeline(timeline, path)
        loaded = read_timeline(path)
        assert loaded.site.latitude == pytest.approx(timeline.site.latitude)
        assert loaded.site.longitude == pytest.approx(timeline.site.longitude)
        assert loaded.site.elevation == pytest.approx(timeline.site.elevation)
        # Should be the FYST default instance (same nasmyth_port/plate_scale).
        assert loaded.site.nasmyth_port == timeline.site.nasmyth_port
        assert loaded.site.plate_scale == timeline.site.plate_scale

    def test_custom_site_coordinates_preserved(self, tmp_path):
        """A non-FYST site must be reconstructed from metadata, not replaced."""
        from fyst_trajectories.site import (
            AxisLimits,
            Site,
            SunAvoidanceConfig,
            TelescopeLimits,
        )

        custom_site = Site(
            name="custom",
            description="Test site",
            latitude=-30.0,
            longitude=-70.0,
            elevation=2500.0,
            atmosphere=None,
            telescope_limits=TelescopeLimits(
                azimuth=AxisLimits(min=-180.0, max=360.0, max_velocity=3.0, max_acceleration=1.0),
                elevation=AxisLimits(min=20.0, max=90.0, max_velocity=1.0, max_acceleration=0.5),
            ),
            sun_avoidance=SunAvoidanceConfig(
                enabled=True, exclusion_radius=45.0, warning_radius=50.0
            ),
            nasmyth_port="right",
            plate_scale=13.89,
        )
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        timeline = ObservingTimeline(
            blocks=[
                TimelineBlock(
                    t_start=t0,
                    t_stop=t0 + TimeDelta(60, format="sec"),
                    block_type="idle",
                    patch_name="noop",
                    az_start=180.0,
                    az_end=180.0,
                    elevation=50.0,
                    scan_index=0,
                )
            ],
            site=custom_site,
            start_time=t0,
            end_time=t0 + TimeDelta(60, format="sec"),
            overhead_model=OverheadModel(),
            calibration_policy=CalibrationPolicy(),
        )
        path = tmp_path / "custom_site.ecsv"
        write_timeline(timeline, path)
        loaded = read_timeline(path)
        assert loaded.site.latitude == pytest.approx(-30.0)
        assert loaded.site.longitude == pytest.approx(-70.0)
        assert loaded.site.elevation == pytest.approx(2500.0)


class TestBoresightAngle:
    """F-4: boresight_angle round-trips through ECSV."""

    def test_boresight_roundtrip(self, tmp_path):
        site = get_fyst_site()
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        # Construct a block with a known nonzero boresight_angle.
        az = 120.0
        el = 55.0
        expected = compute_nasmyth_rotation(az, el, site)
        block = TimelineBlock(
            t_start=t0,
            t_stop=t0 + TimeDelta(60, format="sec"),
            block_type="science",
            patch_name="bore",
            az_start=az - 5,
            az_end=az + 5,
            elevation=el,
            scan_index=0,
            scan_type="pong",
            boresight_angle=expected,
            metadata={
                "ra_center": 180.0,
                "dec_center": -30.0,
                "width": 2.0,
                "height": 2.0,
                "velocity": 0.5,
                "scan_params": {},
            },
        )
        timeline = ObservingTimeline(
            blocks=[block],
            site=site,
            start_time=t0,
            end_time=t0 + TimeDelta(60, format="sec"),
            overhead_model=OverheadModel(),
            calibration_policy=CalibrationPolicy(),
        )

        path = tmp_path / "bore.ecsv"
        write_timeline(timeline, path)
        loaded = read_timeline(path)
        assert len(loaded.blocks) == 1
        assert loaded.blocks[0].boresight_angle == pytest.approx(expected, abs=1e-9)


class TestNasmythConsistency:
    """Arch-16: compute_nasmyth_rotation matches Coordinates.get_parallactic_angle.

    The two implementations use different input variables (AltAz vs HA)
    but share the same underlying spherical trigonometry for the
    parallactic angle. We verify the equivalence by computing the HA-based
    formula directly from (HA, dec, lat) and the AltAz-based formula from
    the matching (az, el) point on the celestial sphere. Both forms must
    agree to machine precision on the shared geometric input.
    """

    def test_nasmyth_rotation_matches_coordinates(self):
        site = get_fyst_site()
        lat_rad = math.radians(site.latitude)

        # Sample (HA, dec) pairs that are well-separated from the zenith.
        samples = [
            (-30.0, -40.0),
            (15.0, -10.0),
            (75.0, -55.0),
            (-120.0, -70.0),
            (45.0, -20.0),
        ]

        for ha_deg, dec_deg in samples:
            ha_rad = math.radians(ha_deg)
            dec_rad = math.radians(dec_deg)

            # Geometric conversion from (HA, dec) to (az, el) at this
            # latitude (same spherical triangle used by get_parallactic_angle).
            sin_el = math.sin(lat_rad) * math.sin(dec_rad) + math.cos(lat_rad) * math.cos(
                dec_rad
            ) * math.cos(ha_rad)
            el_rad = math.asin(sin_el)
            cos_el = math.cos(el_rad)
            if cos_el < 1e-9:
                continue  # skip zenith samples
            sin_az = -math.sin(ha_rad) * math.cos(dec_rad) / cos_el
            cos_az = (math.sin(dec_rad) - math.sin(el_rad) * math.sin(lat_rad)) / (
                cos_el * math.cos(lat_rad)
            )
            az_deg = math.degrees(math.atan2(sin_az, cos_az))
            el_deg = math.degrees(el_rad)
            if el_deg > 85.0:
                continue

            # HA-based parallactic angle (same formula used by
            # Coordinates.get_parallactic_angle).
            numerator_ha = math.sin(ha_rad)
            denominator_ha = math.cos(dec_rad) * math.tan(lat_rad) - math.sin(dec_rad) * math.cos(
                ha_rad
            )
            pa_ha_deg = math.degrees(math.atan2(numerator_ha, denominator_ha))
            ha_bangle = site.nasmyth_sign * el_deg + pa_ha_deg

            # AltAz-based equivalent (what compute_nasmyth_rotation computes).
            altaz_bangle = compute_nasmyth_rotation(az_deg, el_deg, site)

            diff = math.fmod(altaz_bangle - ha_bangle + 540.0, 360.0) - 180.0
            assert abs(diff) < 1e-9, (
                f"PA mismatch at HA={ha_deg} dec={dec_deg}: "
                f"altaz={altaz_bangle}, ha={ha_bangle}, diff={diff}"
            )

    def test_nasmyth_rotation_matches_coordinates_via_instance(self):
        """Smoke-check equivalence using Coordinates with refraction off.

        Astropy's full ICRS→AltAz transform applies precession, nutation
        and aberration, so AltAz-derived PA won't match HA-derived PA to
        machine precision even with refraction off. We tolerate a fraction
        of a degree here and rely on the analytic test above for exact
        equivalence.
        """
        from fyst_trajectories.site import AtmosphericConditions

        site = get_fyst_site()
        coords = Coordinates(site, atmosphere=AtmosphericConditions.no_refraction())
        time = Time("2026-06-15T05:00:00", scale="utc")
        ra, dec = 120.0, -30.0

        az, el = coords.radec_to_altaz(ra, dec, time)
        pa = coords.get_parallactic_angle(ra, dec, time)

        altaz_bangle = compute_nasmyth_rotation(float(az), float(el), site)
        ha_bangle = site.nasmyth_sign * float(el) + float(pa)
        diff = math.fmod(altaz_bangle - ha_bangle + 540.0, 360.0) - 180.0
        # Loose tolerance because astropy applies frame corrections.
        assert abs(diff) < 1.0

    def test_nasmyth_rotation_vector_sanity(self):
        """Quick numerical sanity check at known coordinates."""
        site = get_fyst_site()
        # At zero azimuth and due south, numerator -sin(az)*cos(lat) = 0
        # → PA is 0 or 180 depending on the denominator sign.
        bangle = compute_nasmyth_rotation(0.0, 45.0, site)
        assert np.isfinite(bangle)


class TestOverheadModelRoundTrip:
    """All OverheadModel fields must survive an ECSV write/read round-trip."""

    def test_all_overhead_fields_round_trip(self, tmp_path):
        """Every OverheadModel field must be restored from ECSV metadata.

        Constructs every field with a value distinct from the class
        default (and distinct from every *other* field's default) so a
        future field that is added without the corresponding I/O wiring
        — like the BEAM_MAP regression — fails loudly instead of
        coincidentally matching a default on the read side.
        """
        import dataclasses

        site = get_fyst_site()
        t0 = Time("2026-06-15T02:00:00", scale="utc")

        # Use non-default values for every field so we can detect missing ones.
        overhead = OverheadModel(
            retune_duration=7.0,
            pointing_cal_duration=200.0,
            focus_duration=350.0,
            skydip_duration=400.0,
            planet_cal_duration=700.0,
            beam_map_duration=999.0,  # non-default to catch missing serialisation
            settle_time=8.0,
            min_scan_duration=90.0,
            max_scan_duration=4000.0,
        )

        # Sanity check that every field truly differs from the class default,
        # so the test is genuinely round-trip-sensitive for every field.
        defaults = OverheadModel()
        for fld in dataclasses.fields(OverheadModel):
            assert getattr(overhead, fld.name) != getattr(defaults, fld.name), (
                f"Test setup bug: OverheadModel.{fld.name} matches class default; "
                f"the round-trip test cannot detect a serialisation gap on this field."
            )

        timeline = ObservingTimeline(
            blocks=[
                TimelineBlock(
                    t_start=t0,
                    t_stop=t0 + TimeDelta(60, format="sec"),
                    block_type="idle",
                    patch_name="noop",
                    az_start=180.0,
                    az_end=180.0,
                    elevation=50.0,
                    scan_index=0,
                )
            ],
            site=site,
            start_time=t0,
            end_time=t0 + TimeDelta(60, format="sec"),
            overhead_model=overhead,
            calibration_policy=CalibrationPolicy(),
        )

        path = tmp_path / "overhead_rt.ecsv"
        write_timeline(timeline, path)
        loaded = read_timeline(path)

        for fld in dataclasses.fields(OverheadModel):
            original = getattr(overhead, fld.name)
            loaded_val = getattr(loaded.overhead_model, fld.name)
            assert loaded_val == pytest.approx(original), (
                f"OverheadModel.{fld.name}: wrote {original}, read back {loaded_val}"
            )


class TestCalibrationPolicyRoundTrip:
    """All CalibrationPolicy fields must survive an ECSV write/read round-trip."""

    def test_all_calibration_fields_round_trip(self, tmp_path):
        """Every CalibrationPolicy field must be restored from ECSV metadata.

        Like the OverheadModel round-trip test, every field is set to a
        value distinct from its class default. The ``beam_map_cadence``
        field — historically dropped on round-trip because the I/O path
        was not wired to it — is given a non-None value here so a
        regression on that field fails loudly.
        """
        import dataclasses

        site = get_fyst_site()
        t0 = Time("2026-06-15T02:00:00", scale="utc")

        # Use non-default values for every field.
        cal_policy = CalibrationPolicy(
            retune_cadence=10.0,
            pointing_cadence=2000.0,
            focus_cadence=8000.0,
            skydip_cadence=12000.0,
            planet_cal_cadence=50000.0,
            beam_map_cadence=86400.0,  # non-default (default None) to catch dropped serialisation
            planet_targets=("mars", "venus"),
            planet_min_elevation=25.0,
        )

        # Sanity check: every field really differs from the class default.
        defaults = CalibrationPolicy()
        for fld in dataclasses.fields(CalibrationPolicy):
            assert getattr(cal_policy, fld.name) != getattr(defaults, fld.name), (
                f"Test setup bug: CalibrationPolicy.{fld.name} matches class default; "
                f"the round-trip test cannot detect a serialisation gap on this field."
            )

        timeline = ObservingTimeline(
            blocks=[
                TimelineBlock(
                    t_start=t0,
                    t_stop=t0 + TimeDelta(60, format="sec"),
                    block_type="idle",
                    patch_name="noop",
                    az_start=180.0,
                    az_end=180.0,
                    elevation=50.0,
                    scan_index=0,
                )
            ],
            site=site,
            start_time=t0,
            end_time=t0 + TimeDelta(60, format="sec"),
            overhead_model=OverheadModel(),
            calibration_policy=cal_policy,
        )

        path = tmp_path / "calpol_rt.ecsv"
        write_timeline(timeline, path)
        loaded = read_timeline(path)

        for fld in dataclasses.fields(CalibrationPolicy):
            original = getattr(cal_policy, fld.name)
            loaded_val = getattr(loaded.calibration_policy, fld.name)
            assert loaded_val == original, (
                f"CalibrationPolicy.{fld.name}: wrote {original}, read back {loaded_val}"
            )

    def test_beam_map_cadence_none_round_trips(self, tmp_path):
        """``beam_map_cadence=None`` (the manual-only default) survives round-trip.

        ECSV preserves ``None`` in table metadata cleanly, so the
        default-constructed policy must round-trip without silently
        switching to a non-None value or raising on the read side.
        """
        site = get_fyst_site()
        t0 = Time("2026-06-15T02:00:00", scale="utc")

        cal_policy = CalibrationPolicy()  # beam_map_cadence is None
        assert cal_policy.beam_map_cadence is None

        timeline = ObservingTimeline(
            blocks=[
                TimelineBlock(
                    t_start=t0,
                    t_stop=t0 + TimeDelta(60, format="sec"),
                    block_type="idle",
                    patch_name="noop",
                    az_start=180.0,
                    az_end=180.0,
                    elevation=50.0,
                    scan_index=0,
                )
            ],
            site=site,
            start_time=t0,
            end_time=t0 + TimeDelta(60, format="sec"),
            overhead_model=OverheadModel(),
            calibration_policy=cal_policy,
        )

        path = tmp_path / "calpol_none_rt.ecsv"
        write_timeline(timeline, path)
        loaded = read_timeline(path)
        assert loaded.calibration_policy.beam_map_cadence is None


class TestCalibrationBlockMetadataRoundTrip:
    """Calibration block metadata (e.g. planet target) must survive round-trip."""

    def test_planet_cal_target_round_trip(self, tmp_path):
        """Planet calibration block metadata with target name survives ECSV."""
        site = get_fyst_site()
        t0 = Time("2026-06-15T02:00:00", scale="utc")

        blocks = [
            TimelineBlock(
                t_start=t0,
                t_stop=t0 + TimeDelta(600, format="sec"),
                block_type="calibration",
                patch_name="planet_cal",
                az_start=150.0,
                az_end=150.0,
                elevation=40.0,
                scan_index=0,
                scan_type="planet_cal",
                metadata={"cal_type": "planet_cal", "target": "jupiter"},
            ),
            TimelineBlock(
                t_start=t0 + TimeDelta(600, format="sec"),
                t_stop=t0 + TimeDelta(1200, format="sec"),
                block_type="science",
                patch_name="deep_field",
                az_start=120.0,
                az_end=240.0,
                elevation=50.0,
                scan_index=1,
                scan_type="pong",
                metadata={
                    "ra_center": 180.0,
                    "dec_center": -30.0,
                    "width": 4.0,
                    "height": 4.0,
                    "velocity": 0.5,
                    "scan_params": {},
                },
            ),
        ]

        timeline = ObservingTimeline(
            blocks=blocks,
            site=site,
            start_time=t0,
            end_time=t0 + TimeDelta(1200, format="sec"),
            overhead_model=OverheadModel(),
            calibration_policy=CalibrationPolicy(),
        )

        path = tmp_path / "cal_meta_rt.ecsv"
        write_timeline(timeline, path)
        loaded = read_timeline(path)

        cal_blocks = [b for b in loaded.blocks if b.block_type == BlockType.CALIBRATION]
        assert len(cal_blocks) == 1
        assert cal_blocks[0].metadata["target"] == "jupiter"
        assert cal_blocks[0].metadata["cal_type"] == "planet_cal"

    def test_retune_block_metadata_empty(self, tmp_path):
        """Retune blocks without extra metadata still round-trip as empty."""
        site = get_fyst_site()
        t0 = Time("2026-06-15T02:00:00", scale="utc")

        blocks = [
            TimelineBlock(
                t_start=t0,
                t_stop=t0 + TimeDelta(5, format="sec"),
                block_type="calibration",
                patch_name="retune",
                az_start=180.0,
                az_end=180.0,
                elevation=50.0,
                scan_index=0,
                scan_type="retune",
            ),
        ]

        timeline = ObservingTimeline(
            blocks=blocks,
            site=site,
            start_time=t0,
            end_time=t0 + TimeDelta(5, format="sec"),
            overhead_model=OverheadModel(),
            calibration_policy=CalibrationPolicy(),
        )

        path = tmp_path / "retune_meta_rt.ecsv"
        write_timeline(timeline, path)
        loaded = read_timeline(path)

        assert len(loaded.blocks) == 1
        assert loaded.blocks[0].metadata == {}


class TestRetuneEventsRoundTrip:
    """Round-2: ``retune_events`` carried on science block metadata round-trips.

    ``Trajectory.retune_events`` is the canonical home for event-level
    retune provenance, but :class:`TimelineBlock` does not contain a
    :class:`Trajectory`. The existing ``block_meta_json`` extra-payload
    channel is reused to carry a per-block ``retune_events`` list for
    ECSV round-trip. Encoding is a list of ``[t_start, duration]`` float
    pairs; on read it decodes back into a tuple of
    :class:`~fyst_trajectories.RetuneEvent` instances.
    """

    def test_retune_events_round_trip_via_science_block_metadata(self, tmp_path):
        """Science block carrying ``retune_events`` survives write/read."""
        from fyst_trajectories import RetuneEvent

        site = get_fyst_site()
        t0 = Time("2026-06-15T02:00:00", scale="utc")

        events = (
            RetuneEvent(t_start=30.0, duration=5.0),
            RetuneEvent(t_start=120.0, duration=3.0),
            RetuneEvent(t_start=200.0, duration=8.0),
        )

        blocks = [
            TimelineBlock(
                t_start=t0,
                t_stop=t0 + TimeDelta(1800, format="sec"),
                block_type="science",
                patch_name="retune_field",
                az_start=120.0,
                az_end=240.0,
                elevation=50.0,
                scan_index=0,
                rising=True,
                scan_type="pong",
                metadata={
                    "ra_center": 180.0,
                    "dec_center": -30.0,
                    "width": 4.0,
                    "height": 4.0,
                    "velocity": 0.5,
                    "scan_params": {"spacing": 0.1, "num_terms": 4},
                    "retune_events": events,
                },
            ),
        ]

        timeline = ObservingTimeline(
            blocks=blocks,
            site=site,
            start_time=t0,
            end_time=t0 + TimeDelta(1800, format="sec"),
            overhead_model=OverheadModel(),
            calibration_policy=CalibrationPolicy(),
        )

        path = tmp_path / "retune_events_rt.ecsv"
        write_timeline(timeline, path)
        loaded = read_timeline(path)

        assert len(loaded.blocks) == 1
        loaded_events = loaded.blocks[0].metadata["retune_events"]
        # Same type (tuple of RetuneEvent), same ordering, same values.
        assert isinstance(loaded_events, tuple)
        assert len(loaded_events) == 3
        for orig, got in zip(events, loaded_events):
            assert isinstance(got, RetuneEvent)
            assert got.t_start == pytest.approx(orig.t_start)
            assert got.duration == pytest.approx(orig.duration)

    def test_block_without_retune_events_roundtrips_clean(self, tmp_path):
        """Blocks without retune_events do not acquire a phantom key on round-trip."""
        timeline = _make_test_timeline()
        path = tmp_path / "no_retune.ecsv"
        write_timeline(timeline, path)
        loaded = read_timeline(path)
        for b in loaded.blocks:
            assert "retune_events" not in b.metadata

    def test_pre_encoded_retune_events_passthrough(self, tmp_path):
        """Writer accepts ``[t_start, duration]`` pairs in addition to ``RetuneEvent``.

        This is the documented robustness path for callers who construct
        block metadata from JSON or other plain-Python sources without
        materialising ``RetuneEvent`` instances. Round-trip must produce
        canonical ``RetuneEvent`` tuples regardless of input shape.
        """
        from fyst_trajectories import RetuneEvent

        site = get_fyst_site()
        t0 = Time("2026-06-15T02:00:00", scale="utc")

        # Caller-supplied pre-encoded payload: list of [t_start, duration] pairs.
        pre_encoded = [[30.0, 5.0], [120.0, 3.0]]

        blocks = [
            TimelineBlock(
                t_start=t0,
                t_stop=t0 + TimeDelta(1800, format="sec"),
                block_type="science",
                patch_name="retune_field",
                az_start=120.0,
                az_end=240.0,
                elevation=50.0,
                scan_index=0,
                rising=True,
                scan_type="pong",
                metadata={"retune_events": pre_encoded},
            ),
        ]

        timeline = ObservingTimeline(
            blocks=blocks,
            site=site,
            start_time=t0,
            end_time=t0 + TimeDelta(1800, format="sec"),
            overhead_model=OverheadModel(),
            calibration_policy=CalibrationPolicy(),
        )

        path = tmp_path / "pre_encoded_rt.ecsv"
        write_timeline(timeline, path)
        loaded = read_timeline(path)

        loaded_events = loaded.blocks[0].metadata["retune_events"]
        assert isinstance(loaded_events, tuple)
        assert len(loaded_events) == 2
        assert all(isinstance(e, RetuneEvent) for e in loaded_events)
        assert loaded_events[0].t_start == pytest.approx(30.0)
        assert loaded_events[0].duration == pytest.approx(5.0)
        assert loaded_events[1].t_start == pytest.approx(120.0)
        assert loaded_events[1].duration == pytest.approx(3.0)
