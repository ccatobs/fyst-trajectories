"""Tests for TOAST-compatible ECSV I/O."""

import dataclasses
import re

import pytest
from astropy.table import QTable, Table
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
    """A timeline with no blocks still writes."""

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
    """Block count, types, times, scan types and timeline metadata survive a read."""

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
            # ISO strings carry milliseconds; this timeline's times are whole seconds.
            assert abs(orig.t_start.unix - loaded_b.t_start.unix) < 1e-3
            assert abs(orig.t_stop.unix - loaded_b.t_stop.unix) < 1e-3

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


class TestTimelineWindowRoundTrip:
    """The declared timeline window (start_time/end_time) survives a round-trip."""

    def _padded_timeline(self):
        """Build a timeline whose window pads past the block extents on both ends.

        A single one-hour science block sits in the middle of a two-hour
        declared window with idle-free padding at each edge, so ``total_time``
        (and thus ``efficiency``) depends on the persisted window rather than
        the block extents.
        """
        site = get_fyst_site()
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        blocks = [
            TimelineBlock(
                t_start=t0 + TimeDelta(1800, format="sec"),
                t_stop=t0 + TimeDelta(5400, format="sec"),
                block_type="science",
                patch_name="deep_field",
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
                    "scan_params": {},
                },
            ),
        ]
        return ObservingTimeline(
            blocks=blocks,
            site=site,
            start_time=t0,
            end_time=t0 + TimeDelta(7200, format="sec"),
            overhead_model=OverheadModel(),
            calibration_policy=CalibrationPolicy(),
        )

    def test_window_and_efficiency_survive_round_trip(self, tmp_path):
        timeline = self._padded_timeline()
        assert timeline.total_time == pytest.approx(7200.0)
        assert timeline.efficiency == pytest.approx(0.5)

        path = tmp_path / "padded.ecsv"
        write_timeline(timeline, path)
        loaded = read_timeline(path)

        assert loaded.start_time.unix == pytest.approx(timeline.start_time.unix, abs=1e-3)
        assert loaded.end_time.unix == pytest.approx(timeline.end_time.unix, abs=1e-3)
        assert loaded.total_time == pytest.approx(7200.0, abs=1.0)
        assert loaded.efficiency == pytest.approx(0.5, abs=1e-3)

    def test_older_file_falls_back_to_block_extents(self, tmp_path):
        timeline = self._padded_timeline()
        path = tmp_path / "padded.ecsv"
        write_timeline(timeline, path)

        # Simulate an older file written before the window keys existed by
        # stripping them from the table metadata.
        table = Table.read(str(path), format="ascii.ecsv")
        del table.meta["timeline_start_time"]
        del table.meta["timeline_end_time"]
        legacy_path = tmp_path / "legacy.ecsv"
        table.write(str(legacy_path), format="ascii.ecsv", overwrite=True)

        loaded = read_timeline(legacy_path)
        # The window collapses onto the block extents: the one-hour block span.
        assert loaded.total_time == pytest.approx(3600.0, abs=1.0)
        assert loaded.efficiency == pytest.approx(1.0, abs=1e-3)


class TestTimesAreWrittenInUtc:
    """The file holds every time in UTC, the scale ``read_timeline`` reads them in."""

    @pytest.mark.parametrize("scale", ["tt", "tai", "tdb", "ut1"])
    def test_a_timeline_held_in_another_scale_reads_back_at_its_instants(self, scale, tmp_path):
        utc = _make_test_timeline()
        given = dataclasses.replace(
            utc,
            blocks=[
                dataclasses.replace(
                    b, t_start=getattr(b.t_start, scale), t_stop=getattr(b.t_stop, scale)
                )
                for b in utc.blocks
            ],
            start_time=getattr(utc.start_time, scale),
            end_time=getattr(utc.end_time, scale),
        )
        path = tmp_path / f"{scale}.ecsv"
        write_timeline(given, path)
        back = read_timeline(path)

        def off(a, b):
            return abs((a - b).to_value("s"))

        # The file stores times to the millisecond.
        assert off(back.start_time, utc.start_time) <= 5e-4
        assert off(back.end_time, utc.end_time) <= 5e-4
        for block, original in zip(back.blocks, utc.blocks, strict=True):
            assert off(block.t_start, original.t_start) <= 5e-4
            assert off(block.t_stop, original.t_stop) <= 5e-4

    @pytest.mark.parametrize(
        "attributes", [{"precision": 0}, {"out_subfmt": "date"}], ids=["precision 0", "date"]
    )
    def test_a_utc_timeline_with_other_output_attributes_writes_millisecond_strings(
        self, attributes, tmp_path
    ):
        """The times are written to the millisecond whatever the ``Time`` objects print.

        Every time is moved 0.3714 s off the whole second, so a string to
        the second, or a date alone, would read back at another instant.
        """
        shift = TimeDelta(0.3714, format="sec")

        def given(t):
            return Time(t + shift, **attributes)

        utc = _make_test_timeline()
        timeline = dataclasses.replace(
            utc,
            blocks=[
                dataclasses.replace(b, t_start=given(b.t_start), t_stop=given(b.t_stop))
                for b in utc.blocks
            ],
            start_time=given(utc.start_time),
            end_time=given(utc.end_time),
        )
        path = tmp_path / "attributes.ecsv"
        write_timeline(timeline, path)

        table = Table.read(path, format="ascii.ecsv")
        millisecond = re.compile(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3}")
        strings = [
            *table["start_time"],
            *table["stop_time"],
            table.meta["timeline_start_time"],
            table.meta["timeline_end_time"],
        ]
        assert all(millisecond.fullmatch(str(s)) for s in strings), strings
        back = read_timeline(path)
        for block, original in zip(back.blocks, utc.blocks, strict=True):
            assert abs((block.t_start - (original.t_start + shift)).to_value("s")) <= 5e-4
            assert abs((block.t_stop - (original.t_stop + shift)).to_value("s")) <= 5e-4


class TestCanonicalColumnNames:
    """New writes use the TOAST names and ISO times, never the legacy MJD/scan ones."""

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
        assert first == "2026-06-15 02:00:00.000"


class TestEmptyTimelineRoundTrip:
    """A timeline with no blocks reads back with no blocks."""

    def test_the_placeholder_row_does_not_become_a_block(self, tmp_path):
        """ECSV needs a row; the timeline it stands for still has none.

        A planner that finds nothing observable inside its window returns
        an empty timeline, which is a documented outcome. The file needs
        a placeholder row, but reading that row back as a real idle block
        dated 2000-01-01 would fail the library's own ``validate()`` and
        put a fabricated entry into any rendering. The header marks the
        file empty and the reader drops the row.
        """
        site = get_fyst_site()
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        timeline = ObservingTimeline(
            blocks=[],
            site=site,
            start_time=t0,
            end_time=t0 + TimeDelta(3600, format="sec"),
            overhead_model=OverheadModel(),
            calibration_policy=CalibrationPolicy(),
        )
        path = tmp_path / "empty.ecsv"
        write_timeline(timeline, path)

        # The file itself carries the placeholder row and says so.
        table = Table.read(str(path), format="ascii.ecsv")
        assert len(table) == 1
        assert table.meta["timeline_is_empty"] is True

        reloaded = read_timeline(path)
        assert reloaded.blocks == []
        assert reloaded.validate() == []
        assert reloaded.start_time.iso == timeline.start_time.iso
        assert reloaded.end_time.iso == timeline.end_time.iso
        # The marker is a header field, not stray user metadata.
        assert "timeline_is_empty" not in reloaded.metadata


class TestAzFinalColumn:
    """The end-pose column: written, read back, and absent in older files.

    ``az_final`` records where a swept block leaves the telescope when
    that differs from the envelope bound ``azmax``. It is a FYST
    extension column, so a file written before it existed must still
    load, with every block reading back as ``None``.
    """

    def _swept_timeline(self):
        site = get_fyst_site()
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        swept = TimelineBlock(
            t_start=t0,
            t_stop=t0 + TimeDelta(600, format="sec"),
            block_type="calibration",
            patch_name="planet_cal",
            az_start=100.0,
            az_end=200.0,
            elevation=50.0,
            scan_index=0,
            scan_type="planet_cal",
            az_final=120.5,
        )
        parked = TimelineBlock(
            t_start=t0 + TimeDelta(600, format="sec"),
            t_stop=t0 + TimeDelta(900, format="sec"),
            block_type="idle",
            patch_name="no_target",
            az_start=120.5,
            az_end=120.5,
            elevation=50.0,
            scan_index=0,
            scan_type="idle",
        )
        return ObservingTimeline(
            blocks=[swept, parked],
            site=site,
            start_time=t0,
            end_time=t0 + TimeDelta(900, format="sec"),
            overhead_model=OverheadModel(),
            calibration_policy=CalibrationPolicy(),
        )

    def test_round_trips_and_leaves_the_envelope_alone(self, tmp_path):
        timeline = self._swept_timeline()
        path = tmp_path / "az_final.ecsv"
        write_timeline(timeline, path)

        table = Table.read(str(path), format="ascii.ecsv")
        assert "az_final" in table.colnames
        assert float(table["azmin"][0]) == 100.0
        assert float(table["azmax"][0]) == 200.0

        reloaded = read_timeline(path)
        swept, parked = reloaded.blocks
        assert swept.az_final == pytest.approx(120.5)
        assert swept.end_pose_az == pytest.approx(120.5)
        assert (swept.az_start, swept.az_end) == (100.0, 200.0)
        # A block that ends where its envelope does writes NaN and reads
        # back as None, not 0.0.
        assert parked.az_final is None
        assert parked.end_pose_az == 120.5
        assert reloaded.validate() == []

    def test_a_file_without_the_column_still_loads(self, tmp_path):
        """Drop the column the way a writer without ``az_final`` does."""
        timeline = self._swept_timeline()
        path = tmp_path / "az_final.ecsv"
        legacy = tmp_path / "legacy.ecsv"
        write_timeline(timeline, path)

        table = QTable.read(str(path), format="ascii.ecsv")
        table.remove_column("az_final")
        table.write(str(legacy), format="ascii.ecsv", overwrite=True)

        reloaded = read_timeline(legacy)
        assert all(b.az_final is None for b in reloaded.blocks)
        # Such a file records only the envelope, so the swept block's end
        # pose is its envelope bound.
        assert reloaded.blocks[0].end_pose_az == 200.0


class TestMetadataPersistence:
    """Per-block science metadata must survive an ECSV round-trip."""

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

    @pytest.mark.filterwarnings(
        "ignore:Trajectory (azimuth|elevation) acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
    def test_metadata_roundtrip_through_simulation_bridge(self, tmp_path):
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
            # The patch's own keys, plus the whole periods the subscan holds.
            n_cycles = b.metadata["scan_params"]["n_cycles"]
            assert isinstance(n_cycles, int)
            assert n_cycles >= 1
            assert b.metadata["scan_params"] == {
                "spacing": 0.12,
                "num_terms": 5,
                "n_cycles": n_cycles,
            }

        # The simulation bridge should succeed end-to-end on the loaded
        # timeline and reproduce the same pong geometry (width/height) as
        # the in-memory timeline, not a hardcoded 180/-30/4x4 fallback.
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
    """Nasmyth port, plate scale, sun radii and description persist; limits do not."""

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

    @pytest.mark.parametrize(
        "written_site",
        [
            dataclasses.replace(get_fyst_site(), nasmyth_port="left"),
            get_fyst_site(sun_avoidance_enabled=False),
            get_fyst_site(sun_exclusion_radius=50.0, sun_warning_radius=55.0),
        ],
        ids=["left_port", "sun_avoidance_off", "sun_radii_50_55"],
    )
    def test_fyst_coordinates_keep_the_persisted_fields(self, tmp_path, written_site):
        """A FYST-coordinate site reads back with the fields it was written with.

        The reader takes the limits from the FYST default for these
        coordinates, so the Nasmyth port and the Sun settings are the
        fields a default-only round trip cannot see being dropped.
        """
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
            site=written_site,
            start_time=t0,
            end_time=t0 + TimeDelta(60, format="sec"),
            overhead_model=OverheadModel(),
            calibration_policy=CalibrationPolicy(),
        )
        path = tmp_path / "fyst_persisted.ecsv"
        write_timeline(timeline, path)
        loaded = read_timeline(path)
        assert loaded.site == written_site

    def test_custom_site_coordinates_preserved(self, tmp_path):
        """A non-FYST site round-trips from metadata, not replaced.

        Without this, a custom ``nasmyth_port="left"`` reads back as the FYST
        default ``"right"``, silently flipping the field-rotation sign;
        ``nasmyth_port``/``plate_scale``/sun radii must persist. ``telescope_limits``
        are not persisted, so a ``PointingWarning`` is emitted on read.
        """
        from fyst_trajectories.exceptions import PointingWarning
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
            telescope_limits=TelescopeLimits(
                azimuth=AxisLimits(min=-180.0, max=360.0, max_velocity=3.0, max_acceleration=1.0),
                elevation=AxisLimits(min=20.0, max=90.0, max_velocity=1.0, max_acceleration=0.5),
            ),
            sun_avoidance=SunAvoidanceConfig(
                enabled=True, exclusion_radius=40.0, warning_radius=48.0
            ),
            nasmyth_port="left",
            plate_scale=10.0,
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
        with pytest.warns(PointingWarning, match="telescope_limits"):
            loaded = read_timeline(path)
        assert loaded.site.latitude == pytest.approx(-30.0)
        assert loaded.site.longitude == pytest.approx(-70.0)
        assert loaded.site.elevation == pytest.approx(2500.0)
        # Without this, these fields reset to FYST defaults (silent lossy round-trip).
        assert loaded.site.nasmyth_port == "left"
        assert loaded.site.plate_scale == pytest.approx(10.0)
        assert loaded.site.sun_avoidance.exclusion_radius == pytest.approx(40.0)
        assert loaded.site.sun_avoidance.warning_radius == pytest.approx(48.0)

    def test_site_description_round_trips_without_accumulation(self, tmp_path):
        """Description round-trips, telescope_name reflects the site, no metadata leak."""
        from fyst_trajectories.exceptions import PointingWarning
        from fyst_trajectories.site import (
            AxisLimits,
            Site,
            SunAvoidanceConfig,
            TelescopeLimits,
        )

        custom_site = Site(
            name="myobs",
            description="My Test Observatory",
            latitude=-30.0,
            longitude=-70.0,
            elevation=2500.0,
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
        path = tmp_path / "desc_site.ecsv"
        write_timeline(timeline, path)
        with pytest.warns(PointingWarning, match="Reconstructing a non-FYST Site"):
            loaded = read_timeline(path)

        # Description round-trips; without this it is silently dropped.
        assert loaded.site.description == "My Test Observatory"
        # telescope_name reflects the site, not a hardcoded "FYST".
        assert Table.read(str(path), format="ascii.ecsv").meta["telescope_name"] == "myobs"
        # site_description is structural, not leaked into user metadata.
        assert "site_description" not in loaded.metadata

        # Second write/read cycle: no metadata accumulation; description stable.
        path2 = tmp_path / "desc_site2.ecsv"
        write_timeline(loaded, path2)
        with pytest.warns(PointingWarning, match="Reconstructing a non-FYST Site"):
            loaded2 = read_timeline(path2)
        assert loaded2.site.description == "My Test Observatory"
        assert "site_description" not in loaded2.metadata


class TestBoresightAngle:
    """``boresight_angle`` round-trips through ECSV."""

    def test_boresight_roundtrip(self, tmp_path):
        site = get_fyst_site()
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        # Construct a block with a known nonzero boresight_angle.
        az = 120.0
        el = 55.0
        expected = Coordinates(site).get_field_rotation_from_altaz(az, el)
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


class TestOverheadModelRoundTrip:
    """All OverheadModel fields must survive an ECSV write/read round-trip."""

    def test_all_overhead_fields_round_trip(self, tmp_path):
        """Every OverheadModel field must be restored from ECSV metadata.

        Constructs every field with a value distinct from the class
        default (and distinct from every *other* field's default) so a
        future field added without the corresponding I/O wiring fails
        loudly instead of coincidentally matching a default on the read
        side.
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
        field defaults to ``None``, so it is given a non-None value here
        and an I/O path not wired to it fails loudly.
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
            planet_cal_scan=True,  # non-default (default False)
            planet_cal_passes=5,  # non-default (default 3)
            planet_cal_el_step=1.5,  # non-default (default None) to catch dropped serialisation
            planet_cal_footprint="i1",  # non-default (default "c")
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


class TestRetuneEventsRoundTrip:
    """``retune_events`` carried on science block metadata round-trips.

    ``Trajectory.retune_events`` is the canonical home for event-level
    retune provenance, but :class:`TimelineBlock` does not contain a
    :class:`Trajectory`. The existing ``block_meta_json`` extra-payload
    channel is reused to carry a per-block ``retune_events`` list for
    ECSV round-trip. Encoding is a list of ``[t_start, duration]`` float
    pairs; on read it decodes back into a tuple of
    :class:`~fyst_trajectories.RetuneEvent` instances.
    """

    def test_retune_events_round_trip_via_science_block_metadata(self, tmp_path):
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


class TestToastDegUnits:
    """A fyst-written ECSV carries deg units, so TOAST GroundSchedule reads it.

    TOAST v5's ``_read_v5`` builds a ``GroundScan`` per row and its
    ``__init__`` immediately calls ``az_min.to_value(u.degree)`` on the
    azmin/azmax/el/boresight_angle columns. Bare unit-less floats make that
    raise (a ``Column`` has no ``.to_value``). We reproduce that unit-consuming
    step with a ``QTable`` (TOAST reads the ECSV as a QTable) instead of
    importing ``toast``.
    """

    def test_angle_columns_carry_deg_units(self, tmp_path):
        from astropy import units as u
        from astropy.table import QTable

        t0 = Time("2026-06-15T02:00:00", scale="utc")
        timeline = ObservingTimeline(
            blocks=[
                TimelineBlock(
                    t_start=t0,
                    t_stop=t0 + TimeDelta(60, format="sec"),
                    block_type="science",
                    patch_name="patchA",
                    az_start=120.0,
                    az_end=140.0,
                    elevation=55.0,
                    scan_index=0,
                )
            ],
            site=get_fyst_site(),
            start_time=t0,
            end_time=t0 + TimeDelta(60, format="sec"),
            overhead_model=OverheadModel(),
            calibration_policy=CalibrationPolicy(),
        )
        path = tmp_path / "toast.ecsv"
        write_timeline(timeline, path)

        qt = QTable.read(str(path), format="ascii.ecsv")
        for col in ("azmin", "azmax", "el", "boresight_angle"):
            assert qt[col].unit == u.deg, f"column {col} missing deg unit"
            # The exact call TOAST's GroundScan.__init__ makes; must not raise.
            vals = qt[col].to_value(u.degree)
            assert len(vals) >= 1

        # Site coordinates persisted as Quantities (deg/deg/m).
        assert isinstance(qt.meta["site_lat"], u.Quantity)
        assert qt.meta["site_lat"].to_value(u.deg) == pytest.approx(get_fyst_site().latitude)
        assert qt.meta["site_alt"].to_value(u.m) == pytest.approx(get_fyst_site().elevation)
