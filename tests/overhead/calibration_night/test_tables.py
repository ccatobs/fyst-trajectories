"""Tests for the per-body scan-parameter tables."""

import math

import pytest

from fyst_trajectories.overhead import (
    DEFAULT_SCAN_TABLES,
    ElevationBin,
    ScanParameterTable,
    load_scan_tables,
)
from fyst_trajectories.overhead.calibration_night.tables import table_for


def _bin(lo, hi, throw=2.44, dwell=600.0):
    return ElevationBin(lo, hi, az_throw=throw, dwell_reference=dwell)


class TestElevationBin:
    """Bounds and positivity are enforced at construction."""

    def test_centre(self):
        assert _bin(30.0, 35.0).el_centre == 32.5

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(el_lo=35.0, el_hi=30.0), "el_lo must be below el_hi"),
            (dict(az_throw=0.0), "az_throw must be positive"),
            (dict(dwell_reference=-1.0), "dwell_reference must be positive"),
            (dict(el_lo=float("nan")), "el_lo must be finite"),
        ],
    )
    def test_rejections(self, kwargs, match):
        base = dict(el_lo=30.0, el_hi=35.0, az_throw=2.44, dwell_reference=600.0)
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            ElevationBin(**base)


class TestScanParameterTable:
    """Contiguity, lookup, extrapolation."""

    def test_sorts_bins_and_reports_the_range(self):
        table = ScanParameterTable((_bin(35.0, 40.0), _bin(30.0, 35.0)))
        assert [b.el_lo for b in table.bins] == [30.0, 35.0]
        assert table.el_range == (30.0, 40.0)

    def test_empty_overlap_and_gap_raise(self):
        with pytest.raises(ValueError, match="must not be empty"):
            ScanParameterTable(())
        with pytest.raises(ValueError, match="overlap"):
            ScanParameterTable((_bin(30.0, 36.0), _bin(35.0, 40.0)))
        with pytest.raises(ValueError, match="gap"):
            ScanParameterTable((_bin(30.0, 34.0), _bin(35.0, 40.0)))

    def test_for_elevation_boundaries(self):
        table = ScanParameterTable((_bin(30.0, 35.0, throw=1.0), _bin(35.0, 40.0, throw=2.0)))
        assert table.for_elevation(29.9) is None
        assert table.for_elevation(30.0).az_throw == 1.0
        assert table.for_elevation(35.0).az_throw == 2.0  # a lower bound belongs to the upper bin
        assert table.for_elevation(40.0).az_throw == 2.0  # the top bin includes its upper bound
        assert table.for_elevation(40.1) is None

    def test_az_throw_extrapolates_at_constant_sky_width_above_the_top_bin(self):
        table = ScanParameterTable((_bin(45.0, 50.0, throw=3.11),))
        assert table.az_throw_at(47.0) == 3.11
        expected = 3.11 * math.cos(math.radians(47.5)) / math.cos(math.radians(62.0))
        assert table.az_throw_at(62.0) == pytest.approx(expected)
        with pytest.raises(ValueError, match="below the table"):
            table.az_throw_at(20.0)


class TestDefaultTables:
    """The shipped instrument-team defaults, as transcribed."""

    def test_shared_table(self):
        shared = DEFAULT_SCAN_TABLES["default"]
        assert shared.el_range == (30.0, 50.0)
        assert [b.az_throw for b in shared.bins] == [2.44, 2.61, 2.83, 3.11]
        assert [b.dwell_reference / 60.0 for b in shared.bins] == [10.0, 10.0, 13.0, 15.0]

    def test_uranus_table(self):
        uranus = DEFAULT_SCAN_TABLES["uranus"]
        assert uranus.el_range == (30.0, 40.0)
        assert [b.az_throw for b in uranus.bins] == [2.44, 2.61]
        assert {b.dwell_reference for b in uranus.bins} == {900.0}

    def test_table_for_falls_back_to_default(self):
        assert table_for(DEFAULT_SCAN_TABLES, "Saturn") is DEFAULT_SCAN_TABLES["default"]
        assert table_for(DEFAULT_SCAN_TABLES, "URANUS") is DEFAULT_SCAN_TABLES["uranus"]
        with pytest.raises(KeyError, match="no 'default' table"):
            table_for({"uranus": DEFAULT_SCAN_TABLES["uranus"]}, "saturn")

    def test_widths_imply_a_constant_sky_extent(self):
        """The widths divided by 1/cos(el) are one on-sky width, near 2.08 deg."""
        shared = DEFAULT_SCAN_TABLES["default"]
        sky = [b.az_throw * math.cos(math.radians(b.el_centre)) for b in shared.bins]
        assert max(sky) - min(sky) < 0.05
        assert sum(sky) / len(sky) == pytest.approx(2.08, abs=0.02)


class TestLoadScanTables:
    """The CSV loader accepts the table's own headings."""

    def test_loads_a_file_with_the_source_table_headings(self, tmp_path):
        path = tmp_path / "tables.csv"
        path.write_text(
            "Body,el_lo,el_hi,Scan Time,Scan Width (deg az)\n"
            "default,30,35,10 min,2.44\n"
            "default,35,40,10 min,2.61\n"
            "Uranus,30,35,15 min,2.44\n"
            "uranus,35,40,15,2.61\n",
            encoding="utf-8",
        )
        tables = load_scan_tables(path)
        assert set(tables) == {"default", "uranus"}
        assert tables["default"].bins[1].az_throw == 2.61
        assert tables["default"].bins[0].dwell_reference == 600.0
        assert tables["uranus"].el_range == (30.0, 40.0)

    def test_missing_column_and_bad_value(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("body,el_lo,el_hi,scan_time_min\nx,30,35,10\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing a 'az_throw' column"):
            load_scan_tables(path)
        path.write_text(
            "body,el_lo,el_hi,scan_time_min,az_throw\nx,30,35,ten,2.4\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="cannot read scan time"):
            load_scan_tables(path)

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("body,el_lo,el_hi,scan_time_min,az_throw\n", encoding="utf-8")
        with pytest.raises(ValueError, match="holds no table rows"):
            load_scan_tables(path)
