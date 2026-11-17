"""Tests for ``check_earth_orientation_data``, the IERS coverage check.

The suite pins astropy's Earth-orientation table to the vendored IERS-A file
``tests/data/finals2000A.all`` (see ``conftest.py``), which runs from 1973-01-02 to
2027-07-03. The stale-prediction cases build astropy's own auto-updating table class
from that file, set the host's clock as astropy reads it and keep astropy off the
network, so astropy's own age check judges the table by that clock, whatever the real
date and whatever ``astropy-iers-data`` is installed.
"""

import contextlib
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time
from astropy.utils import iers

import fyst_trajectories
from fyst_trajectories import check_earth_orientation_data
from fyst_trajectories.exceptions import PointingError

VENDORED_TABLE = Path(__file__).parent / "data" / "finals2000A.all"
MISSING_IERS_FILE = (Path(__file__).parent / "data" / "no_such_iers_file.all").as_uri()
TABLE_START = "1973-01-02 00:00:00.000"
TABLE_END = "2027-07-03 00:00:00.000"
STALE_REASON = (
    "interpolating from IERS_Auto using predictive values that are more than 30.0 days old"
)
# The stale-prediction span, inside the predictions, and the two clocks it is judged by,
# each counted from the table's first predicted instant.
SPAN_AFTER_PREDICTIONS = 11 * u.day
SPAN_DAYS = 2.0
FRESH_CLOCK = 10 * u.day
STALE_CLOCK = 60 * u.day


def _iers_state():
    return (
        iers.conf.auto_download,
        iers.conf.auto_max_age,
        iers.conf.iers_degraded_accuracy,
        iers.earth_orientation_table.get(),
    )


def _auto_table():
    """Return astropy's IERS_Auto table read from the vendored file and its first prediction."""
    table = iers.IERS_Auto.read(file=str(VENDORED_TABLE))
    predictions_from = Time(table.meta["predictive_mjd"], format="mjd", scale="utc")
    return table, predictions_from


@contextlib.contextmanager
def _host_clock(instant):
    """Make ``Time.now()`` return ``instant`` inside the block, as on a host whose clock reads it.

    Astropy judges an IERS_Auto table's age by ``Time.now()``, so its age check and its
    refresh decision follow this clock. The original is restored when the block exits,
    a failing test included.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Time, "now", classmethod(lambda cls: instant))
        yield


@contextlib.contextmanager
def _offline():
    """Keep astropy off the network as on a host that cannot reach the IERS servers.

    Astropy 6 and earlier refresh a stale IERS_Auto table even with auto_download off,
    so the download URLs point at a file that does not exist.
    """
    with (
        iers.conf.set_temp("auto_download", False),
        iers.conf.set_temp("iers_auto_url", MISSING_IERS_FILE),
        iers.conf.set_temp("iers_auto_url_mirror", MISSING_IERS_FILE),
    ):
        yield


class TestCoveredSpan:
    """A span inside the table returns where the table ends."""

    @pytest.mark.parametrize("scale", ["utc", "tt"])
    def test_a_span_inside_the_table_returns_the_tables_end(self, scale):
        start = Time("2026-03-15T04:00:00", scale=scale)
        covered_until = check_earth_orientation_data(start, days=2.0)
        assert covered_until.scale == "utc"
        assert covered_until.iso == TABLE_END

    def test_the_tables_last_row_is_not_covered(self):
        # Astropy interpolates only between rows, so the last row's instant is outside.
        end = Time(TABLE_END, scale="utc")
        check_earth_orientation_data(end - 1 * u.s)
        with pytest.raises(PointingError, match="does not cover"):
            check_earth_orientation_data(end)


class TestUncoveredSpan:
    """A span the table does not reach is a PointingError naming both."""

    @pytest.mark.parametrize(
        ("start", "days", "named"),
        [
            ("1970-01-01T00:00:00", 0.0, "1970-01-01 00:00:00.000"),
            ("2027-08-01T00:00:00", 0.0, "2027-08-01 00:00:00.000"),
            ("2027-06-30T12:00:00", 10.0, "2027-06-30 12:00:00.000 to 2027-07-10 12:00:00.000"),
        ],
        ids=["starts_before", "starts_beyond", "ends_beyond"],
    )
    def test_names_the_span_and_the_tables_range(self, start, days, named):
        with pytest.raises(PointingError) as excinfo:
            check_earth_orientation_data(Time(start, scale="utc"), days=days)
        message = str(excinfo.value)
        assert f"does not cover {named}:" in message
        assert f"runs from {TABLE_START} to {TABLE_END}" in message
        assert "astropy-iers-data" not in message

    def test_a_start_in_another_scale_is_named_in_utc(self):
        start = Time("2027-08-01T00:00:00", scale="tt")
        with pytest.raises(PointingError) as excinfo:
            check_earth_orientation_data(start)
        assert f"does not cover {start.utc.iso}:" in str(excinfo.value)

    def test_a_single_instant_is_named_once(self):
        with pytest.raises(PointingError) as excinfo:
            check_earth_orientation_data(Time("2027-08-01T00:00:00", scale="utc"))
        assert str(excinfo.value).count("2027-08-01") == 1

    @pytest.mark.parametrize(
        "instant",
        ["1973-01-01T23:59:59", "1973-01-02T00:00:00", "2027-07-02T23:59:59", TABLE_END],
    )
    def test_agrees_with_astropys_own_range_check(self, instant):
        # Under astropy's default "error" setting, Time.ut1 refuses exactly the instants
        # the check refuses; a coordinate transform there warns and takes UT1-UTC as zero
        # instead.
        start = Time(instant, scale="utc")
        with iers.conf.set_temp("iers_degraded_accuracy", "error"):
            try:
                start.ut1  # noqa: B018
                astropy_refuses = False
            except iers.IERSRangeError:
                astropy_refuses = True
            try:
                check_earth_orientation_data(start)
                check_refuses = False
            except PointingError:
                check_refuses = True
        assert check_refuses == astropy_refuses


# Astropy 6 and earlier try the missing file and warn that the download failed.
@pytest.mark.filterwarnings("ignore:failed to download:astropy.utils.exceptions.AstropyWarning")
class TestStalePredictions:
    """Astropy's refusal of old predictions reaches the caller as a PointingError."""

    # The first two tests differ only in the clock: the same table and span, judged 10
    # and 60 days after the first prediction.

    def test_astropys_refusal_becomes_a_pointing_error_with_its_reason(self):
        table, predictions_from = _auto_table()
        start = predictions_from + SPAN_AFTER_PREDICTIONS
        end = start + SPAN_DAYS * u.day
        with (
            _offline(),
            _host_clock(predictions_from + STALE_CLOCK),
            iers.earth_orientation_table.set(table),
        ):
            with pytest.raises(ValueError) as astropy_refusal:
                start.ut1  # noqa: B018
            with pytest.raises(PointingError) as excinfo:
                check_earth_orientation_data(start, days=SPAN_DAYS)
        assert type(astropy_refusal.value) is ValueError
        assert " ".join(str(astropy_refusal.value).split()).startswith(STALE_REASON)
        message = str(excinfo.value)
        named = f"{start.utc.iso} to {end.utc.iso}"
        assert f"refuses its Earth-orientation (IERS) table for {named}: " in message
        assert STALE_REASON in message
        assert type(excinfo.value.__cause__) is ValueError

    def test_predictions_within_astropys_age_limit_are_accepted(self):
        table, predictions_from = _auto_table()
        start = predictions_from + SPAN_AFTER_PREDICTIONS
        with (
            _offline(),
            _host_clock(predictions_from + FRESH_CLOCK),
            iers.earth_orientation_table.set(table),
        ):
            covered_until = check_earth_orientation_data(start, days=SPAN_DAYS)
        assert covered_until.iso == TABLE_END

    def test_a_span_that_reaches_the_predictions_is_judged_at_its_end(self):
        table, predictions_from = _auto_table()
        with (
            _offline(),
            _host_clock(predictions_from + STALE_CLOCK),
            iers.earth_orientation_table.set(table),
        ):
            check_earth_orientation_data(predictions_from - 2 * u.day, days=1.0)
            with pytest.raises(PointingError, match="refuses"):
                check_earth_orientation_data(predictions_from - 1 * u.day, days=2.0)


class TestAstropysOwnErrorsPassThrough:
    """An error astropy raises before the coverage is known reaches the caller unchanged."""

    def test_an_invalid_auto_max_age_is_astropys_value_error(self):
        # Astropy checks auto_max_age only where it may download, so downloads stay on
        # here, from a file that does not exist.
        table, predictions_from = _auto_table()
        with (
            _host_clock(predictions_from + FRESH_CLOCK),
            iers.conf.set_temp("auto_download", True),
            iers.conf.set_temp("iers_auto_url", MISSING_IERS_FILE),
            iers.conf.set_temp("iers_auto_url_mirror", MISSING_IERS_FILE),
            iers.conf.set_temp("auto_max_age", 5.0),
            iers.earth_orientation_table.set(table),
            pytest.raises(ValueError, match="auto_max_age") as excinfo,
        ):
            check_earth_orientation_data(predictions_from + 11 * u.day)
        assert type(excinfo.value) is ValueError

    def test_a_start_in_ut1_beyond_the_table_is_astropys_range_error(self):
        start = Time("2027-08-01T00:00:00", scale="ut1")
        with (
            iers.conf.set_temp("iers_degraded_accuracy", "error"),
            pytest.raises(iers.IERSRangeError),
        ):
            check_earth_orientation_data(start)


class TestMalformedRequest:
    """A malformed request is a plain ValueError or TypeError."""

    @pytest.mark.parametrize("days", [-1.0, float("nan"), float("inf"), True])
    def test_days_must_be_finite_and_non_negative(self, days):
        with pytest.raises(ValueError, match="days") as excinfo:
            check_earth_orientation_data(Time("2026-03-15T04:00:00", scale="utc"), days=days)
        assert not isinstance(excinfo.value, PointingError)

    def test_start_must_be_a_single_instant(self):
        start = Time(["2026-03-15T04:00:00", "2026-03-16T04:00:00"], scale="utc")
        with pytest.raises(ValueError, match="single instant") as excinfo:
            check_earth_orientation_data(start)
        assert not isinstance(excinfo.value, PointingError)

    def test_start_must_be_a_time(self):
        with pytest.raises(TypeError, match="astropy Time"):
            check_earth_orientation_data("2026-03-15T04:00:00")  # type: ignore[arg-type]


class TestLeavesAstropyAsItWas:
    """The check leaves astropy's configuration and the active table as they were."""

    def test_configuration_and_active_table_are_unchanged(self):
        before = _iers_state()
        check_earth_orientation_data(Time("2026-03-15T04:00:00", scale="utc"), days=2.0)
        with pytest.raises(PointingError):
            check_earth_orientation_data(Time("2027-08-01T00:00:00", scale="utc"))
        after = _iers_state()
        assert after[:3] == before[:3]
        assert after[3] is before[3]
        assert len(after[3]) == len(before[3])


def test_exported_from_the_package_root():
    assert (
        fyst_trajectories.check_earth_orientation_data
        is fyst_trajectories.coordinates.check_earth_orientation_data
    )
    assert "check_earth_orientation_data" in fyst_trajectories.__all__


def test_the_suites_table_is_the_vendored_file():
    # The expectations above read the vendored table's range; fail loudly if the pin moved.
    table = iers.earth_orientation_table.get()
    assert type(table) is iers.IERS_A
    mjd = table["MJD"].to_value(u.day)
    assert Time(mjd[[0, -1]], format="mjd", scale="utc").iso.tolist() == [TABLE_START, TABLE_END]
    assert np.all(np.diff(mjd) == 1.0)
