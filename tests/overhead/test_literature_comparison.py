"""Literature comparison tests for the overhead model.

Validates that the overhead-model defaults are consistent with
published values from operational KID-based telescopes and CMB
survey instruments.

Two retune numbers live in this library and they model different
operations. ``DEFAULT_RETUNE_DURATION_SEC`` (a few seconds) is the
in-scan tone-correction gap that ``inject_retune`` stamps into a
trajectory; NIKA2's KID tuning, under 2 s, is its closest counterpart.
``OverheadModel.retune_duration`` (minutes) is the whole-array retune
reserved between scan blocks, probe-tone placement followed by a
target sweep across every module, the instrument team's commissioning
estimate. The tests below compare each number to its own counterpart.

References
----------
- NIKA2: Perotto et al. 2020, "Calibration and performance of the NIKA2
  camera at the IRAM 30-m Telescope", A&A 637, A71,
  doi:10.1051/0004-6361/201936220; Adam et al. 2018, "The NIKA2
  large-field-of-view millimetre continuum camera for the 30 m IRAM
  telescope", A&A 609, A115, doi:10.1051/0004-6361/201731503. The KID
  tone-matching "tuning" completes in under 2 s (Adam et al. 2018,
  sec. 3.2), in a dedicated sub-scan at each scan boundary. Pointing is
  monitored hourly (Perotto et al. 2020, sec. 3.2) and focus measured
  every other hour during daytime (sec. 3.1).
- Simons Observatory: SO schedlib. TES bias steps default to a 30 min
  cadence and take ~60 s; detector setup runs at block start. A bias step
  is a TES operation with no KID analogue, so it is not compared here.
"""

from fyst_trajectories.overhead import CalibrationPolicy, OverheadModel
from fyst_trajectories.retune import DEFAULT_RETUNE_DURATION_SEC


class TestRetuneVsLiterature:
    """Validate the two retune durations against their own counterparts."""

    def test_in_scan_gap_conservative_vs_nika2_tuning(self):
        """The in-scan gap (5 s) is conservative vs NIKA2's tuning (<2 s).

        NIKA2's KID tone-matching "tuning" completes in under 2 s.
        The 5 s default provides margin for FYST's larger detector count
        (>100,000 KIDs vs NIKA2's ~3,000) without being excessive. The default is an
        instrument-team placeholder pending on-sky tune timing; the band moves if that
        timing lands outside it.
        """
        nika2_tuning = 2.0  # seconds, upper bound from NIKA2 operations

        assert DEFAULT_RETUNE_DURATION_SEC >= nika2_tuning, (
            f"In-scan retune gap ({DEFAULT_RETUNE_DURATION_SEC}s) should be >= "
            f"NIKA2 tuning ({nika2_tuning}s) given larger KID count"
        )
        assert DEFAULT_RETUNE_DURATION_SEC <= 10.0, (
            f"In-scan retune gap ({DEFAULT_RETUNE_DURATION_SEC}s) seems excessive"
        )

    def test_block_retune_matches_the_instrument_team_estimate(self):
        """The block retune (300 s) is the instrument team's ~5 min estimate.

        A whole-array retune re-places the probe tones and runs a target
        sweep across every module: minutes, not the seconds of an
        in-scan tone correction. Pin the exact default so a silent
        regression is caught, then sanity-check the order of magnitude
        (a seconds-vs-minutes slip in either direction fails the band). The estimate
        awaits on-sky timing, and the pin moves with it.
        """
        model = OverheadModel()

        assert model.retune_duration == 300.0, (
            f"Block retune default changed from 300.0 s to "
            f"{model.retune_duration}s, update this test and the default together"
        )
        assert 60.0 <= model.retune_duration <= 600.0, (
            f"Block retune ({model.retune_duration}s) is not minutes-scale"
        )

    def test_block_retune_and_in_scan_gap_are_independent(self):
        """The two retune numbers model different operations and are not synced.

        A block retune reserves minutes for tone placement plus a target
        sweep; the in-scan gap is a few seconds of tone correction. Their
        equality would mean one was re-synced to the other by mistake.
        """
        model = OverheadModel()

        assert model.retune_duration != DEFAULT_RETUNE_DURATION_SEC
        assert model.retune_duration > 10 * DEFAULT_RETUNE_DURATION_SEC


class TestCalibrationCadencesVsLiterature:
    """Pointing at 1 h and focus at 2 h follow NIKA2's hourly pointing and daytime focus."""

    def test_pointing_cadence_within_literature_range(self):
        """Default pointing cadence is 3600 s (1 h).

        NIKA2 monitors pointing hourly (Perotto et al. 2020, sec. 3.2), which the
        3600 s default matches. Pin the exact default value so a silent
        regression or a stale docstring is caught, then sanity-check it stays
        within the [20 min, 1 h] band. ``1800 s`` remains a reasonable
        commissioning override, passed explicitly.
        """
        policy = CalibrationPolicy()

        # Canonical default, operations-team-owned: the pin moves with that team's decision.
        assert policy.pointing_cadence == 3600.0, (
            f"Pointing cadence default changed from 3600.0 s to "
            f"{policy.pointing_cadence}s, update this test and the default together"
        )
        # Sanity: still within the [20 min, 1 h] band.
        assert 1200.0 <= policy.pointing_cadence <= 3600.0

    def test_focus_cadence_matches_nika2(self):
        """Default focus cadence is 7200 s (2 h), NIKA2's daytime focus cadence.

        NIKA2 measures focus every other hour during daytime (Perotto et al.
        2020, sec. 3.1). The default is an operations-team placeholder, so
        this pin moves with that team's decision.
        """
        policy = CalibrationPolicy()

        assert policy.focus_cadence == 7200.0, (
            f"Focus cadence default changed from 7200.0 s to "
            f"{policy.focus_cadence}s, update this test and the default together"
        )


class TestOverallOverheadVsLiterature:
    """Calibration overhead lands in the model's own sanity band."""

    def test_overhead_fraction_within_sanity_band(self):
        """Calibration bookings land in the model's own 15-25% sanity band.

        The band is this model's sanity range, not a published figure. A
        planned timeline's ``efficiency`` is a different quantity: it also
        pays slew and idle time.

        With a minutes-scale block retune before every science subscan
        (cadence 0), the subscan length sets the retune fraction. The
        estimate uses the model's own ``max_scan_duration`` as the subscan
        length and books pointing, focus and skydip at their cadences:
        300 + 180 + 150 + 100 = 730 s of every hour. Every term is a
        commissioning placeholder, so the pinned total moves with them.
        Ten-minute subscans leave the band (second check).
        """
        model = OverheadModel()
        policy = CalibrationPolicy()
        one_hour = 3600.0

        # One retune per scan boundary (cadence 0), subscans at the
        # model's forced-split length (3600 s by default).
        retune_overhead = (one_hour / model.max_scan_duration) * model.retune_duration
        cadenced_overhead = one_hour * (
            model.pointing_cal_duration / policy.pointing_cadence
            + model.focus_duration / policy.focus_cadence
            + model.skydip_duration / policy.skydip_cadence
        )
        cal_fraction = (retune_overhead + cadenced_overhead) / one_hour

        assert abs(cal_fraction - 730.0 / one_hour) < 1e-12
        assert 0.15 <= cal_fraction <= 0.25, (
            f"Calibration fraction ({cal_fraction:.1%}) outside the sanity band [15%, 25%]"
        )

        # The same block retune at ten-minute subscans costs six retunes
        # an hour and leaves the sanity band; the defaults only
        # reconcile with it at hour-scale subscans.
        n_short_scans = one_hour / 600.0  # 6 scans
        short_scan_fraction = (n_short_scans * model.retune_duration + cadenced_overhead) / one_hour
        assert short_scan_fraction > 0.25


class TestRetuneCadenceComparison:
    """The default retune cadence is the scan-coupled 0."""

    def test_retune_cadence_default_is_scan_coupled(self):
        """The default ``retune_cadence`` is 0, the scan-coupled setting.

        Cadence 0 books a retune immediately before every science subscan;
        the scheduler tests pin that behaviour. The default is an
        instrument-team placeholder pending the retune cadence and trigger
        decisions, and this pin moves with them.
        """
        policy = CalibrationPolicy()
        assert policy.retune_cadence == 0.0
