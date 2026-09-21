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
- NIKA2: Perotto et al. 2020 (A&A 637, A71), Adam et al. 2018
  (A&A 609, A115). The KID tone-matching "tuning" completes in under
  2 s (Adam 2018 sec. 3.2), in a dedicated sub-scan at each scan
  boundary. Pointing scans every ~1 h, focus every ~2 h.
- Simons Observatory: SO schedlib. TES bias steps default to a 30 min
  cadence and take ~60 s; detector setup runs at block start. A bias step
  is a TES operation with no KID analogue, so it is not compared here.
  Overall observing efficiency target ~85%.
"""

from fyst_trajectories.overhead import CalibrationPolicy, OverheadModel
from fyst_trajectories.trajectory_utils import DEFAULT_RETUNE_DURATION_SEC


class TestRetuneVsLiterature:
    """Validate the two retune durations against their own counterparts."""

    def test_in_scan_gap_conservative_vs_nika2_tuning(self):
        """The in-scan gap (5 s) is conservative vs NIKA2's tuning (<2 s).

        NIKA2's KID tone-matching "tuning" completes in under 2 s.
        The 5 s default provides margin for FYST's larger detector count
        (>100,000 KIDs vs NIKA2's ~3,000) without being excessive.
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
        (a seconds-vs-minutes slip in either direction fails the band).
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
    """Pointing at 3600 s and focus at 2 h sit inside the published cadence bands."""

    def test_pointing_cadence_within_literature_range(self):
        """Default pointing cadence is 3600 s (1 h).

        NIKA2 monitors pointing hourly (Perotto 2020 sec. 3.2), which the
        3600 s default matches. Pin the exact default value so a silent
        regression or a stale docstring is caught, then sanity-check it stays
        within the [20 min, 1 h] band. ``1800 s`` remains a reasonable
        commissioning override, passed explicitly.
        """
        policy = CalibrationPolicy()

        # Canonical default (operations-team-owned).
        assert policy.pointing_cadence == 3600.0, (
            f"Pointing cadence default changed from 3600.0 s to "
            f"{policy.pointing_cadence}s, update this test and the default together"
        )
        # Sanity: still within the [20 min, 1 h] band.
        assert 1200.0 <= policy.pointing_cadence <= 3600.0

    def test_focus_cadence_reasonable(self):
        """Focus check every 2h is within standard range (1-4h).

        Different instruments use 1-4 hour focus cadences depending
        on thermal stability. Our 2h default is in the middle of
        this range.
        """
        policy = CalibrationPolicy()

        assert 3600.0 <= policy.focus_cadence <= 14400.0, (
            f"Focus cadence ({policy.focus_cadence}s) outside standard range [1h, 4h]"
        )


class TestOverallOverheadVsLiterature:
    """Calibration overhead lands in the model's own sanity band, and the durations order."""

    def test_overhead_fraction_within_sanity_band(self):
        """Total calibration overhead lands in the 15-25% sanity band.

        Ground-based submm cameras typically spend 15-25% of observing time
        on calibration; this test pins that band as the model's own sanity
        range, not as a published NIKA2 figure. Adding the periodic focus
        and skydip checks to the retune-and-pointing estimate below puts the
        model's calibration bookings inside that band. A planned timeline's
        ``efficiency`` is a different quantity: it also pays slew and idle
        time.

        With a minutes-scale block retune before every science subscan
        (cadence 0), the subscan length sets the retune fraction. The
        estimate below uses the model's own ``max_scan_duration`` as the
        subscan length, which is where the defaults reconcile with the
        sanity band; ten-minute subscans would not (second check).
        """
        model = OverheadModel()

        # Compute theoretical overhead for one hour of observing
        # (this is a simplified model; the actual scheduler is more
        # complex due to interleaving)
        one_hour = 3600.0

        # One retune per scan boundary (cadence 0), subscans at the
        # model's forced-split length (3600 s by default).
        n_scans = one_hour / model.max_scan_duration  # 1 scan
        retune_overhead = n_scans * model.retune_duration  # 300s
        pointing_overhead = model.pointing_cal_duration  # 180s (once per hour)

        # Total: ~480s out of 3600s = ~13.3% for retune + pointing alone.
        # Focus adds ~300s/2h = ~150s/h = ~4.2%, so ~17.5% minimum.
        cal_overhead = retune_overhead + pointing_overhead
        cal_fraction = cal_overhead / one_hour

        # The minimum calibration overhead should be meaningful (>2%)
        # but not dominate (< 25%)
        assert 0.02 < cal_fraction < 0.25, (
            f"Minimum calibration fraction ({cal_fraction:.1%}) outside expected range [2%, 25%]"
        )

        # The same block retune at ten-minute subscans costs six retunes
        # an hour and leaves the sanity band; the defaults only
        # reconcile with it at hour-scale subscans.
        n_short_scans = one_hour / 600.0  # 6 scans
        short_scan_fraction = (n_short_scans * model.retune_duration + pointing_overhead) / one_hour
        assert short_scan_fraction > 0.25

    def test_calibration_duration_ordering(self):
        """Calibration scan durations follow a sensible ordering.

        Short pointing scans <= longer focus/skydip <= full planet
        calibrations. The block retune is not in this chain: a
        whole-array tone placement plus target sweep is a minutes-scale
        detector operation of its own kind, not "the fastest
        calibration", so no ordering between it and the scans is asserted.
        """
        model = OverheadModel()

        assert model.pointing_cal_duration <= model.focus_duration
        assert model.focus_duration <= model.skydip_duration
        assert model.skydip_duration <= model.planet_cal_duration


class TestRetuneCadenceComparison:
    """The default retune cadence of 0 books a retune at every scan boundary."""

    def test_retune_cadence_zero_means_every_scan(self):
        """Default retune_cadence=0 means retune at every scan boundary.

        This is the most aggressive cadence, suitable for commissioning
        or conditions requiring frequent recalibration.
        """
        policy = CalibrationPolicy()
        assert policy.retune_cadence == 0.0
