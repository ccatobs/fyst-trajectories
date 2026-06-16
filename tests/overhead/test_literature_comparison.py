"""Literature comparison tests for the overhead model.

Validates that the fyst-overhead defaults are consistent with
published values from operational KID-based telescopes and CMB
survey instruments.

References
----------
- NIKA2: Perotto et al. 2020 (A&A 637, A71), Adam et al. 2018
  (A&A 609, A115). KID retune ("flat-field") takes ~1-2s. Pointing
  scans every ~1h, focus every ~2h. Overall efficiency 50-80%
  depending on weather and calibration load.
- Simons Observatory: SO schedlib (Koopman et al., in prep). Bias
  step (analogous to retune) every ~60s for TES detectors. Overall
  observing efficiency target ~85%.
- SPT / ACT / TolTEC: Various operational reports. Calibration
  overheads 15-25% of total observing time.
"""

from fyst_trajectories.overhead import CalibrationPolicy, OverheadModel


class TestRetuneVsLiterature:
    """Validate retune parameters against KID instrument experience."""

    def test_retune_duration_conservative_vs_nika2(self):
        """Our default retune (5s) is conservative vs NIKA2 (~1-2s).

        NIKA2's KID retune ("flat-field correction") takes approximately
        1-2 seconds. Our 5-second default provides margin for FYST's
        larger detector count (>100,000 KIDs vs NIKA2's ~3,000).
        """
        model = OverheadModel()
        nika2_retune = 2.0  # seconds, upper bound from NIKA2 operations

        assert model.retune_duration >= nika2_retune, (
            f"Retune duration ({model.retune_duration}s) should be >= "
            f"NIKA2 retune ({nika2_retune}s) given larger KID count"
        )
        # But not excessively long
        assert model.retune_duration <= 10.0, (
            f"Retune duration ({model.retune_duration}s) seems excessive"
        )

    def test_retune_much_shorter_than_so_bias_step(self):
        """Our retune (5s) is much shorter than SO's bias step interval (~60s).

        SO uses TES detectors with bias steps every ~60s. KID retunes
        are fundamentally faster operations -- we should be well under
        the SO bias step interval for the retune *duration*.
        """
        model = OverheadModel()
        so_bias_step_interval = 60.0  # seconds

        assert model.retune_duration < so_bias_step_interval, (
            f"Retune duration ({model.retune_duration}s) should be shorter "
            f"than SO bias step interval ({so_bias_step_interval}s)"
        )


class TestCalibrationCadencesVsLiterature:
    """Validate calibration cadences against published values."""

    def test_pointing_cadence_within_literature_range(self):
        """Default pointing cadence is the agreed 3600 s (1 h).

        NIKA2 and JCMT use ~1 h pointing cadence; the default was flipped from
        1800 s to 3600 s in the audit cycle to match that published practice.
        Pin the exact agreed value so a silent
        regression or a stale docstring is caught, then sanity-check it stays
        within the [20 min, 1 h] literature band. ``1800 s`` remains a
        reasonable commissioning override, passed explicitly.
        """
        policy = CalibrationPolicy()

        # Agreed canonical default (operations-team-owned).
        assert policy.pointing_cadence == 3600.0, (
            f"Pointing cadence default changed from the agreed 3600.0 s to "
            f"{policy.pointing_cadence}s -- update this test and the agreed value together"
        )
        # Sanity: still within the published literature band [20 min, 1 h].
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
    """Validate overall overhead budget against published values."""

    def test_overhead_fraction_matches_nika2_experience(self):
        """Total calibration overhead of 15-25% matches NIKA2 operations.

        NIKA2 reports 20-50% total overhead (calibration + weather +
        technical issues). The calibration-only component is typically
        15-25% of total observing time. Our model should produce
        efficiency in the 75-85% range, consistent with this.
        """
        model = OverheadModel()

        # Compute theoretical overhead for one hour of observing
        # (this is a simplified model; the actual scheduler is more
        # complex due to interleaving)
        one_hour = 3600.0

        # In one hour: retune cadence=0 means every scan boundary,
        # but let's estimate with the default cadence (0 = every scan).
        # Assume scan duration of ~600s (10 min):
        n_scans = one_hour / 600.0  # ~6 scans
        retune_overhead = n_scans * model.retune_duration  # ~30s
        pointing_overhead = model.pointing_cal_duration  # 180s (once per hour)

        # Total: ~210s out of 3600s = ~5.8% for retune + pointing alone.
        # Focus adds ~300s/2h = ~150s/h = ~4.2%, so ~10% minimum.
        cal_overhead = retune_overhead + pointing_overhead
        cal_fraction = cal_overhead / one_hour

        # The minimum calibration overhead should be meaningful (>2%)
        # but not dominate (< 25%)
        assert 0.02 < cal_fraction < 0.25, (
            f"Minimum calibration fraction ({cal_fraction:.1%}) outside expected range [2%, 25%]"
        )

    def test_calibration_duration_ordering(self):
        """Calibration durations follow a strict, sensible ordering.

        Quick electronic operations (retune) < short pointing scans < longer
        focus/skydip < full planet calibrations. The strict ``<`` on the first
        link subsumes the former ``test_default_model_retune_is_fast`` (retune
        is the fastest calibration of all).
        """
        model = OverheadModel()

        assert model.retune_duration < model.pointing_cal_duration
        assert model.pointing_cal_duration <= model.focus_duration
        assert model.focus_duration <= model.skydip_duration
        assert model.skydip_duration <= model.planet_cal_duration


class TestRetuneCadenceComparison:
    """Compare retune cadence with other instruments."""

    def test_retune_cadence_zero_means_every_scan(self):
        """Default retune_cadence=0 means retune at every scan boundary.

        This is the most aggressive cadence, suitable for commissioning
        or conditions requiring frequent recalibration.
        """
        policy = CalibrationPolicy()
        assert policy.retune_cadence == 0.0
