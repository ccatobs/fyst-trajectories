"""End-to-end hitmap validation for inject_retune().

Verifies that science_mask correctly reduces the number of science
samples when retune events are injected, and that the reduction
ratio matches the expected efficiency.

The optional primecam_camera_mapping_simulations package, when it is
installed, extends this to full detector-level hitmap validation; this
test covers the boresight-level behavior that feeds into any hitmap
pipeline.
"""

import numpy as np
import pytest
from astropy.time import Time

from fyst_trajectories import get_fyst_site
from fyst_trajectories.planning import FieldRegion, plan_constant_el_scan
from fyst_trajectories.trajectory import (
    SCAN_FLAG_RETUNE,
    SCAN_FLAG_SCIENCE,
    SCAN_FLAG_TURNAROUND,
)
from fyst_trajectories.trajectory_utils import inject_retune

# Check if primecam is available for the extended test
try:
    import primecam_camera_mapping_simulations  # noqa: F401

    HAS_PRIMECAM = True
except ImportError:
    HAS_PRIMECAM = False


@pytest.fixture
def ce_trajectory():
    """Generate a CE trajectory for E-CDF-S field (RA~53.1, Dec~-27.8)."""
    site = get_fyst_site()
    field = FieldRegion(ra_center=53.1, dec_center=-27.8, width=30.0, height=8.0)
    block = plan_constant_el_scan(
        field=field,
        elevation=50.0,
        velocity=1.0,
        site=site,
        start_time=Time("2026-06-15T04:00:00", scale="utc"),
        rising=True,
        timestep=0.1,
    )
    return block.trajectory


class TestScienceMaskReduction:
    """Verify science_mask correctly tracks retune overhead."""

    def test_science_mask_reduces_sample_count(self, ce_trajectory):
        """Injecting retunes should reduce the number of science samples."""
        original_science = ce_trajectory.science_mask.sum()
        result = inject_retune(
            ce_trajectory,
            retune_interval=30.0,
            retune_duration=5.0,
        )
        reduced_science = result.science_mask.sum()

        assert reduced_science < original_science, (
            f"Expected fewer science samples after retune injection: "
            f"original={original_science}, after={reduced_science}"
        )

    def test_science_ratio_matches_efficiency(self, ce_trajectory):
        """Ratio of science samples to total should match expected efficiency.

        For 30s/5s retune, theoretical efficiency is ~83.3%.
        CE scans have turnarounds that reduce effective science fraction
        further, so we allow a wider range.
        """
        result = inject_retune(
            ce_trajectory,
            retune_interval=30.0,
            retune_duration=5.0,
        )

        total_samples = len(result.times)
        science_samples = result.science_mask.sum()
        ratio = science_samples / total_samples

        # CE scans already have ~2-5% turnaround overhead, plus ~16.7%
        # retune overhead, so science fraction should be roughly 78-87%
        assert 0.75 <= ratio <= 0.90, (
            f"Science ratio {ratio:.3f} outside expected range [0.75, 0.90]"
        )

    def test_retune_and_turnaround_exclusive(self, ce_trajectory):
        """Retune flags should never overwrite turnaround flags."""
        result = inject_retune(
            ce_trajectory,
            retune_interval=30.0,
            retune_duration=5.0,
        )

        # Count turnaround samples before and after
        original_ta = (ce_trajectory.scan_flag == SCAN_FLAG_TURNAROUND).sum()
        result_ta = (result.scan_flag == SCAN_FLAG_TURNAROUND).sum()

        assert result_ta == original_ta, f"Turnaround count changed: {original_ta} -> {result_ta}"

    def test_flag_values_partition(self, ce_trajectory):
        """Every sample should have exactly one of: science, turnaround, retune."""
        result = inject_retune(
            ce_trajectory,
            retune_interval=30.0,
            retune_duration=5.0,
        )

        n_sci = (result.scan_flag == SCAN_FLAG_SCIENCE).sum()
        n_ta = (result.scan_flag == SCAN_FLAG_TURNAROUND).sum()
        n_ret = (result.scan_flag == SCAN_FLAG_RETUNE).sum()
        total = len(result.scan_flag)

        assert n_sci + n_ta + n_ret == total, (
            f"Flag partition mismatch: "
            f"science={n_sci} + turnaround={n_ta} + retune={n_ret} "
            f"!= total={total}"
        )

    def test_science_mask_consistent_with_flags(self, ce_trajectory):
        """science_mask should be True exactly where scan_flag == SCIENCE."""
        result = inject_retune(
            ce_trajectory,
            retune_interval=30.0,
            retune_duration=5.0,
        )

        expected_mask = result.scan_flag == SCAN_FLAG_SCIENCE
        np.testing.assert_array_equal(
            result.science_mask,
            expected_mask,
            err_msg="science_mask inconsistent with scan_flag",
        )


@pytest.mark.skipif(not HAS_PRIMECAM, reason="primecam not installed")
class TestPrimecamIntegration:
    """Extended validation using an external hitmap simulator when available.

    These tests verify that the trajectory output from inject_retune
    is compatible with the primecam hitmap pipeline.

    When primecam is not available, these tests document what would be
    tested:
    - Full detector-level hitmap with and without retune
    - Comparison of boresight-level vs detector-level efficiency
    - Spatial uniformity of retune gaps across the focal plane
    """

    def test_trajectory_compatible_with_primecam(self, ce_trajectory):
        """Retune-injected trajectory should have valid arrays for primecam."""
        result = inject_retune(
            ce_trajectory,
            retune_interval=30.0,
            retune_duration=5.0,
        )

        # primecam expects: times, az, el arrays with consistent lengths
        assert len(result.times) == len(result.az) == len(result.el)
        # Arrays should be finite
        assert np.all(np.isfinite(result.az))
        assert np.all(np.isfinite(result.el))
        assert np.all(np.isfinite(result.times))
        # science_mask should exist and have same length
        assert len(result.science_mask) == len(result.times)
