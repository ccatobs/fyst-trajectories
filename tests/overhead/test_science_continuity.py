"""Two science blocks booked back to back join without a step.

Where one science block follows another with no block between them, the
telescope is booked no move between them, so the commanded path must run on
from one block into the next: on every shared night of ``_nights.py`` the
steps in azimuth, elevation and their rates across the boundary are no
larger than the largest step inside either block (so neither a jump nor a
reversal at full speed passes).

Back-to-back constant-elevation blocks are slices of one planned scan (the
crossing pass of a field, or one window scan of a patch with an azimuth
window), so the next block's rebuilt first sample is one sample interval
after the previous block's last. A pong or daisy block is a scan of its
own and holds both of its end samples: the next one starts where the
previous one ended, at most one interval after its last sample (two pong
subscans of whole periods, booked with every overhead zero, share that
instant).
"""

import numpy as np
import pytest

from fyst_trajectories.overhead import BlockType

from ._nights import NIGHTS, night, rebuilt

#: Time tolerance in seconds on the interval across a boundary.
_TIME_TOL = 1e-6

#: Tolerance on a step across a boundary, in degrees or deg/s.
_ANGLE_TOL = 1e-9


def _back_to_back(name):
    """Return ``(first block, its rebuild, the next block's rebuild)`` for each adjacent pair.

    A pair is two science blocks with no block between them.
    """
    scans = {id(block): scan for block, scan in rebuilt(name)}
    blocks = list(night(name).blocks)
    return [
        (first, scans[id(first)], scans[id(second)])
        for first, second in zip(blocks, blocks[1:], strict=False)
        if first.block_type == BlockType.SCIENCE
        and second.block_type == BlockType.SCIENCE
        and id(first) in scans
        and id(second) in scans
    ]


def _largest_step(values):
    return float(np.max(np.abs(np.diff(values)))) if values.size > 1 else 0.0


@pytest.mark.parametrize("name", sorted(NIGHTS))
def test_back_to_back_science_blocks_join_without_a_step(name):
    """The path runs on across the boundary, and no sample of one scan is in two blocks."""
    for block, first, second in _back_to_back(name):
        a, b = first.trajectory, second.trajectory
        where = (name, b.start_time.isot)
        interval = float(a.times[-1] - a.times[-2])
        gap = (b.start_time - a.start_time).sec + float(b.times[0]) - float(a.times[-1])
        if block.scan_type in ("pong", "daisy"):
            assert -_TIME_TOL <= gap <= interval + _TIME_TOL, where
        else:
            assert gap == pytest.approx(interval, abs=_TIME_TOL), where
        for axis in ("az", "el", "az_vel", "el_vel"):
            before, after = getattr(a, axis), getattr(b, axis)
            step = abs(float(after[0]) - float(before[-1]))
            largest = max(_largest_step(before), _largest_step(after))
            assert step <= largest + _ANGLE_TOL, (*where, axis, step, largest)


def test_a_night_with_back_to_back_blocks_is_covered():
    """The guard has pairs of both kinds to check.

    The window night is cut with no retune between its blocks, and the
    night with every overhead zero books two pong subscans back to back.
    """
    assert _back_to_back("window_cut")
    pong_pairs = [block for block, _, _ in _back_to_back("zero_settle")]
    assert any(block.scan_type == "pong" for block in pong_pairs)
