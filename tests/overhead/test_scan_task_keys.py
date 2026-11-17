"""The simulator's ``scan_params`` keys against the keys the telescope's scan tasks read.

The offline simulator describes a scan with an ``ObservingPatch`` (its
fields and its ``scan_params``) or, for a planet pass, with the
``scan_params`` a calibration block records. On the telescope the same
scan is a typed PCS scan task whose builder reads a ``scan_params`` dict.
``tests/data/pcs_scan_param_keys_180746d.json`` lists the keys each task's
builder reads at PCS revision ``180746d``; a key outside that list is
dropped there without a message.

This test holds each simulator vocabulary against that list. Every key on
one side and not the other is listed below with the reason, so a key added
on either side fails here until it is accounted for. The keys the change
prepared for PCS's next revision adds are listed beside them.
"""

import json
from pathlib import Path

import pytest

from fyst_trajectories.overhead.schemas import _SCAN_TYPE_TO_SCAN_PARAM_KEYS

_SNAPSHOT = Path(__file__).resolve().parents[1] / "data" / "pcs_scan_param_keys_180746d.json"

#: The scan task each simulator scan type runs as.
_TASKS = {
    "constant_el": "constant_el_scan",
    "pong": "pong_scan",
    "daisy": "daisy_scan",
    "source_ces": "source_scan",
}

_PATCH_FIELD = "an ObservingPatch field, not a scan_params key"

#: Keys the scan task reads that the simulator's ``scan_params`` does not
#: hold, and why.
_TASK_ONLY = {
    "constant_el": {
        "ra_center": _PATCH_FIELD,
        "dec_center": _PATCH_FIELD,
        "width": _PATCH_FIELD,
        "height": _PATCH_FIELD,
        "elevation": _PATCH_FIELD,
        "velocity": _PATCH_FIELD,
        "angle": "a rotated box: the simulator's checks follow an unrotated field",
        "max_search_hours": "the simulator uses the planner's own crossing search",
        "step_seconds": "the simulator uses the planner's own crossing search",
    },
    "pong": {
        "ra_center": _PATCH_FIELD,
        "dec_center": _PATCH_FIELD,
        "width": _PATCH_FIELD,
        "height": _PATCH_FIELD,
        "velocity": _PATCH_FIELD,
    },
    "daisy": {
        "ra": "the patch's ra_center",
        "dec": "the patch's dec_center",
        "velocity": _PATCH_FIELD,
        "duration": "set by the simulator: the time the visit has",
        "y_offset": "a shifted daisy: the simulator's checks follow a daisy on the field centre",
    },
    "source_ces": {
        "ra": "a planet pass targets a solar-system body",
        "dec": "a planet pass targets a solar-system body",
        "pm_ra": "a planet pass targets a solar-system body",
        "pm_dec": "a planet pass targets a solar-system body",
        "ref_epoch": "a planet pass targets a solar-system body",
        "allow_partial": "a planet pass crosses the whole footprint",
        "az_branch": "the simulator's slew chooses the wrap",
        "sampling_step_seconds": "the planner's own search grid",
    },
}

#: Keys the simulator holds that the scan task does not read at ``180746d``,
#: why, and whether the change prepared for PCS's next revision reads them.
_SIMULATOR_ONLY = {
    "constant_el": {
        "az_min": ("the azimuth window the scan sweeps", True),
        "az_max": ("the azimuth window the scan sweeps", True),
    },
    "pong": {},
    "daisy": {},
    "source_ces": {
        "az_speed": ("a sweep override the planner records and replays", True),
        "az_throw": ("a sweep override the planner records and replays", True),
        "dwell": ("a sweep override the planner records and replays", True),
        "footprint_margin": ("the margin the footprint was widened by", True),
        "pass_index": ("which pass of the sequence, a record only", True),
        "n_passes": ("how many passes were asked for, a record only", True),
        "eta_offset_deg": (
            "the footprint's shift in elevation for this pass; zero for the centre bay",
            False,
        ),
    },
}

#: Keys the change prepared for PCS's next revision adds to a task, none read
#: at ``180746d``: the simulator's keys marked True above, and others, among
#: them the bay a scan is planned for, by name.
_ADDED_FOR_THE_NEXT_REVISION = {
    "constant_el_scan": {"az_min", "az_max", "duration", "module", "footprint"},
    "pong_scan": {"module", "footprint"},
    "daisy_scan": {"module", "footprint"},
    "source_scan": {
        "footprint_margin",
        "az_speed",
        "az_throw",
        "dwell",
        "pass_index",
        "n_passes",
        "module",
    },
}


def _snapshot():
    return json.loads(_SNAPSHOT.read_text(encoding="utf-8"))


def test_the_snapshot_names_its_revision():
    snapshot = _snapshot()
    assert (snapshot["repository"], snapshot["commit"]) == ("ccatobs/pcs", "180746d")
    assert set(snapshot["tasks"]) == set(_TASKS.values())


@pytest.mark.parametrize("scan_type", sorted(_TASKS))
def test_every_key_on_one_side_only_is_accounted_for(scan_type):
    task = _TASKS[scan_type]
    read = set(_snapshot()["tasks"][task]["keys"])
    held = set(_SCAN_TYPE_TO_SCAN_PARAM_KEYS[scan_type])
    assert read - held == set(_TASK_ONLY[scan_type])
    assert held - read == set(_SIMULATOR_ONLY[scan_type])
    reasons = [*_TASK_ONLY[scan_type].values()]
    reasons += [why for why, _ in _SIMULATOR_ONLY[scan_type].values()]
    assert all(reasons)


@pytest.mark.parametrize("scan_type", sorted(_TASKS))
def test_the_next_revision_reads_what_it_is_listed_as_reading(scan_type):
    task = _TASKS[scan_type]
    added = _ADDED_FOR_THE_NEXT_REVISION[task]
    assert not added & set(_snapshot()["tasks"][task]["keys"])
    for key, (_, next_revision) in _SIMULATOR_ONLY[scan_type].items():
        assert (key in added) == next_revision, key
