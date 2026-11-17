"""Every argument of a planner the offline scheduler calls is accounted for.

For each scan type the scheduler books, every parameter of the planner's
signature is exactly one of: set from the observing patch (one of its
fields, or a ``scan_params`` key), set by the scheduler itself, or not
expressible, with the reason. A new planner argument fails this test until
it is classified, so an option the planner gains is never silently out of
reach, or silently ignored, in the simulated night.
"""

import dataclasses
import inspect

import pytest

from fyst_trajectories.overhead import CalibrationPolicy, ObservingPatch
from fyst_trajectories.overhead.schemas import _SCAN_TYPE_TO_SCAN_PARAM_KEYS
from fyst_trajectories.planning import (
    plan_constant_el_scan,
    plan_daisy_scan,
    plan_pong_scan,
    plan_source_ces_passes,
)

#: Where each argument comes from: ``("patch", field)`` for an
#: ``ObservingPatch`` field, ``("scan_params", key, ...)`` for its
#: ``scan_params`` keys, ``("policy", field)`` for a ``CalibrationPolicy``
#: field, ``("scheduler", why)`` for a value the scheduler sets, and
#: ``("not expressible", why)``.
_NO_REFRACTION = (
    "not expressible",
    "trajectories are planned in vacuum; refraction is applied once, downstream",
)
_SUN_MODEL = (
    "scheduler",
    "the scheduler judges every scan with its own Sun model before booking it",
)
_START = ("scheduler", "the instant the subscan starts")
_SITE = ("scheduler", "the site the night is scheduled for")

_PARTITION = {
    "constant_el": (
        plan_constant_el_scan,
        {
            "field": ("patch", "ra_center", "dec_center", "width", "height"),
            "elevation": ("patch", "elevation"),
            "velocity": ("patch", "velocity"),
            "detector_offset": ("patch", "detector_offset"),
            "az_window": ("scan_params", "az_min", "az_max"),
            "rising": ("scan_params", "rising"),
            "lsa_window": ("scan_params", "lsa_window"),
            "az_accel": ("scan_params", "az_accel"),
            "timestep": ("scan_params", "timestep"),
            "az_padding": ("scan_params", "az_padding"),
            "site": _SITE,
            "start_time": _START,
            "duration": (
                "scheduler",
                "a window subscan's whole legs, sized to the time the visit has left",
            ),
            "angle": (
                "not expressible",
                "the scheduler's crossing gate, elevation and Sun checks and swept range follow "
                "the field as an unrotated box, so a rotated box would be scheduled by "
                "geometry it does not have",
            ),
            "max_search_hours": (
                "not expressible",
                "the crossing gate and the rebuild must search for the pass the same way, so "
                "both use the planner's own search",
            ),
            "step_seconds": (
                "not expressible",
                "the crossing gate and the rebuild must search for the pass the same way, so "
                "both use the planner's own search",
            ),
            "atmosphere": _NO_REFRACTION,
            "sun_safe": _SUN_MODEL,
        },
    ),
    "pong": (
        plan_pong_scan,
        {
            "field": ("patch", "ra_center", "dec_center", "width", "height"),
            "velocity": ("patch", "velocity"),
            "detector_offset": ("patch", "detector_offset"),
            "spacing": ("scan_params", "spacing"),
            "num_terms": ("scan_params", "num_terms"),
            "timestep": ("scan_params", "timestep"),
            "angle": ("scan_params", "angle"),
            "n_cycles": ("scan_params", "n_cycles"),
            "site": _SITE,
            "start_time": _START,
            "atmosphere": _NO_REFRACTION,
            "sun_safe": _SUN_MODEL,
        },
    ),
    "daisy": (
        plan_daisy_scan,
        {
            "ra": ("patch", "ra_center"),
            "dec": ("patch", "dec_center"),
            "velocity": ("patch", "velocity"),
            "detector_offset": ("patch", "detector_offset"),
            "radius": ("scan_params", "radius"),
            "turn_radius": ("scan_params", "turn_radius"),
            "avoidance_radius": ("scan_params", "avoidance_radius"),
            "start_acceleration": ("scan_params", "start_acceleration"),
            "timestep": ("scan_params", "timestep"),
            "site": _SITE,
            "start_time": _START,
            "duration": ("scheduler", "the time the visit has, up to the longest subscan"),
            "y_offset": (
                "not expressible",
                "the scheduler's elevation bounds and Sun clip follow a daisy centred on the "
                "field, so a shifted pattern would be judged where it does not run",
            ),
            "atmosphere": _NO_REFRACTION,
            "sun_safe": _SUN_MODEL,
        },
    ),
    "source_ces": (
        plan_source_ces_passes,
        {
            "body": ("scheduler", "the first listed planet that is up and clear of the Sun"),
            "footprint": ("policy", "planet_cal_footprint"),
            "n_passes": ("policy", "planet_cal_passes"),
            "el_step": ("policy", "planet_cal_el_step"),
            "site": _SITE,
            "start_time": ("scheduler", "the clock when the calibration falls due"),
            "timestep": ("scheduler", "the planet-pass sampling interval"),
            "sun_safe": ("scheduler", "the night's own Sun model"),
            "ra": ("not expressible", "a planet calibration targets a solar-system body"),
            "dec": ("not expressible", "a planet calibration targets a solar-system body"),
            "pm_ra": ("not expressible", "a planet calibration targets a solar-system body"),
            "pm_dec": ("not expressible", "a planet calibration targets a solar-system body"),
            "ref_epoch": ("not expressible", "a planet calibration targets a solar-system body"),
            "el_bore": ("not expressible", "derived from the anchor, the clock"),
            "mode": ("not expressible", "derived from the body's motion at the anchor"),
            "boresight_rot": ("not expressible", "Prime-Cam has no boresight rotator"),
            "window": ("not expressible", "the passes are anchored at the clock"),
            "night": ("not expressible", "the passes are anchored at the clock"),
            "step": ("not expressible", "the policy offers the passes' elevation step only"),
            "eta_offsets": ("not expressible", "the policy offers the passes' elevation step only"),
            "sampling_step_seconds": (
                "not expressible",
                "the planner's own search grid, which a rebuild repeats from the record",
            ),
            "az_accel": ("not expressible", "the planner's default; a pass records it"),
            "az_padding": ("not expressible", "the planner's default; a pass records it"),
            "az_branch": ("not expressible", "the slew phase chooses the wrap"),
            "allow_partial": ("not expressible", "a calibration pass crosses the whole cover"),
            "v_az": ("not expressible", "solved by the planner from the body's drift"),
            "az_speed": ("not expressible", "solved by the planner from the body's drift"),
            "az_throw": ("not expressible", "solved by the planner from the footprint"),
            "dwell": ("not expressible", "a calibration pass crosses the whole cover"),
            "atmosphere": _NO_REFRACTION,
        },
    ),
}

_KINDS = {"patch", "scan_params", "policy", "scheduler", "not expressible"}


@pytest.mark.parametrize("scan_type", sorted(_PARTITION))
def test_every_planner_argument_is_classified(scan_type):
    planner, partition = _PARTITION[scan_type]
    arguments = set(inspect.signature(planner).parameters)
    assert set(partition) == arguments, (
        f"{planner.__name__}: unclassified {sorted(arguments - set(partition))}, "
        f"not in the signature {sorted(set(partition) - arguments)}"
    )


@pytest.mark.parametrize("scan_type", sorted(_PARTITION))
def test_every_class_names_where_the_value_comes_from(scan_type):
    _, partition = _PARTITION[scan_type]
    patch_fields = {f.name for f in dataclasses.fields(ObservingPatch)}
    policy_fields = {f.name for f in dataclasses.fields(CalibrationPolicy)}
    for argument, (kind, *where) in partition.items():
        assert kind in _KINDS, argument
        assert where and all(where), argument
        if kind == "patch":
            assert set(where) <= patch_fields, argument
        elif kind == "policy":
            assert set(where) <= policy_fields, argument


@pytest.mark.parametrize("scan_type", ["constant_el", "pong", "daisy"])
def test_the_scan_params_vocabulary_is_exactly_the_expressible_keys(scan_type):
    """Every ``scan_params`` key reaches a planner argument, and no other key is declared."""
    _, partition = _PARTITION[scan_type]
    expressible = {
        key for kind, *keys in partition.values() if kind == "scan_params" for key in keys
    }
    assert expressible == set(_SCAN_TYPE_TO_SCAN_PARAM_KEYS[scan_type])
