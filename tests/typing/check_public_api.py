"""Use the public surface as its consumers do, for a type checker.

pyright reads this file (``pyright -p tests/typing``, the CI typing job);
nothing executes it and pytest does not collect it. Each ``assert_type``
pins a planner's return annotation, which would otherwise decay to an
unchecked type without an error. The rest reads the planners' computed
parameters and the ``/path`` payload the way a dispatcher does.
"""

from collections.abc import Sequence

from typing_extensions import assert_type

from fyst_trajectories.dispatch import choose_encoder_solution
from fyst_trajectories.patterns import rewrap_trajectory_azimuth
from fyst_trajectories.planning import (
    ComputedParams,
    ConstantElComputedParams,
    DaisyAltAzComputedParams,
    DaisyComputedParams,
    FieldRegion,
    PongAltAzComputedParams,
    PongComputedParams,
    ScanBlock,
    SourceCESComputedParams,
    plan_constant_el_scan,
    plan_daisy_altaz_scan,
    plan_daisy_scan,
    plan_pong_altaz_scan,
    plan_pong_rotation_scans,
    plan_pong_scan,
    plan_source_ces,
    plan_source_ces_passes,
)
from fyst_trajectories.site import Site
from fyst_trajectories.trajectory_utils import PathPayload, to_path_payload

SCAN_DISPATCH_BUFFER_SEC = 10.0


def plan_every_scan(field: FieldRegion, site: Site, start: str) -> list[ScanBlock[ComputedParams]]:
    """Each planner returns a block typed by its own computed parameters."""
    pong = plan_pong_scan(field, velocity=0.5, spacing=0.1, site=site, start_time=start)
    assert_type(pong, ScanBlock[PongComputedParams])
    period: float = pong.computed_params["period"]

    rotations = plan_pong_rotation_scans(
        field, n_rotations=2, start_time=start, velocity=0.5, spacing=0.1, site=site
    )
    assert_type(rotations, list[ScanBlock[PongComputedParams]])
    tiling: Sequence[ScanBlock[ComputedParams]] = rotations

    pong_altaz = plan_pong_altaz_scan(
        180.0,
        50.0,
        width=2.0,
        height=2.0,
        spacing=0.1,
        velocity=0.5,
        site=site,
        start_time=start,
    )
    assert_type(pong_altaz, ScanBlock[PongAltAzComputedParams])
    az_center: float = pong_altaz.computed_params["az_center"]

    daisy = plan_daisy_scan(
        10.0,
        -30.0,
        radius=0.5,
        velocity=0.3,
        turn_radius=0.2,
        avoidance_radius=0.0,
        start_acceleration=0.5,
        site=site,
        start_time=start,
        duration=300.0,
    )
    assert_type(daisy, ScanBlock[DaisyComputedParams])
    daisy_duration: float = daisy.computed_params["duration"]

    daisy_altaz = plan_daisy_altaz_scan(
        180.0,
        50.0,
        radius=0.5,
        velocity=0.3,
        turn_radius=0.2,
        avoidance_radius=0.0,
        start_acceleration=0.5,
        site=site,
        start_time=start,
        duration=300.0,
    )
    assert_type(daisy_altaz, ScanBlock[DaisyAltAzComputedParams])
    el_center: float = daisy_altaz.computed_params["el_center"]

    ce = plan_constant_el_scan(field, elevation=50.0, velocity=1.0, site=site, start_time=start)
    assert_type(ce, ScanBlock[ConstantElComputedParams])
    az_throw: float = ce.computed_params["az_throw"]
    start_iso: str = ce.computed_params["start_time_iso"]

    source = plan_source_ces(body="mars", footprint="c", start_time=start, site=site)
    assert_type(source, ScanBlock[SourceCESComputedParams])
    v_az: float = source.computed_params["v_az"]

    passes = plan_source_ces_passes(
        body="mars", footprint="c", n_passes=2, start_time=start, site=site
    )
    assert_type(passes, list[ScanBlock[SourceCESComputedParams]])
    el_bore: float = passes[0].computed_params["el_bore"]

    assert period and az_center and daisy_duration and el_center and az_throw and start_iso
    assert v_az and el_bore
    return [*tiling, pong_altaz, daisy, daisy_altaz, ce, source, *passes]


def refloor_start_time(payload: PathPayload, now_unix: float) -> float:
    """Push a late start out to the dispatch floor and return the drift."""
    original = payload["start_time"]
    payload["start_time"] = max(float(payload["start_time"]), now_unix + SCAN_DISPATCH_BUFFER_SEC)
    return payload["start_time"] - original


def dispatch(block: ScanBlock[ComputedParams], site: Site, now_unix: float) -> float:
    """Choose the wrap, shift the trajectory into it and build the ``/path`` body."""
    trajectory = block.trajectory
    assert trajectory.start_time is not None
    solution = choose_encoder_solution(
        10.0,
        45.0,
        float(trajectory.az[0]),
        float(trajectory.el[0]),
        trajectory.start_time,
        site,
        goal_az_span=(float(trajectory.az.min()), float(trajectory.az.max())),
    )
    az_shift: float = solution.az_shift
    commanded = rewrap_trajectory_azimuth(trajectory, az_shift)
    payload = to_path_payload(commanded)
    refloor_start_time(payload, now_unix)
    coordsys: str = payload["coordsys"]
    assert coordsys == "Horizon"
    end_unix: float = payload["start_time"] + payload["points"][-1][0]
    return end_unix
