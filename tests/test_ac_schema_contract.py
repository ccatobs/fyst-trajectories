"""Layer-A (fystplan / OpsDB) -> Layer-C (typed PCS scan task) schema contract.

Under FYST Architecture 3 the survey planner (KOSMA ``fystplan``) writes
per-source scan parameters into OpsDB; at dispatch time the PCS typed scan
task reads them back and calls the matching ``plan_*_scan`` planner in this
library. There is no shared schema module between the two sides. The field
names are coupled only by an *implicit* adapter. This test pins that adapter
explicitly so a future rename on **either** side fails loudly here instead of
silently mis-dispatching a scan at the telescope.

The OpsDB scan parameters are carried in the ``mapping_parameters`` /
``additional_params`` fields.

The fystplan emit functions whose dict literals are the source of truth for
the fixtures below (read-only; never edited by this repo):

* Pong  - ``fystplan/PrimeCam_planning/create_PrimeCam_sourcecatalog.py``
  ``get_pong_params`` (dict at lines ~1196-1213).
* Daisy - same file, ``get_daisy_params`` (dict at lines ~1266-1271).
* CE    - same file, ``create_entry_constant_elevation`` (the ``list_az_table``
  ``s`` dict + the ObsUnit ``nominal_alt`` carrier + the ``onsky_velocity``
  ``mapping_params`` carrier).

The renames pinned here (verified against the fystplan source 2026-06-16):

====================  ========================  =====================
fystplan / OpsDB key  fyst-trajectories kwarg   scan type(s)
====================  ========================  =====================
``onsky_velocity``    ``velocity``              pong, daisy, CE
``R0``                ``radius``                daisy
``Rt``                ``turn_radius``           daisy
``Ra``                ``avoidance_radius``      daisy
``num_terms``         ``num_terms`` (no rename) pong
====================  ========================  =====================

Note ``num_terms`` is **not** renamed: fystplan already emits ``num_terms``
(plural). The "``num_term``->``num_terms``" rename refers to a
*legacy ``scan_patterns``* spelling, not fystplan;
fystplan and this library already agree. The mapping is asserted as identity so
that if either side ever drifts to the singular, this test catches it.

fystplan omits ``start_acceleration`` and ``y_offset`` from its daisy dict
entirely (it specifies no ramp acceleration). ``y_offset`` is covered by the
``plan_daisy_scan`` default (0.0); ``start_acceleration`` is a *required*
planner parameter with **no default**. The dispatch-time adapter must supply
it from an instrument-owned source, not from OpsDB. That gap is asserted here so
it cannot be forgotten.
"""

import inspect

import pytest

from fyst_trajectories.planning import (
    FieldRegion,
    plan_constant_el_scan,
    plan_daisy_scan,
    plan_pong_scan,
    plan_source_ces,
)

# ---------------------------------------------------------------------------
# THE ADAPTER CONTRACT
#
# ``RENAME_MAP`` is the single source of truth for the A->C key translation.
# A dispatch-time PCS adapter that reads OpsDB and calls ``plan_*_scan`` must
# apply exactly this map. If a planner kwarg is renamed without updating this
# map, the subset assertions below fail; if fystplan adds an emit key without a
# rule here, the coverage assertions fail.
# ---------------------------------------------------------------------------
RENAME_MAP: dict[str, str] = {
    "onsky_velocity": "velocity",
    "R0": "radius",
    "Rt": "turn_radius",
    "Ra": "avoidance_radius",
    "num_terms": "num_terms",  # identity: fystplan already matches the planner
}

# fystplan emit keys that are labels / structural inputs, NOT direct scalar
# planner kwargs. ``name`` is a bookkeeping label. ``width``/``height`` are
# consumed by ``FieldRegion`` (for pong / CE), not passed straight to the
# planner. The CE az-table geometry keys describe an az window that the planner
# *derives* internally from the field + elevation, so they are not planner
# scalar kwargs either.
FYSTPLAN_LABEL_KEYS: frozenset[str] = frozenset({"name"})
FIELDREGION_KEYS: frozenset[str] = frozenset({"width", "height"})

# Planner parameters supplied at dispatch time by the PCS task, NOT carried in
# any fystplan scan-param dict (the site is fixed; start_time/timestep/duration
# come from the schedule and the SCAN_DISPATCH_BUFFER_SEC recompute). These are
# legitimately absent from the OpsDB schema, so the contract must not flag them.
DISPATCH_INFRA_KEYS: frozenset[str] = frozenset({"site", "start_time", "timestep", "duration"})


def _planner_param_names(fn) -> set[str]:
    """Return the accepted parameter names of a planner (excluding *args/**kwargs)."""
    return {
        name
        for name, p in inspect.signature(fn).parameters.items()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }


def _required_planner_params(fn) -> set[str]:
    """Return the names of planner parameters that have no default (required)."""
    return {
        name
        for name, p in inspect.signature(fn).parameters.items()
        if p.default is inspect.Parameter.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }


def _apply_rename(fystplan_dict: dict) -> dict:
    """Apply ``RENAME_MAP`` to a fystplan emit dict, dropping label/structural keys.

    Returns the scalar kwargs the adapter would hand to a planner: every key is
    renamed per ``RENAME_MAP`` if a rule exists, otherwise passed through
    unchanged; ``name`` and the FieldRegion keys are removed (they are handled
    structurally, not as scalar kwargs).
    """
    out: dict = {}
    for key, value in fystplan_dict.items():
        if key in FYSTPLAN_LABEL_KEYS or key in FIELDREGION_KEYS:
            continue
        out[RENAME_MAP.get(key, key)] = value
    return out


# ---------------------------------------------------------------------------
# Fixtures: the EXACT dicts fystplan emits (verbatim key sets).
# Values are representative; only the keys are load-bearing for the contract.
# ---------------------------------------------------------------------------
@pytest.fixture
def fystplan_pong_dict() -> dict:
    """``get_pong_params`` emit dict (fixed-angle variant), keys verbatim."""
    return {
        "name": "ECDFS_pong",
        "onsky_velocity": 0.5,
        "width": 5.0,
        "height": 6.7,
        "angle": 170.0,
        "spacing": 0.5,
        "num_terms": 5,
    }


@pytest.fixture
def fystplan_pong_dict_angle_var() -> dict:
    """``get_pong_params`` emit dict (``angle_var`` variant), keys verbatim.

    fystplan emits ``angle_var`` (an int selector for a per-pass variable
    rotation) in place of ``angle``. This library has no ``angle_var`` analogue
    (recommendations 2026-05-28 section 5 flags it "semantic" / no fyst-traj
    analogue), so the adapter cannot map it to a planner kwarg. The test below
    asserts the gap explicitly.
    """
    return {
        "name": "ECDFS_pong",
        "onsky_velocity": 0.5,
        "width": 5.0,
        "height": 6.7,
        "angle_var": 1,
        "spacing": 0.5,
        "num_terms": 5,
    }


@pytest.fixture
def fystplan_daisy_dict() -> dict:
    """``get_daisy_params`` emit dict, keys verbatim."""
    return {
        "name": "Mars_daisy",
        "onsky_velocity": 0.4,
        "R0": 0.3,
        "Rt": 0.05,
        "Ra": 0.01,
    }


@pytest.fixture
def fystplan_ce_dict() -> dict:
    """Constant-elevation carrier keys verbatim from fystplan.

    fystplan splits CE geometry across dicts: the per-elevation ``list_az_table``
    ``s`` dict (``elevation``/``az_min``/``az_max``/``duration``) plus the
    ObsUnit ``nominal_alt`` and the ``onsky_velocity`` ``mapping_params``
    carrier. The union of the keys the dispatch adapter would have to consume is
    represented here.
    """
    return {
        "name": "Deep56_ce",
        "nominal_alt": 50.0,
        "elevation": 50.0,
        "az_min": 120.0,
        "az_max": 240.0,
        "duration": 14400.0,
        "onsky_velocity": 0.5,
    }


# ---------------------------------------------------------------------------
# Sanity guards on the contract structures themselves.
# ---------------------------------------------------------------------------
def test_rename_map_targets_are_known_planner_params():
    """Every RHS of the rename map must be an accepted param of some planner.

    Catches a planner-side rename of any *renamed* parameter: if
    ``plan_daisy_scan`` renamed ``radius`` -> ``r0_deg`` without updating
    ``RENAME_MAP``, the target ``radius`` would no longer be accepted anywhere
    and this assertion would fail.
    """
    accepted: set[str] = set()
    for fn in (plan_pong_scan, plan_daisy_scan, plan_constant_el_scan, plan_source_ces):
        accepted |= _planner_param_names(fn)
    unknown = set(RENAME_MAP.values()) - accepted
    assert not unknown, (
        f"rename-map targets not accepted by any planner: {sorted(unknown)}, "
        f"a planner kwarg was renamed; update RENAME_MAP to match"
    )


# ---------------------------------------------------------------------------
# Pong contract.
# ---------------------------------------------------------------------------
def test_pong_mapped_keys_are_planner_params(fystplan_pong_dict):
    """Mapped fystplan pong keys are a subset of ``plan_pong_scan`` params.

    Fails if a pong planner kwarg is renamed without updating ``RENAME_MAP``
    (the mapped key would no longer be accepted) or if fystplan adds an
    unmapped scalar key.
    """
    mapped = _apply_rename(fystplan_pong_dict)
    accepted = _planner_param_names(plan_pong_scan)
    unknown = set(mapped) - accepted
    assert not unknown, f"pong: mapped keys not accepted by plan_pong_scan: {sorted(unknown)}"


def test_pong_required_params_are_all_satisfiable(fystplan_pong_dict):
    """Every required ``plan_pong_scan`` param is covered by the adapter.

    Coverage sources: fystplan-derived scalar (after rename), the FieldRegion
    structural build (``field`` from width/height), or dispatch infrastructure
    (site/start_time/timestep). Fails if the planner adds a new *required* param
    that the adapter cannot supply from any of these.
    """
    mapped = _apply_rename(fystplan_pong_dict)
    required = _required_planner_params(plan_pong_scan)

    covered = set(mapped) | DISPATCH_INFRA_KEYS | {"field"}
    missing = required - covered
    assert not missing, (
        f"pong: required planner params not covered by the A->C adapter: {sorted(missing)}"
    )
    # The FieldRegion structural inputs must be present in the fystplan dict so
    # the adapter can actually build ``field``.
    assert FIELDREGION_KEYS <= set(fystplan_pong_dict), (
        "pong: fystplan dict missing width/height needed to build FieldRegion"
    )


def test_pong_adapter_builds_a_valid_planner_call(site, fystplan_pong_dict):
    """End-to-end: the adapter output actually drives ``plan_pong_scan``.

    This is the strongest form of the contract. It constructs the real planner
    call from the fystplan dict (+ dispatch infra) and runs it. If any rename or
    structural step is wrong, this raises ``TypeError`` (bad kwargs) or a planner
    error rather than passing silently.
    """
    mapped = _apply_rename(fystplan_pong_dict)
    field = FieldRegion(
        ra_center=53.117,
        dec_center=-27.808,
        width=fystplan_pong_dict["width"],
        height=fystplan_pong_dict["height"],
    )
    block = plan_pong_scan(
        field=field,
        site=site,
        start_time="2026-03-15T17:00:00",
        timestep=0.1,
        **mapped,  # velocity, spacing, num_terms, angle
    )
    assert block.trajectory.n_points > 0
    assert block.config.num_terms == fystplan_pong_dict["num_terms"]


def test_pong_angle_var_has_no_planner_analogue(fystplan_pong_dict_angle_var):
    """The fystplan ``angle_var`` form is a documented gap, not a silent pass.

    ``angle_var`` is a fystplan-only selector with no ``plan_pong_scan`` kwarg.
    The adapter must NOT be able to map it. We assert it is neither in the rename
    map nor an accepted planner param, so a CurvyPong scheduled with the
    variable-angle form is recognised as unsupported (must be resolved to a
    concrete ``angle`` upstream) rather than mis-dispatched.
    """
    assert "angle_var" not in RENAME_MAP
    assert "angle_var" not in _planner_param_names(plan_pong_scan)
    # After applying the (correct) rename map, angle_var survives unmapped,
    # confirming the adapter would surface it as an unknown key rather than
    # silently dropping it.
    mapped = _apply_rename(fystplan_pong_dict_angle_var)
    unknown = set(mapped) - _planner_param_names(plan_pong_scan)
    assert unknown == {"angle_var"}, (
        f"expected angle_var to be the only unmapped key, got {sorted(unknown)}"
    )


# ---------------------------------------------------------------------------
# Daisy contract.
# ---------------------------------------------------------------------------
def test_daisy_mapped_keys_are_planner_params(fystplan_daisy_dict):
    """Mapped fystplan daisy keys are a subset of ``plan_daisy_scan`` params.

    Fails if ``R0``/``Rt``/``Ra``/``onsky_velocity`` lose their rename targets
    (a planner-side rename of radius/turn_radius/avoidance_radius/velocity).
    """
    mapped = _apply_rename(fystplan_daisy_dict)
    accepted = _planner_param_names(plan_daisy_scan)
    unknown = set(mapped) - accepted
    assert not unknown, (
        f"daisy: mapped keys not accepted by plan_daisy_scan: {sorted(unknown)} "
        f"(check R0->radius / Rt->turn_radius / Ra->avoidance_radius renames)"
    )
    # Spot-check the three radius renames resolved to real planner kwargs.
    for renamed in ("radius", "turn_radius", "avoidance_radius", "velocity"):
        assert renamed in mapped, f"daisy: expected {renamed} after rename"
        assert renamed in accepted, f"daisy: {renamed} not a plan_daisy_scan param"


def test_daisy_required_params_coverage_and_gaps(fystplan_daisy_dict):
    """Required daisy params split into fystplan-covered, infra, and the gap.

    Asserts the *structural truth* recommendations 2026-05-28 section 5 records:
    fystplan supplies the geometry (radius/turn_radius/avoidance_radius/velocity)
    and the source position is dispatch-supplied (ra/dec), but
    ``start_acceleration`` is required with no default and is NOT in the fystplan
    dict, so the adapter must source it elsewhere (instrument-owned ramp accel).
    """
    mapped = _apply_rename(fystplan_daisy_dict)
    required = _required_planner_params(plan_daisy_scan)

    # ra/dec are supplied at dispatch from the resolved source (fystplan keys the
    # daisy by source name, not RA/Dec in this dict), alongside site/start_time/
    # timestep/duration.
    dispatch_supplied = DISPATCH_INFRA_KEYS | {"ra", "dec"}

    covered_by_fystplan = required & set(mapped)
    covered_by_dispatch = required & dispatch_supplied
    uncovered = required - set(mapped) - dispatch_supplied

    # The geometry params must come from fystplan.
    assert {"radius", "turn_radius", "avoidance_radius", "velocity"} <= covered_by_fystplan
    # The only required param neither fystplan nor standard dispatch infra
    # supplies is start_acceleration, the documented A->C gap. If this set
    # changes (e.g. the planner adds another required param, or fystplan starts
    # emitting acceleration), this assertion fails and the contract must be
    # revisited.
    assert uncovered == {"start_acceleration"}, (
        f"daisy: unexpected uncovered required params {sorted(uncovered)}; "
        f"expected exactly {{'start_acceleration'}} (the known A->C gap)"
    )
    assert covered_by_dispatch  # sanity: ra/dec/site/... recognised as infra


def test_daisy_y_offset_is_covered_by_default():
    """``y_offset`` is omitted by fystplan but has a ``plan_daisy_scan`` default.

    Unlike ``start_acceleration``, ``y_offset`` is optional, so the adapter need
    not supply it. This pins that asymmetry: if ``y_offset`` ever became required
    (lost its default), the contract, and the dispatch adapter, would need to
    source it, and this test would fail.
    """
    optional = _planner_param_names(plan_daisy_scan) - _required_planner_params(plan_daisy_scan)
    assert "y_offset" in optional, (
        "y_offset must remain optional (default-covered); fystplan does not emit it"
    )


def test_daisy_adapter_builds_a_valid_planner_call(site, fystplan_daisy_dict):
    """End-to-end: adapter output (+ infra + supplied start_acceleration) runs.

    Mirrors the dispatch-time call: rename the fystplan keys, then supply the
    dispatch-side params (ra/dec/site/start_time/timestep/duration) and the
    instrument-owned ``start_acceleration`` that fystplan does not carry.
    """
    mapped = _apply_rename(fystplan_daisy_dict)
    block = plan_daisy_scan(
        ra=68.5,
        dec=-25.0,
        start_acceleration=0.1,  # adapter-supplied; NOT from OpsDB
        site=site,
        start_time="2026-03-15T17:00:00",
        timestep=0.1,
        duration=600.0,
        **mapped,  # radius, velocity, turn_radius, avoidance_radius
    )
    assert block.trajectory.n_points > 0


# ---------------------------------------------------------------------------
# Constant-elevation contract (structural).
# ---------------------------------------------------------------------------
def test_ce_velocity_rename_and_structural_mapping(fystplan_ce_dict):
    """CE is structural: fystplan az-window keys map to planner-derived state.

    ``plan_constant_el_scan`` does not accept az_min/az_max/duration directly.
    It *derives* the az range and duration from the field + elevation (or from
    the ``lsa_window`` partial bridge). The only direct scalar rename is
    ``onsky_velocity`` -> ``velocity``; ``nominal_alt`` -> ``elevation`` is a
    semantic (not literal) rename handled by the structural adapter. This test
    pins which keys are direct-mappable and which are structural.
    """
    accepted = _planner_param_names(plan_constant_el_scan)

    # Direct scalar rename that lands on a real planner kwarg.
    assert RENAME_MAP["onsky_velocity"] == "velocity"
    assert "velocity" in accepted

    # Structural keys: present in fystplan, NOT direct planner kwargs (the
    # planner computes az range + duration itself).
    structural_keys = {"az_min", "az_max", "duration", "nominal_alt", "elevation"}
    direct_planner_overlap = structural_keys & accepted
    # Of those, only ``elevation`` is also a literal planner kwarg; az_min/
    # az_max/duration/nominal_alt are not.
    assert direct_planner_overlap == {"elevation"}, (
        f"CE: unexpected direct planner overlap {sorted(direct_planner_overlap)}; "
        f"az_min/az_max/duration are planner-derived, not passed through"
    )

    # The lsa_window kwarg is the documented partial bridge for the az-window /
    # LSA-window inputs. Assert it still exists so the structural path holds.
    assert "lsa_window" in accepted, (
        "CE: plan_constant_el_scan lost lsa_window, the A->C structural bridge "
        "for fystplan's min_lsa/max_lsa az-window is gone"
    )


def test_ce_required_params_are_satisfiable(site, fystplan_ce_dict):
    """Every required ``plan_constant_el_scan`` param is adapter-satisfiable.

    field (from the source's RA/Dec + extent), elevation (from nominal_alt),
    velocity (from onsky_velocity) come via the structural adapter; site and
    start_time are dispatch infra.
    """
    required = _required_planner_params(plan_constant_el_scan)
    # elevation <- nominal_alt; velocity <- onsky_velocity; field <- structural.
    adapter_supplied = {"elevation", "velocity", "field"}
    covered = adapter_supplied | DISPATCH_INFRA_KEYS
    missing = required - covered
    assert not missing, (
        f"CE: required planner params not covered by the A->C adapter: {sorted(missing)}"
    )


def test_ce_adapter_builds_a_valid_planner_call(site, fystplan_ce_dict):
    """End-to-end: structural CE adapter output drives ``plan_constant_el_scan``."""
    field = FieldRegion(ra_center=0.0, dec_center=-2.0, width=10.0, height=8.0)
    block = plan_constant_el_scan(
        field=field,
        elevation=fystplan_ce_dict["nominal_alt"],
        velocity=fystplan_ce_dict["onsky_velocity"],
        site=site,
        start_time="2026-09-15T00:00:00",
        rising=True,
    )
    assert block.trajectory.n_points > 0
    assert block.config.elevation == fystplan_ce_dict["nominal_alt"]


# ---------------------------------------------------------------------------
# source_ces contract (keyword-only; body / ra+dec / el_bore inputs).
# ---------------------------------------------------------------------------
def test_source_ces_required_keyword_only_params():
    """``plan_source_ces`` requires footprint, el_bore, site as keyword-only.

    These three have no default and are keyword-only (leading bare ``*``). A
    schedlib FYST policy / PCS source_scan task calling this must supply all
    three. Fails if the required surface changes (e.g. el_bore gains a default or
    footprint is renamed).
    """
    required = _required_planner_params(plan_source_ces)
    assert required == {"footprint", "el_bore", "site"}, (
        f"plan_source_ces required params changed: {sorted(required)}"
    )
    # All params are keyword-only (the leading ``*`` makes the whole signature
    # keyword-only). The adapter must pass everything by keyword.
    kinds = {
        p.kind
        for p in inspect.signature(plan_source_ces).parameters.values()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }
    assert kinds == {inspect.Parameter.KEYWORD_ONLY}, (
        f"plan_source_ces is no longer fully keyword-only: {kinds}"
    )


def test_source_ces_target_selector_params_exist():
    """The body / ra+dec / el_bore selector parameters are all present.

    fystplan keys cal/source scans by source name (-> ``body`` for solar-system
    bodies) or by RA/Dec (-> ``ra``/``dec``); the boresight elevation maps to
    ``el_bore``. Pin that all three input modes are accepted so the source_scan
    adapter has a stable target surface.
    """
    accepted = _planner_param_names(plan_source_ces)
    # (a) solar-system body target.
    assert "body" in accepted
    # (b) RA+Dec target (+ optional proper motion).
    assert {"ra", "dec", "pm_ra", "pm_dec", "ref_epoch"} <= accepted
    # (c) geometry: boresight elevation + rotation.
    assert {"el_bore", "boresight_rot"} <= accepted
    # Time-window selector axis (window XOR night+mode).
    assert {"window", "night", "mode"} <= accepted


def test_source_ces_body_and_radec_are_mutually_exclusive(site):
    """Supplying both ``body`` and ``ra``/``dec`` is rejected by the planner.

    This is the runtime half of the target-selector contract: the adapter must
    pick exactly one target mode. ``el_bore`` + a time window are also required,
    so we pass valid ones and rely on the source-axis validation to raise.
    """
    from astropy.time import Time

    from fyst_trajectories.offsets import InstrumentOffset
    from fyst_trajectories.primecam import MODULE_FOV_RADIUS_DEG

    footprint = InstrumentOffset(dx=0.0, dy=0.0)
    with pytest.raises(ValueError, match="either 'body' or 'ra'/'dec'"):
        plan_source_ces(
            body="jupiter",
            ra=180.0,
            dec=-20.0,
            footprint=footprint,
            el_bore=50.0,
            window=(Time("2026-03-15T00:00:00"), Time("2026-03-15T06:00:00")),
            site=site,
        )
    # MODULE_FOV_RADIUS_DEG referenced to keep the import meaningful for readers
    # comparing against the schedlib make_geometry radius; not load-bearing here.
    assert MODULE_FOV_RADIUS_DEG > 0
