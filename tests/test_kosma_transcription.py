"""Check the inline KOSMA focal-plane model against the file it reproduces.

``test_kosma_cross_validation.py`` reproduces ``mm_to_arcsec`` and
``compute_focal_plane`` from ``utils/tests_with_focalplane.py`` in
``ccatobs/kosma-ocs-translator`` at the commit ``KOSMA_SOURCE_COMMIT`` records.
These tests read that file at that commit with ``git show``, from a clone of
the repository beside this checkout (or, from a git worktree, beside the
checkout that owns it), run the file's own definitions and compare them with
the reproduction. The file cannot be imported: at import it looks for a
running KOSMA focal-plane server and exits. So only its model definitions are
compiled and run, without its imports or its module-level code.

Without git, the clone or that commit in it, the tests that read the file skip
and say what is missing; nothing is fetched.
"""

import ast
import itertools
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
import test_kosma_cross_validation as reproduction

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CLONE_NAME = reproduction.KOSMA_SOURCE_REPOSITORY.rsplit("/", 1)[1]

#: The top-level definitions that make up the file's model.
_MODEL_NAMES = ("DEG2RAD", "mm_to_arcsec", "TelescopeData", "InstrumentData", "compute_focal_plane")

_PORTS = ("Right", "Left", "Cassegrain", "Unknown")  # the last takes the fallback branch
_ELEVATIONS = (0.0, 20.0, 45.0, 60.0, 85.0, 90.0)
_FOCAL_PLANE_ROTATIONS = (0.0, 10.0, -33.0)
_FLANGE_ROTATIONS = (0.0, 15.0)
# (10.1, -20.0) is the file's test_focal_plane_static; 461.3 mm is the Prime-Cam inner-ring radius.
_REFERENCE_POSITIONS = (
    (0.0, 0.0),
    (1.0, 0.0),
    (0.0, 1.0),
    (10.1, -20.0),
    (-35.0, 12.5),
    (461.3, 0.0),
)


def _git(cwd: Path, *args: str) -> str | None:
    """Return the output of ``git <args>`` run in ``cwd``, or None if git fails or is absent."""
    try:
        result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=False)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.decode("utf-8")


def _clone_candidates() -> list[Path]:
    """Return where a translator clone may sit: beside this checkout, then beside its owner.

    From a git worktree the owner is the checkout whose ``.git`` directory the
    worktree shares; from that checkout itself both candidates are the same.
    """
    checkouts = [_REPO_ROOT]
    common_dir = _git(_REPO_ROOT, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if common_dir is not None:
        checkouts.append(Path(common_dir.strip()).parent)
    return list(dict.fromkeys(checkout.parent / _CLONE_NAME for checkout in checkouts))


@pytest.fixture(scope="module")
def kosma_source() -> str:
    """Return the reproduced file at the recorded commit; skip, naming the gap, without it."""
    commit = reproduction.KOSMA_SOURCE_COMMIT
    candidates = _clone_candidates()
    clone = next((path for path in candidates if (path / ".git").exists()), None)
    if clone is None:
        looked = ", ".join(str(path) for path in candidates)
        pytest.skip(
            f"no clone of {reproduction.KOSMA_SOURCE_REPOSITORY} at {looked}: clone "
            f"https://github.com/{reproduction.KOSMA_SOURCE_REPOSITORY} beside this "
            "checkout to check the KOSMA reproduction against its source"
        )
    if _git(clone, "--version") is None:
        pytest.skip(f"git is not available to read {reproduction.KOSMA_SOURCE_FILE} from {clone}")
    if _git(clone, "cat-file", "-e", f"{commit}^{{commit}}") is None:
        pytest.skip(f"the clone at {clone} lacks commit {commit}: run git -C {clone} fetch")
    source = _git(clone, "show", f"{commit}:{reproduction.KOSMA_SOURCE_FILE}")
    assert source is not None, f"{reproduction.KOSMA_SOURCE_FILE} is not in commit {commit}"
    return source


def _definition_name(node: ast.stmt) -> str | None:
    """Return the name a top-level function, class or single-name assignment defines."""
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name
    if (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ):
        return node.targets[0].id
    return None


def _discard(*args: object, **kwargs: object) -> None:
    """Stand in for ``print``, which ``compute_focal_plane`` calls on every evaluation."""


@pytest.fixture(scope="module")
def kosma_model(kosma_source: str) -> dict:
    """Compile the file's model definitions alone and return their namespace."""
    nodes = [
        node for node in ast.parse(kosma_source).body if _definition_name(node) in _MODEL_NAMES
    ]
    found = sorted(_definition_name(node) for node in nodes)
    assert found == sorted(_MODEL_NAMES), f"the file defines {found}, expected {_MODEL_NAMES}"
    namespace = {
        "__name__": "kosma_focal_plane_model",
        "dataclass": dataclass,
        "math": math,
        "print": _discard,
    }
    filename = f"{reproduction.KOSMA_SOURCE_COMMIT[:7]}:{reproduction.KOSMA_SOURCE_FILE}"
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), namespace)
    return namespace


def _static_case_arguments(source: str) -> dict[str, dict[str, object]]:
    """Return the keyword arguments of the data classes the file's static case builds.

    The static case is ``test_focal_plane_static``, which sets every input as a
    literal; ``test_focal_plane_with_kio`` reads them from a running KOSMA system.
    """
    static = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "test_focal_plane_static"
    )
    return {
        node.func.id: {keyword.arg: ast.literal_eval(keyword.value) for keyword in node.keywords}
        for node in ast.walk(static)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("TelescopeData", "InstrumentData")
    }


def test_module_docstring_names_the_recorded_source():
    """The reproduction's docstring names the repository, file and commit it records."""
    for recorded in (
        reproduction.KOSMA_SOURCE_REPOSITORY,
        reproduction.KOSMA_SOURCE_FILE,
        reproduction.KOSMA_SOURCE_COMMIT,
    ):
        assert recorded in reproduction.__doc__


def test_constants_are_the_static_case_values(kosma_source):
    """The plate scale and focal-length correction are those ``test_focal_plane_static`` sets."""
    arguments = _static_case_arguments(kosma_source)
    assert arguments["TelescopeData"]["tel_plate_scale"] == reproduction.KOSMA_PLATE_SCALE
    assert (
        arguments["InstrumentData"]["instr_focal_length_correction"]
        == reproduction.KOSMA_FOCAL_LENGTH_CORRECTION
    )


def test_mm_to_arcsec_matches_source(kosma_model):
    """The reproduction converts focal-plane millimetres to arcsec as the file does."""
    for mm in (-461.3, -1.0, 0.0, 0.5, 10.1, 461.3):
        expected = kosma_model["mm_to_arcsec"](mm, reproduction.KOSMA_FOCAL_LENGTH)
        assert reproduction.kosma_mm_to_arcsec(mm) == pytest.approx(expected, rel=1e-12), mm


@pytest.mark.parametrize("port", _PORTS)
def test_focal_plane_offset_matches_source(kosma_source, kosma_model, port):
    """The reproduction gives the file's focal length and focal-plane offsets.

    The file's ``TelescopeData`` is built with the nonzero
    ``tel_angle_focal_plane`` of its ``test_focal_plane_static``, which the
    reproduction has no parameter for, so the match also shows
    ``compute_focal_plane`` does not read it.
    """
    static_angle = _static_case_arguments(kosma_source)["TelescopeData"]["tel_angle_focal_plane"]
    for el, fp_rotation, flange, (x_mm, y_mm) in itertools.product(
        _ELEVATIONS, _FOCAL_PLANE_ROTATIONS, _FLANGE_ROTATIONS, _REFERENCE_POSITIONS
    ):
        telescope = kosma_model["TelescopeData"](
            tel_telescope="FYST",
            tel_angle_focal_plane=static_angle,
            tel_elv_act=el,
            tel_plate_scale=reproduction.KOSMA_PLATE_SCALE,
        )
        instrument = kosma_model["InstrumentData"](
            instr_name="test",
            instr_boresight_offset_x=0.0,
            instr_boresight_offset_y=0.0,
            instr_elevation_axis_x=0.0,
            instr_elevation_axis_y=0.0,
            instr_flange_rotation=flange,
            instr_focal_length_correction=reproduction.KOSMA_FOCAL_LENGTH_CORRECTION,
            instr_port_name=port,
            instr_rotator_axis_x=0.0,
            instr_rotator_axis_y=0.0,
            instr_focal_plane_rotation=fp_rotation,
            instr_reference_pos_x=x_mm,
            instr_reference_pos_y=y_mm,
            boresight_reference_pos_x=0.0,
            boresight_reference_pos_y=0.0,
        )
        result = kosma_model["compute_focal_plane"](telescope, instrument)
        offset = reproduction.kosma_focal_plane_offset(
            x_mm,
            y_mm,
            el,
            port=port,
            instr_focal_plane_rotation=fp_rotation,
            instr_flange_rotation=flange,
        )
        case = (
            f"el={el} instr_focal_plane_rotation={fp_rotation} "
            f"instr_flange_rotation={flange} reference=({x_mm}, {y_mm})"
        )
        assert result["focal_length_mm"] == pytest.approx(
            reproduction.KOSMA_FOCAL_LENGTH, rel=1e-12
        ), case
        assert offset == pytest.approx(
            (result["fp_offset_x_arcsec"], result["fp_offset_y_arcsec"]), rel=1e-12, abs=1e-9
        ), case
