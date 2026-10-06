"""The documentation figures regenerate, at their registered sizes, and are all used.

``docs/make_figures.py`` rebuilds every generated figure under ``docs/figures/``
from the library's own plotting functions. These tests run each registered
builder into a temporary directory and check the file exists, has the
registered pixel size and is not blank. They deliberately do not compare bytes
with the committed PNGs: rasterisation differs across matplotlib and FreeType
versions, so byte equality would fail on a clean machine for no reason.
Regenerating the committed figures after a behaviour change is a release
checklist step.

Tier: each render is classified by the imports of the builder it runs, at the
parametrization site. The two night builders import the simulator and are
marked ``offline``; every other test here draws only library-tier plots.
"""

import importlib.util
import inspect
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from _sun_stubs import HAVE_SUN_AVOIDANCE  # noqa: E402
from _tiers import imported_names, is_simulator_tier  # noqa: E402
from matplotlib.image import imread  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent / "docs"
SCRIPT = DOCS / "make_figures.py"
FIGURES = DOCS / "figures"

#: Hand-authored diagram exports: committed beside their .html source, not built
#: by the script.
DIAGRAMS = ("dispatch_flow", "sim_live_lanes")

#: Pixel tolerance on each axis (the figures use a fixed size, never a tight box).
SIZE_TOLERANCE_PX = 2

_FIGURE_DIRECTIVE = re.compile(r"^\.\. figure:: (\S+)\s*$", re.M)


def _load_script():
    spec = importlib.util.spec_from_file_location("make_figures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Registered first: the frozen dataclass in the script resolves its
    # string annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MAKE_FIGURES = _load_script()
REGISTRY = MAKE_FIGURES.REGISTRY


def _builder_is_simulator_tier(spec) -> bool:
    """Report whether a figure's builder imports the simulator tier."""
    source = textwrap.dedent(inspect.getsource(spec.build))
    return any(is_simulator_tier(name) for name in imported_names(source))


def _render_param(spec):
    marks = [pytest.mark.offline] if _builder_is_simulator_tier(spec) else []
    return pytest.param(spec, id=spec.name, marks=marks)


def _figure_directives() -> list[tuple[Path, Path]]:
    """Every ``.. figure::`` directive in the docs, as (page, resolved target)."""
    directives = []
    for page in sorted([*DOCS.glob("*.rst"), *DOCS.glob("api/*.rst")]):
        for target in _FIGURE_DIRECTIVE.findall(page.read_text(encoding="utf-8")):
            # Sphinx reads a leading slash as the source root, anything else
            # as relative to the page.
            base = DOCS if target.startswith("/") else page.parent
            directives.append((page, (base / target.lstrip("/")).resolve()))
    return directives


def _assert_size(path: Path, size_px: tuple[int, int]) -> None:
    height, width = imread(path).shape[:2]
    assert abs(width - size_px[0]) <= SIZE_TOLERANCE_PX, (path.name, width, size_px)
    assert abs(height - size_px[1]) <= SIZE_TOLERANCE_PX, (path.name, height, size_px)


def test_registry_names_are_unique():
    names = [spec.name for spec in REGISTRY]
    assert len(names) == len(set(names))
    assert not set(names) & set(DIAGRAMS)


def test_the_night_builders_are_classified_as_simulator_tier():
    # The render marks rely on the import scan finding the builders'
    # function-local simulator imports.
    simulator = {spec.name for spec in REGISTRY if _builder_is_simulator_tier(spec)}
    assert simulator == {"night_gantt", "calibration_night"}


@pytest.mark.parametrize("spec", [_render_param(spec) for spec in REGISTRY])
def test_figure_renders_at_its_registered_size(spec, tmp_path):
    if spec.needs_sun_avoidance and not HAVE_SUN_AVOIDANCE:
        pytest.skip(f"{spec.name} draws a CAD panel, which needs the shared sun_avoidance library")
    path = MAKE_FIGURES.render(spec, tmp_path)
    assert path == tmp_path / spec.filename
    _assert_size(path, spec.size_px)
    pixels = imread(path)[..., :3]
    assert pixels.std() > 0.02, f"{spec.name} rendered (nearly) blank"


@pytest.mark.parametrize("spec", REGISTRY, ids=lambda spec: spec.name)
def test_committed_figure_has_its_registered_size(spec):
    _assert_size(FIGURES / spec.filename, spec.size_px)


@pytest.mark.parametrize("name", DIAGRAMS)
def test_diagram_export_sits_beside_its_source(name):
    png = FIGURES / f"{name}.png"
    if not png.is_file():
        pytest.skip(f"{png.name} is not committed yet, so there is no export to pair")
    assert (FIGURES / f"{name}.html").is_file(), f"{png.name} has no {name}.html beside it"


def test_every_committed_png_is_a_known_figure():
    known = {spec.filename for spec in REGISTRY} | {f"{name}.png" for name in DIAGRAMS}
    unknown = {path.name for path in FIGURES.glob("*.png")} - known
    assert not unknown, f"not built by make_figures.py and not a diagram: {sorted(unknown)}"


def test_every_committed_png_is_referenced_by_a_figure_directive():
    # Sphinx fails on a missing file under -W but says nothing about an orphan.
    referenced = {target for _, target in _figure_directives()}
    orphans = sorted(
        path.name for path in FIGURES.glob("*.png") if path.resolve() not in referenced
    )
    assert not orphans, f"committed but no page shows them: {orphans}"


def test_every_figure_directive_points_at_a_committed_png():
    directives = _figure_directives()
    assert directives, "no figure directive found; the scan pattern has drifted"
    missing = [
        f"{page.relative_to(DOCS).as_posix()}: {target.name}"
        for page, target in directives
        if not target.is_file() or target.parent != FIGURES.resolve()
    ]
    assert not missing, f"figure targets missing from docs/figures/: {missing}"


def test_script_runs_from_the_command_line(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--only", "primecam_footprint", "--out-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == ["primecam_footprint.png"]
