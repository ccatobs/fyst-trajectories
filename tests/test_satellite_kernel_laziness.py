"""Lazy-load and dependency guards for the satellite (Titan) ephemeris path.

These pin the two invariants the satellite path must hold: importing the library
never touches a kernel (no import-time I/O), and the satellite dependency
(``jplephem``) stays an optional ``[ephemeris]`` extra rather than leaking into
the core dependency set.
"""

import subprocess
import sys
from pathlib import Path

import pytest


def test_import_with_bogus_kernel_env_is_lazy():
    """A bogus FYST_SATELLITE_KERNEL must not error at import (kernel load is lazy).

    The kernel is only opened when a satellite trajectory is actually requested,
    so importing the package with a non-existent kernel path configured must
    succeed and must not import ``jplephem``.
    """
    code = (
        "import os, sys; "
        "os.environ['FYST_SATELLITE_KERNEL'] = '/no/such/titan_kernel.bsp'; "
        "import fyst_trajectories, fyst_trajectories.coordinates; "
        "assert 'jplephem' not in sys.modules, 'jplephem imported at import time'"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_jplephem_is_optional_not_core():
    """``jplephem`` stays under the ``[ephemeris]`` extra; core deps stay clean."""
    tomllib = pytest.importorskip("tomllib")  # stdlib >= 3.11; skipped on 3.10
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with open(pyproject, "rb") as fh:
        cfg = tomllib.load(fh)

    core = " ".join(cfg["project"]["dependencies"]).lower()
    assert "jplephem" not in core
    assert "skyfield" not in core
    assert "astroquery" not in core

    extras = cfg["project"]["optional-dependencies"]
    assert any("jplephem" in dep for dep in extras.get("ephemeris", []))
