"""The suite's ``--fail-on-skip`` option turns a run that skipped anything into a failure."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent

_PASSING_AND_SKIPPED = """
import pytest


def test_passes():
    pass


@pytest.mark.skip(reason="skipped on purpose")
def test_skipped():
    pass
"""

_PASSING_AND_EXPECTED_FAILURE = """
import pytest


def test_passes():
    pass


@pytest.mark.xfail(reason="expected")
def test_expected_failure():
    assert False
"""

_SKIPPED_AT_COLLECTION = """
import pytest

pytest.importorskip("a_module_that_is_not_installed")


def test_never_collected():
    pass
"""


def _run(tmp_path, *options):
    """Run pytest on the two modules in ``tmp_path`` with this suite's conftest as a plugin."""
    env = dict(os.environ)
    inherited = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{TESTS_DIR}{os.pathsep}{inherited}" if inherited else str(TESTS_DIR)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-c", str(tmp_path / "pytest.ini")]
        + ["-p", "conftest", "-p", "no:cacheprovider", "-q", *options, str(tmp_path)],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=120,
    )


@pytest.fixture
def skipping_run_dir(tmp_path):
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "test_tiny.py").write_text(_PASSING_AND_SKIPPED, encoding="utf-8")
    (tmp_path / "test_absent_dependency.py").write_text(_SKIPPED_AT_COLLECTION, encoding="utf-8")
    return tmp_path


def test_a_skip_passes_without_the_option_and_fails_with_it(skipping_run_dir):
    plain = _run(skipping_run_dir)
    assert plain.returncode == pytest.ExitCode.OK, plain.stdout + plain.stderr

    strict = _run(skipping_run_dir, "--fail-on-skip")
    assert strict.returncode == pytest.ExitCode.TESTS_FAILED, strict.stdout + strict.stderr
    summary = strict.stdout.split("skipped under --fail-on-skip", 1)[1]
    assert "test_tiny.py::test_skipped" in summary
    assert "test_absent_dependency.py" in summary
    assert "test_tiny.py::test_passes" not in summary


def test_an_expected_failure_is_not_a_skip(tmp_path):
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "test_tiny.py").write_text(_PASSING_AND_EXPECTED_FAILURE, encoding="utf-8")

    strict = _run(tmp_path, "--fail-on-skip")
    assert strict.returncode == pytest.ExitCode.OK, strict.stdout + strict.stderr
    assert "skipped under --fail-on-skip" not in strict.stdout
