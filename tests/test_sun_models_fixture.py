"""Offline half of the sun-model record-replay parity harness.

Runs ALWAYS, with the shared sun-avoidance library absent or present: it
verifies the checked-in fixture's integrity (schema, the CAD table's
properties, the SHA pins matching the module constants), regenerates the
``"scalar"`` model's verdicts (which need no library) against the recorded
ones, and locks the cross-model containment invariants that the recorded
verdicts must satisfy. The live half (``test_sun_models_live.py``)
regenerates the library-backed verdicts and is the drift detector.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.time import Time

from fyst_trajectories import get_fyst_site
from fyst_trajectories.sun_models import (
    CAD_TABLE_SHA256,
    SUN_AVOIDANCE_PINNED_SHA,
    _table_sha256,
    load_avoidance_data,
    make_slew_safe,
    make_sun_safe,
)

FIXTURE = Path(__file__).parent / "data" / "sun_avoidance_parity_e6fa12a.npz"

MODEL_KEYS = ("scalar", "cone45", "cone50", "cad", "cad_msa0", "cad_island")


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


def test_fixture_schema(fx):
    for key in MODEL_KEYS:
        assert fx[f"verdict_{key}"].shape == (8, 29 * 72)
        assert fx[f"verdict_{key}"].dtype == bool
        assert fx[f"threshold_{key}"].shape == (8, 29 * 72)
    assert fx["az"].shape == (72,)
    assert fx["el"].shape == (29,)
    assert 90.0 in fx["el"]  # the exactly-at-ceiling sample the nudge guards
    assert fx["az"].min() == -180.0  # negative encoder wraps are exercised
    assert fx["times_iso"].shape == (8,)


def test_fixture_pins_match_module(fx):
    assert str(fx["pinned_sha"]) == SUN_AVOIDANCE_PINNED_SHA
    assert str(fx["cad_table_sha256"]) == CAD_TABLE_SHA256


def test_cad_table_digest_folds_crlf(tmp_path):
    """An LF and a CRLF copy of one table hash to the same digest.

    A checkout that converts line endings (``core.autocrlf`` on Windows)
    stores the shipped table with CRLF, so the pin must hold on either.
    """
    rows = b"gamma,delta\n0.0,50.0\n360.0,50.0\n"
    lf = tmp_path / "lf.csv"
    crlf = tmp_path / "crlf.csv"
    lf.write_bytes(rows)
    crlf.write_bytes(rows.replace(b"\n", b"\r\n"))
    assert _table_sha256(lf) == _table_sha256(crlf) == hashlib.sha256(rows).hexdigest()


def test_cad_table_properties(fx):
    gamma, delta = fx["cad_gamma"], fx["cad_delta"]
    assert len(gamma) == 73
    assert np.all(np.diff(gamma) > 0)  # sorted, as the library's searchsorted assumes
    assert gamma[0] == 0.0
    assert gamma[-1] == 360.0
    assert delta.min() == 50.0  # the CAD table floor
    assert delta.max() == 90.0


def test_scalar_verdicts_regenerate(fx):
    """The no-library model reproduces its recorded verdicts bit-exactly.

    The scalar row is recorded against explicitly pinned 45/50 radii (not
    the site defaults) so this fixture only ever changes for library or
    adapter reasons, never for a site-default policy bump.
    """
    pinned_site = get_fyst_site(sun_exclusion_radius=45.0, sun_warning_radius=50.0)
    predicate = make_sun_safe("scalar", site=pinned_site)
    A, E = np.meshgrid(fx["az"], fx["el"])
    A, E = A.ravel(), E.ravel()
    for i, iso in enumerate(fx["times_iso"]):
        t = Time(str(iso), scale="utc")
        assert np.array_equal(predicate.batch(A, E, t), fx["verdict_scalar"][i])
        assert np.array_equal(predicate.threshold(A, E, t), fx["threshold_scalar"][i])
    assert np.all(fx["threshold_scalar"] == 45.0)


def test_cross_model_containment(fx):
    """Recorded verdicts satisfy the structural safety orderings."""
    v45, v50 = fx["verdict_cone45"], fx["verdict_cone50"]
    cad, msa0, island = fx["verdict_cad"], fx["verdict_cad_msa0"], fx["verdict_cad_island"]
    # The adapter's cone-45 IS our scalar model (the parity claim, as recorded).
    assert np.array_equal(fx["verdict_scalar"], v45)
    # A wider cone only removes safe samples.
    assert np.all(~v50 | v45)
    # CAD-safe implies cone-50-safe: the zone never reaches inside its own floor.
    assert np.all(~cad | v50)
    # The night exemption only ADDS safe samples; the island check only removes.
    assert np.all(~cad | msa0)
    assert np.all(~island | cad)
    # And each ordering is strict somewhere (the fixture actually exercises it).
    assert (~v50 & v45).any()
    assert (~cad & v50).any()
    assert (~cad & msa0).any()
    assert (~island & cad).any()


def test_make_sun_safe_validation_offline():
    with pytest.raises(ValueError, match="radius"):
        make_sun_safe("scalar", radius=45.0)
    with pytest.raises(ValueError, match="min_solar_altitude"):
        make_sun_safe("scalar", min_solar_altitude=float("nan"))
    with pytest.raises(ValueError, match="maxoffset"):
        make_sun_safe("cad", maxoffset=-1.0)
    with pytest.raises(ValueError, match="Unknown avoidance model") as excinfo:
        make_sun_safe("bogus")
    # The message names every model this entry point accepts. "scalar" is
    # handled before the delegation, so the delegate's own message would list
    # only two of the three: a caller who mistypes "Cad" must still see all three.
    for name in ("scalar", "cad", "cone"):
        assert repr(name) in str(excinfo.value)
    with pytest.raises(ValueError, match="radius"):
        make_sun_safe("cone")  # cone requires a radius, with or without the library


@pytest.mark.parametrize(
    "build",
    [
        lambda: make_sun_safe("cad"),
        lambda: make_sun_safe("cone", radius=50.0),
        lambda: make_slew_safe(),
        lambda: load_avoidance_data("cad"),
    ],
    ids=["make_sun_safe-cad", "make_sun_safe-cone", "make_slew_safe-default", "load-cad"],
)
def test_missing_library_error(monkeypatch, build):
    """Absent library => ModuleNotFoundError carrying the pinned install command.

    ``RuntimeError`` is kept for a CAD table that fails the pin, which needs a
    re-pin rather than an install.
    """
    monkeypatch.setitem(sys.modules, "sun_avoidance", None)
    with pytest.raises(ModuleNotFoundError, match=SUN_AVOIDANCE_PINNED_SHA[:12]) as excinfo:
        build()
    assert excinfo.value.name == "sun_avoidance"
    assert not isinstance(excinfo.value, RuntimeError)


def test_input_guards_offline():
    """Rank and elevation-range guards hold on the no-library model too."""
    predicate = make_sun_safe("scalar")
    t = Time("2026-11-15T16:00:00", scale="utc")
    with pytest.raises(ValueError, match="scalar or 1-D"):
        predicate.batch(np.zeros((2, 2)), np.zeros((2, 2)), t)
    with pytest.raises(ValueError, match=r"\[-90, 90\]"):
        predicate.batch([120.0], [100.0], t)


def test_scalar_respects_disabled_site():
    predicate = make_sun_safe("scalar", site=get_fyst_site(sun_avoidance_enabled=False))
    t = Time("2026-11-15T16:00:00", scale="utc")
    assert predicate.batch([0.0, 90.0, 180.0], [45.0, 45.0, 45.0], t).all()
    assert np.all(predicate.threshold([0.0], [45.0], t) == 0.0)
    assert predicate.describe == "scalar (disabled)"
