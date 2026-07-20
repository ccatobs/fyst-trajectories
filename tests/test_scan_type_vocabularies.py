"""Guard tests pinning the intentional scan-type vocabulary asymmetry.

The planning and overhead subpackages each carry a scan-type "union" and a
scan-type "table", and the two subpackages are mirror-inverted about
``source_ces`` by design (see the "Scan-type vocabularies" section in
docs/overhead_integration.rst). These tests fail closed if a future change
accidentally equalizes the two sides.
"""

import typing

import pytest

from fyst_trajectories import list_patterns
from fyst_trajectories.overhead.models import (
    _SCAN_TYPE_TO_SCAN_PARAM_KEYS,
    ObservingPatch,
    ScanParamsDict,
    SourceCESScanParams,
)
from fyst_trajectories.planning._types import (
    _SCAN_TYPE_TO_KEYS,
    ComputedParams,
    SourceCESComputedParams,
    validate_computed_params,
)


def test_source_ces_computed_params_message_signposts():
    """A source_ces computed_params check points the caller at the right validator.

    Passing ``"source_ces"`` raises ``KeyError`` (unchanged exception type)
    whose message names the public ``validate_scan_params`` entry point, so a
    caller who reaches this corner is directed to the validator that does
    accept source-CES params.
    """
    with pytest.raises(KeyError, match="validate_scan_params"):
        validate_computed_params({}, "source_ces")


def test_scan_type_vocabularies_are_intentionally_inverted():
    """The planning and overhead scan-type vocabularies are mirror-inverted.

    ``source_ces`` sits in the planning union and the overhead table, but not
    in the planning table or the overhead union. This asymmetry is deliberate;
    equalizing either side is the trap this test guards against.
    """
    # Tables (validator call sites): planning excludes, overhead includes.
    assert "source_ces" not in _SCAN_TYPE_TO_KEYS
    assert "source_ces" in _SCAN_TYPE_TO_SCAN_PARAM_KEYS

    # Unions (TypedDict attribute annotations): planning includes, overhead excludes.
    assert SourceCESComputedParams in typing.get_args(ComputedParams)
    assert SourceCESScanParams not in typing.get_args(ScanParamsDict)

    # The overhead science-scan-type guard rejects source_ces entirely.
    with pytest.raises(ValueError, match="scan_type must be"):
        ObservingPatch(
            name="src",
            ra_center=0.0,
            dec_center=0.0,
            width=1.0,
            height=1.0,
            scan_type="source_ces",
            velocity=0.5,
        )

    # source_ces is planner-only, never a registered pattern.
    assert "source_ces" not in list_patterns()
    assert len(list_patterns()) == 9
