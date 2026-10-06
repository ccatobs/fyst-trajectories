"""Tests for the private read-only ``dict`` the frozen value types carry."""

import ast
import copy
import dataclasses
import json
import pickle
from pathlib import Path

import pytest
import yaml

import fyst_trajectories._readonly
from fyst_trajectories._readonly import ReadOnlyDict

_MUTATIONS = {
    "setitem": lambda m: m.__setitem__("a", 2.0),
    "delitem": lambda m: m.__delitem__("a"),
    "ior": lambda m: m.__ior__({"b": 2.0}),
    "clear": lambda m: m.clear(),
    "pop": lambda m: m.pop("a"),
    "pop_default": lambda m: m.pop("missing", None),
    "popitem": lambda m: m.popitem(),
    "setdefault": lambda m: m.setdefault("b", 2.0),
    "update": lambda m: m.update(b=2.0),
}


@pytest.mark.parametrize("mutate", _MUTATIONS.values(), ids=_MUTATIONS.keys())
def test_every_mutator_raises(mutate):
    m = ReadOnlyDict({"a": 1.0})
    with pytest.raises(TypeError, match="read-only; copy it with dict"):
        mutate(m)
    assert m == {"a": 1.0}


def test_augmented_assignment_raises():
    m = ReadOnlyDict({"a": 1.0})
    with pytest.raises(TypeError, match="read-only"):
        m |= {"b": 2.0}
    with pytest.raises(TypeError, match="read-only"):
        m["a"] = 2.0
    with pytest.raises(TypeError, match="read-only"):
        del m["a"]


def test_is_a_dict_equal_to_a_plain_dict():
    m = ReadOnlyDict({"a": 1.0, "b": [1, 2]})
    assert isinstance(m, dict)
    assert m == {"a": 1.0, "b": [1, 2]}
    assert repr(m) == "{'a': 1.0, 'b': [1, 2]}"


def test_dict_copy_is_mutable():
    m = ReadOnlyDict({"a": 1.0})
    for plain in (dict(m), m.copy(), m | {}):
        assert type(plain) is dict
        plain["a"] = 2.0
        assert plain == {"a": 2.0}
    assert m == {"a": 1.0}


@pytest.mark.parametrize(
    "duplicate",
    [
        lambda m: pickle.loads(pickle.dumps(m)),
        copy.copy,
        copy.deepcopy,
    ],
    ids=["pickle", "copy", "deepcopy"],
)
def test_copies_keep_the_type_and_the_contents(duplicate):
    m = ReadOnlyDict({"a": 1.0, "nested": {"b": 2}})
    dup = duplicate(m)
    assert type(dup) is ReadOnlyDict
    assert dup == m
    with pytest.raises(TypeError, match="read-only"):
        dup["a"] = 2.0


def test_json_and_yaml_after_dict():
    m = ReadOnlyDict({"a": 1.0, "b": "x"})
    assert json.loads(json.dumps(m)) == {"a": 1.0, "b": "x"}
    # The safe YAML dumper knows only plain containers: convert first.
    with pytest.raises(yaml.representer.RepresenterError):
        yaml.safe_dump(m)
    assert yaml.safe_load(yaml.safe_dump(dict(m))) == {"a": 1.0, "b": "x"}


def test_asdict_of_an_owner():
    @dataclasses.dataclass(frozen=True)
    class Owner:
        params: dict = dataclasses.field(default_factory=dict, hash=False)

    owner = Owner(ReadOnlyDict({"a": 1.0}))
    assert dataclasses.asdict(owner) == {"params": {"a": 1.0}}
    assert json.dumps(dataclasses.asdict(owner)) == '{"params": {"a": 1.0}}'


def test_module_imports_nothing():
    """The module sits below every module that uses it, so it cannot close a cycle."""
    source = Path(fyst_trajectories._readonly.__file__).read_text(encoding="utf-8")
    imports = [
        ast.unparse(node)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import | ast.ImportFrom)
    ]
    assert imports == []
