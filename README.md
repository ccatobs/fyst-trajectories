# fyst-trajectories

[![Documentation Status](https://app.readthedocs.org/projects/fyst-trajectories/badge/?version=latest)](https://fyst-trajectories.readthedocs.io/en/latest/)

Trajectory generation library for the Fred Young Submillimeter Telescope (FYST).
Wraps astropy with FYST-specific site coordinates, telescope limits, scan
pattern generators, focal-plane offsets, sun-avoidance policies, and an
offline observing-night overhead simulator.

**Documentation:** [fyst-trajectories.readthedocs.io](https://fyst-trajectories.readthedocs.io/en/latest/)

## Installation

```bash
pip install "fyst-trajectories @ git+https://github.com/ccatobs/fyst-trajectories.git"
```

## Development

```bash
git clone https://github.com/ccatobs/fyst-trajectories.git
cd fyst-trajectories
pip install -e ".[dev]"

pytest tests/
ruff check . && ruff format --check .
```

### Cross-validation tests

Cross-validation tests verify correctness against independent implementations.
They are gated behind the `--run-slow` flag:

```bash
pytest tests/ --run-slow
```
