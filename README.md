# fyst-trajectories

Trajectory generation library for the Fred Young Submillimeter Telescope (FYST).

## Installation

```bash
pip install "fyst-trajectories @ git+https://github.com/ccatobs/fyst-trajectories.git"
```

[PENDING] For usage documentation, see [fyst-trajectories.readthedocs.io](https://fyst-trajectories.readthedocs.io/).

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
