# fyst-trajectories

[![Documentation Status](https://app.readthedocs.org/projects/fyst-trajectories/badge/?version=latest)](https://fyst-trajectories.readthedocs.io/en/latest/)

Trajectory generation library for the Fred Young Submillimeter Telescope (FYST).
Wraps astropy with FYST-specific site coordinates, telescope limits, scan
pattern generators, focal-plane offsets, sun-avoidance policies, an offline
observing-night overhead simulator, and a calibration-night planner.

**Documentation:** [fyst-trajectories.readthedocs.io](https://fyst-trajectories.readthedocs.io/en/latest/)

## Installation

Pin a release tag (`v0.10.0` is the latest; the
[changelog](https://fyst-trajectories.readthedocs.io/en/latest/changelog.html)
lists what each release changes).

```bash
pip install "fyst-trajectories @ git+https://github.com/ccatobs/fyst-trajectories.git@v0.10.0"
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

## License

BSD 3-Clause; see [LICENSE](LICENSE).
