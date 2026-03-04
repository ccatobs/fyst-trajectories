# fyst-pointing

Pointing library for the Fred Young Submillimeter Telescope (FYST).

## Installation

```bash
pip install "fyst-pointing @ git+https://github.com/ccatobs/fyst-pointing.git"
```

[PENDING] For usage documentation, see [fyst-pointing.readthedocs.io](https://fyst-pointing.readthedocs.io/).

## Development

```bash
git clone https://github.com/ccatobs/fyst-pointing.git
cd fyst-pointing
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
