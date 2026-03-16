Installation
============

**Requires Python 3.10 or higher.**

From GitHub::

    pip install "fyst-pointing @ git+https://github.com/ccatobs/fyst-pointing.git"

Development install
-------------------

Clone and install in editable mode with development extras::

    git clone https://github.com/ccatobs/fyst-pointing.git
    cd fyst-pointing
    pip install -e ".[dev]"

Running tests
-------------

Fast tests::

    pytest tests/

Linting::

    ruff check . && ruff format --check .

Cross-validation tests
^^^^^^^^^^^^^^^^^^^^^^

Cross-validation tests verify numerical correctness against independent
implementations. They are gated behind the ``--run-slow`` flag::

    pytest tests/ --run-slow

- **Skyfield** -- verifies coordinate transforms against an independent astronomy library
- **scan_patterns** -- verifies scan pattern geometry against an observation/planning implementation
  (requires ``pip install mapping @ git+https://github.com/ccatobs/scan_patterns.git``).
  On Windows or strict-locale environments, you may need to first patch
  ``scan_patterns/setup.py`` to add ``encoding='utf-8'`` to the ``open()`` call
  (Python 3.10+ no longer defaults to UTF-8).
- **KOSMA** -- verifies focal plane offset model against the KOSMA telescope control system
