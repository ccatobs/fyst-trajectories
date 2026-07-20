Installation
============

**Requires Python 3.10 or higher.**

From GitHub::

    pip install "fyst-trajectories @ git+https://github.com/ccatobs/fyst-trajectories.git"

Development install
-------------------

Clone and install in editable mode with development extras::

    git clone https://github.com/ccatobs/fyst-trajectories.git
    cd fyst-trajectories
    pip install -e ".[dev]"

Optional dependencies
---------------------

The minimal install pulls only the core runtime dependencies (astropy,
numpy, pyyaml, scipy). The following extras are available for opt-in features:

- ``plotting`` - adds ``matplotlib``; required by
  :mod:`fyst_trajectories.visualization` (``plot_trajectory``,
  ``plot_hit_map``, ``plot_timeline_gantt``, ``plot_sky_coverage``).
- ``performance`` - adds ``numba`` for JIT-compiled hot paths.
- ``ephemeris`` - adds ``jplephem`` for high-precision solar-system body
  positions.
- ``overhead`` - adds ``healpy`` for hit-map accumulation in
  :func:`fyst_trajectories.overhead.accumulate_hitmaps`.
- ``docs`` - adds Sphinx and the rendering extensions used to build
  this site.
- ``dev`` - superset of testing and development tools (pytest,
  pytest-cov, hypothesis, ruff, pre-commit, skyfield, numba).
- ``all`` - installs every extra above.

Install one or more by passing them to ``pip``::

    pip install -e ".[plotting,overhead]"

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

- **Skyfield** - verifies coordinate transforms against an independent astronomy library
- **KOSMA** - verifies focal plane offset model against the KOSMA telescope control system
