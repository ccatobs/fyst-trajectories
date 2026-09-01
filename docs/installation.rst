Installation
============

**Requires Python 3.10 or higher.**

From GitHub, pinned to a release tag (``v0.8.0`` is the latest - pre-1.0,
breaking changes arrive only in minor releases, never in patches)::

    pip install "fyst-trajectories @ git+https://github.com/ccatobs/fyst-trajectories.git@v0.8.0"

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
  ``plot_hit_map``, ``plot_timeline_gantt``, ``plot_sky_coverage``,
  ``plot_visibility``, ``plot_observability_windows``, ``plot_sky_view``,
  ``plot_array_footprint``).
- ``performance`` - adds ``numba`` for JIT-compiled hot paths.
- ``ephemeris`` - adds ``jplephem`` for high-precision solar-system body
  positions.
- ``overhead`` - adds ``healpy`` for hit-map accumulation in
  :func:`fyst_trajectories.overhead.accumulate_hitmaps`.
- ``sun-avoidance`` - not a pip extra. The shared
  `ccatobs/sun-avoidance <https://github.com/ccatobs/sun-avoidance>`_
  library backs the ``"cone"`` and ``"cad"`` avoidance models and
  installs from git at a pinned revision; see :doc:`sun_avoidance`. That
  repository is CCAT-internal and needs collaboration access. The
  default ``"scalar"`` model needs nothing beyond fyst-trajectories.
- ``docs`` - adds Sphinx and the rendering extensions used to build
  this site.
- ``dev`` - superset of testing and development tools (pytest,
  pytest-cov, hypothesis, ruff, pylint, pre-commit, skyfield, numba,
  matplotlib, jplephem).
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
- **KOSMA** - verifies the focal plane offset model against the KOSMA telescope control
  system's formulas, reproduced inline in the test (no KOSMA software is installed or run)
- **scan_patterns** - the AltAz-planner parity tests compare against the ``scanning``
  package when it is installed (``pip install -e ../scan_patterns``) and skip silently
  otherwise, so install it for the full oracle set
