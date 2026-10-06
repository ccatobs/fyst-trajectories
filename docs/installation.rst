Installation
============

**Requires Python 3.10 or higher.**

From GitHub, pinned to a release tag (``v0.10.0`` is the latest; see the
:doc:`changelog` for versions - pre-1.0, breaking changes arrive only in
minor releases, never in patches):

.. code-block:: bash

    pip install "fyst-trajectories @ git+https://github.com/ccatobs/fyst-trajectories.git@v0.10.0"

Development install
-------------------

Clone and install in editable mode with development extras:

.. code-block:: bash

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
  ``plot_source_track``, ``plot_array_footprint``).
- ``performance`` - adds ``numba``, which JIT-compiles the daisy pattern's
  integration loop; everything runs without it, more slowly.
- ``ephemeris`` - adds ``jplephem``, needed to load the JPL satellite SPK
  kernel that the ``SATELLITE_BODIES`` targets (for example Titan) require.
  The bodies in ``SOLAR_SYSTEM_BODIES`` use astropy's built-in ephemeris and
  need no extra.
- ``overhead`` - adds ``healpy`` for hit-map accumulation in
  :func:`fyst_trajectories.overhead.accumulate_hitmaps`.
- ``sun-avoidance`` - not a pip extra. The shared ``ccatobs/sun-avoidance``
  library backs the ``"cone"`` and ``"cad"`` avoidance models and installs
  from git; :doc:`sun_avoidance` has the pinned command and the access
  note. The default ``"scalar"`` model needs nothing beyond
  fyst-trajectories.
- ``docs`` - adds Sphinx and the rendering extensions used to build
  this site.
- ``dev`` - the testing and development tools (pytest, pytest-cov,
  hypothesis, ruff, pylint, pyright, pre-commit, skyfield), plus what the
  ``plotting``, ``performance`` and ``ephemeris`` extras install; it does
  not include ``overhead`` or ``docs``.
- ``all`` - installs every pip extra above (the sun-avoidance library is
  not one; ``overhead`` only off Windows, where ``healpy`` has no build).

Install one or more by passing them to ``pip``:

.. code-block:: bash

    pip install -e ".[plotting,overhead]"

Running tests
-------------

Fast tests:

.. code-block:: bash

    pytest tests/

The suite splits along the two tiers of :doc:`index` under the ``offline``
marker. ``pytest tests/`` runs both; run one at a time with:

.. code-block:: bash

    pytest -m "not offline" tests/   # library tier: what a control system imports
    pytest -m offline tests/         # simulator tier

Linting:

.. code-block:: bash

    ruff check . && ruff format --check .

Cross-validation tests
~~~~~~~~~~~~~~~~~~~~~~

Cross-validation tests verify numerical correctness against independent
implementations. They are gated behind the ``--run-slow`` flag:

.. code-block:: bash

    pytest tests/ --run-slow

- **Skyfield** - verifies coordinate transforms against an independent astronomy library
- **KOSMA** - verifies the focal plane offset model against the KOSMA telescope control
  system's formulas
- **scan_patterns** - the AltAz-planner parity tests and the check of the
  focal-plane projection's absolute sign compare against the ``scanning``
  package when it is installed and are skipped otherwise, so install it for
  the full oracle set:

  .. code-block:: bash

      pip install pandas fast-histogram
      pip install --no-deps "git+https://github.com/ccatobs/scan_patterns.git@refactor/fyst-trajectories"

  ``--no-deps`` is required: that branch pins ``fyst-trajectories`` by URL in
  its own requirements, so without it pip would replace the checkout under
  test. ``pandas`` and ``fast-histogram`` are the runtime imports ``--no-deps``
  skips.
