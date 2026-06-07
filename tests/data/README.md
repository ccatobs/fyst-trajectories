# Test data

## `titan_excerpt.bsp` — Titan satellite ephemeris (cross-validation fixture)

A small (~101 KB), self-contained excerpt of the NAIF JPL **`sat441`** Saturn-satellite
SPK kernel, used **only** as an offline cross-validation fixture for the Titan-tracking
tests. It is **not** on the runtime path — operational Titan tracking uses a
bring-your-own kernel via the `FYST_SATELLITE_KERNEL` environment variable (or an
explicit kwarg). This file lets the test suite validate Titan positions deterministically
and offline.

### Provenance

- **Source:** `https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/sat441.bsp`
  (NAIF generic kernels; US-government **public domain**, freely redistributable).
- **Cut:** 2026-06-03 with `jplephem` 2.24, by the fyst-trajectories project.
- **Command:**
  ```bash
  python -m jplephem excerpt --targets 3,399,10,6,606 2026/6/1 2026/10/1 \
    https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/sat441.bsp \
    titan_excerpt.bsp
  ```
- **Coverage window:** 2026-06-01 .. 2026-10-01 (UTC).
- **Segments (5):** SSB→Saturn-bary (6); Saturn-bary→Titan (606); SSB→Earth-bary (3);
  Earth-bary→Earth (399); SSB→Sun (10).

### Why the target list is `3,399,10,6,606` (not just `6,606`)

astropy's `get_body([(0,6),(6,606)], ..., ephemeris=...)` computes the **observer**
(Earth) position from the *same* kernel, so the excerpt MUST include Earth-barycenter (3),
Earth (399), and the Sun (10). A `--targets 6,606` excerpt loads in skyfield but makes
astropy raise `KeyError: (0, 3)`. The excerpt is a **lossless** cut of the full kernel.

### Validation

Titan apparent Az/El at FYST from this excerpt matches **JPL Horizons** (airless) to
**0.30–0.55 arcsec** over the window, and Titan sits 1.6–2.8 arcmin from Saturn (as
expected for the moon) — far inside the Prime-Cam beam (~15–59 arcsec).

### Regenerating (forward maintenance)

This is a **binary** fixture with a **fixed coverage window**. When test epochs are moved
forward (e.g. out of the IERS prediction window), the excerpt must be **re-cut** (not
edited) for a window covering the new epochs, and the frozen Horizons reference values in
the Titan tests regenerated to match.
