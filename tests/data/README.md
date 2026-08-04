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

## `de421_excerpt.bsp` — planetary ephemeris (skyfield oracle fixture)

A small excerpt of NAIF JPL **`de421`**, the offline ephemeris for the skyfield
cross-validation oracles in `tests/test_coordinates.py`
(`test_get_body_radec_matches_skyfield`) and
`tests/test_coordinates_cross_validation.py` (`test_solar_system_body_positions`).
It replaces per-fixture `skyfield.api.load("de421.bsp")` network downloads
(~16.8 MB from `ssd.jpl.nasa.gov`). **Not** on the runtime path.

### Provenance
- **Source:** `https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de421.bsp`
  (NAIF generic kernels; US-government **public domain**). de421 is byte-identical
  across the NAIF and `ssd.jpl.nasa.gov` mirrors; either works.
- **Command:**
  ```bash
  python -m jplephem excerpt --targets 3,399,301,4,499,5,6,8,10 2025/12/1 2027/1/1 \
    https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de421.bsp \
    de421_excerpt.bsp
  ```
- **Coverage window:** 2025-12-01 .. 2027-01-01 (UTC) — spans the full 2026 test
  calendar. The two files share this one fixture: `test_coordinates.py`'s oracle uses
  `16:30`/`04:30` on 2026-06-15, but `test_coordinates_cross_validation.py` evaluates the
  ephemeris across many epochs (2026-01-01 .. 2026-12-21, plus the Sirius rise/set scan
  from 2026-03-15), so a narrow June window makes those other slow tests raise
  `EphemerisRangeError`. The window covers all of them with a month of margin each side.
- **Targets:** observer SSB→Earth-bary (3), Earth-bary→Earth (399); Moon (301);
  Sun (10); **Mars-bary (4)** and **Mars (499)** — `eph["mars barycenter"]` in the
  cross-val file resolves to 4, while `eph["mars"]` in the oracle resolves to 499 via
  the chain 0→4→499, so **both** are required; Jupiter/Saturn/Neptune barycenters
  (5/6/8). Omitting 499 makes the oracle's `[mars]` case raise `KeyError` — the same
  missing-segment failure documented for Titan above.

### Regenerating
Binary fixture, fixed window. Re-cut if the oracle/cross-val epochs move forward.

## `finals2000A.all` — Earth-orientation (IERS) table (offline fixture)

A snapshot of astropy's IERS-A Earth-orientation table (~3.75 MB), vendored so the suite
resolves Earth orientation offline and deterministically. **Not** on the runtime path.

### Why it is needed
The hard-coded 2026 test epochs sit past the last *measured* IERS row, so a stock astropy
does one of two things on the first coordinate transform, both of which broke CI:
- with network or a warm cache, it fetches ~3.75 MB from `datacenter.iers.org`, the stall
  that timed out the slow CI job, or
- on a cold runner with no cached `finals2000A.all`, it falls back to the shorter table
  bundled with `astropy-iers-data`, which does not cover 2026, and raises `IERSRangeError`
  on every 2026-epoch transform.

`tests/conftest.py` pins this vendored table at import time via
`iers.earth_orientation_table.set(iers.IERS_A.open(...))` and sets
`iers.conf.auto_download = False`. Because the table is set explicitly, local and CI runs
use byte-identical Earth-orientation data at full accuracy, and a local run exercises the
same offline path as CI.

### Provenance
- **Source:** astropy's standard IERS-A product `finals2000A.all`
  (`https://datacenter.iers.org/data/9/finals2000A.all`, mirrored at
  `https://maia.usno.navy.mil/ser7/finals2000A.all`).
- **Snapshot:** 2026-06-26, copied from the local astropy download cache.
- **Coverage:** measured values through mid-2026 plus IERS predictions to ~2027, which
  spans the 2026 test calendar with margin.

### Regenerating (forward maintenance)
Re-cut when the hard-coded test epochs approach the prediction range (astropy will start
warning). Refresh from a current astropy cache or either URL above:
```bash
python -c "from astropy.utils import iers; from astropy.utils.data import download_file; \
import shutil; shutil.copyfile(download_file(iers.conf.iers_auto_url, cache=True), \
'tests/data/finals2000A.all')"
```
This is the same forward-maintenance trigger as the de421/Titan excerpt windows.

## `sun_avoidance_parity_e6fa12a.npz` — sun-model parity record (record-replay fixture)

The record half of the sun-model record-replay harness (~17 KB).
Verdicts and geometric thresholds for every `fyst_trajectories.sun_models` configuration
(`scalar`, `cone45`, `cone50`, `cad`, `cad_msa0`, `cad_island`) over a deterministic
72-azimuth x 29-elevation x 8-epoch grid (includes negative encoder wraps and exactly
el=90), plus the loaded CAD table and the SHA pins.
**Not** on the runtime path.

### How it is used
- `tests/test_sun_models_fixture.py` (always, offline): fixture integrity, the CAD table's
  properties, regeneration of the no-library `scalar` rows, cross-model containment.
- `tests/test_sun_models_live.py` (skipped without the `sun_avoidance` library): the drift
  detector — regenerates every row from the installed library through the adapter and
  requires bit-equality.

### Provenance
- **Generated against:** `ccatobs/sun-avoidance` @ `e6fa12a` (the filename carries the
  pin; `fyst_trajectories.sun_models.SUN_AVOIDANCE_PINNED_SHA` / `CAD_TABLE_SHA256` must
  match the recorded values).
- **Scalar rows:** pinned 45/50 deg radii (explicit, NOT the site defaults), so a
  site-default policy bump never forces a re-cut.
- **IERS:** recorded under the vendored `finals2000A.all` pin above; all 8 epochs sit
  inside its range.

### Regenerating (deliberate re-pin only)
Update the two SHA constants in `sun_models.py`, then:
```bash
python tests/test_sun_models_live.py
```
(the `__main__` block applies the same IERS pin as `conftest.py` and rewrites this file).
Never regenerate to make a red drift test pass without first deciding the new library
revision is the one to pin.
