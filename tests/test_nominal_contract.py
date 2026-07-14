"""Planner-level guard that default emission is nominal (vacuum) az/el.

Pins the P-INCM-ICD-0003-A Eq.(1) contract that fyst-trajectories emits
``Az_nominal`` / ``El_nominal`` (astronomically calculated, unrefracted): the
default no-atmosphere planner path must equal the explicit vacuum path, and a
silent default flip to refracted coordinates (``for_fyst()``) must fail here.
Refraction is applied downstream at OCS and/or the ACU, never in the emitted
trajectory. Sibling seams: ``test_coordinates.py`` (constructor refraction),
``test_ac_schema_contract.py`` (A-to-C params).
"""

import numpy as np
from astropy.time import Time

from fyst_trajectories import (
    AtmosphericConditions,
    Coordinates,
    FieldRegion,
    plan_daisy_scan,
    plan_pong_scan,
    plan_source_ces,
)
from fyst_trajectories.patterns import (
    ConstantElScanConfig,
    ConstantElScanPattern,
    DaisyScanConfig,
    DaisyScanPattern,
    PongScanConfig,
    PongScanPattern,
    TrajectoryBuilder,
)

# A single mid-elevation celestial target shared by the refracting families.
# At the FYST site this culminates near el ~43 deg at ``_START``, where the
# ``for_fyst()`` refraction lift is tens of arcsec: comfortably above the
# ~1e-9 deg tolerance used for the vacuum-equality check, so the guard both
# holds and discriminates. ``_START`` stays within the vendored IERS table.
_RA, _DEC = 120.0, -5.0
_START = Time("2026-03-15T04:00:00", scale="utc")
_DURATION = 30.0
_TIMESTEP = 0.2

# source_ces search anchor (a setting Jupiter-free sidereal arc that reaches
# ``el_bore`` within the night).
_SOURCE_NIGHT = Time("2026-03-15T00:00:00", scale="utc")


def _el_refraction_bump_deg(site) -> float:
    """Elevation lift (deg) ``for_fyst()`` adds at ``(_RA, _DEC, _START)``."""
    _, el_vac = Coordinates(site).radec_to_altaz(_RA, _DEC, obstime=_START)
    _, el_ref = Coordinates(site, atmosphere=AtmosphericConditions.for_fyst()).radec_to_altaz(
        _RA, _DEC, obstime=_START
    )
    return abs(float(el_ref) - float(el_vac))


def _assert_nominal_default(default, no_refraction, for_fyst, bump_deg):
    """Assert default emission is vacuum and that the guard discriminates.

    ``default`` (no atmosphere) must equal ``no_refraction`` bit-for-bit, while
    ``for_fyst`` must lift the elevation by the refraction bump and leave azimuth
    untouched (``dAz_ref`` is zero per P-INCM-ICD-0003-A section 2).
    """
    np.testing.assert_allclose(default.az, no_refraction.az, rtol=0, atol=1e-9)
    np.testing.assert_allclose(default.el, no_refraction.el, rtol=0, atol=1e-9)

    # Premise: the chosen target actually refracts, so the discriminator below
    # is meaningful (a near-zenith target would make this pass vacuously).
    assert bump_deg > 10.0 / 3600.0

    median_el_shift = float(np.median(np.abs(for_fyst.el - default.el)))
    assert median_el_shift > 0.5 * bump_deg

    # Refraction is elevation-only; the azimuth track is unchanged.
    np.testing.assert_allclose(for_fyst.az, default.az, rtol=0, atol=1e-9)


def _pong_config() -> PongScanConfig:
    return PongScanConfig(
        timestep=_TIMESTEP,
        width=1.0,
        height=1.0,
        spacing=0.2,
        velocity=0.5,
        num_terms=4,
        angle=0.0,
    )


def test_pong_default_emission_is_nominal_vacuum(site):
    """Pong emits vacuum az/el by default; for_fyst refracts the elevation."""
    pattern = PongScanPattern(ra=_RA, dec=_DEC, config=_pong_config())
    default = pattern.generate(site, duration=_DURATION, start_time=_START)
    no_refraction = pattern.generate(
        site,
        duration=_DURATION,
        start_time=_START,
        atmosphere=AtmosphericConditions.no_refraction(),
    )
    for_fyst = pattern.generate(
        site, duration=_DURATION, start_time=_START, atmosphere=AtmosphericConditions.for_fyst()
    )
    _assert_nominal_default(default, no_refraction, for_fyst, _el_refraction_bump_deg(site))


def test_daisy_default_emission_is_nominal_vacuum(site):
    """Daisy emits vacuum az/el by default; for_fyst refracts the elevation."""
    config = DaisyScanConfig(
        timestep=_TIMESTEP,
        radius=0.4,
        velocity=0.3,
        turn_radius=0.2,
        avoidance_radius=0.0,
        start_acceleration=0.5,
        y_offset=0.0,
    )
    pattern = DaisyScanPattern(ra=_RA, dec=_DEC, config=config)
    default = pattern.generate(site, duration=_DURATION, start_time=_START)
    no_refraction = pattern.generate(
        site,
        duration=_DURATION,
        start_time=_START,
        atmosphere=AtmosphericConditions.no_refraction(),
    )
    for_fyst = pattern.generate(
        site, duration=_DURATION, start_time=_START, atmosphere=AtmosphericConditions.for_fyst()
    )
    _assert_nominal_default(default, no_refraction, for_fyst, _el_refraction_bump_deg(site))


def test_constant_el_emits_altaz_nominal_verbatim(site):
    """ConstantEl (AltAz) emits the user's nominal az/el; atmosphere is a no-op.

    There is no celestial-to-horizon transform here, so refraction has nowhere
    to enter: default, no_refraction, and for_fyst must all emit identical az/el.
    This pins that ConstantEl never grows a hidden refraction seam.
    """
    config = ConstantElScanConfig(
        timestep=_TIMESTEP,
        az_start=120.0,
        az_stop=130.0,
        elevation=45.0,
        az_speed=0.5,
        az_accel=1.0,
    )
    pattern = ConstantElScanPattern(config)
    default = pattern.generate(site, duration=_DURATION)
    for atmosphere in (AtmosphericConditions.no_refraction(), AtmosphericConditions.for_fyst()):
        other = pattern.generate(site, duration=_DURATION, atmosphere=atmosphere)
        np.testing.assert_allclose(default.az, other.az, rtol=0, atol=1e-9)
        np.testing.assert_allclose(default.el, other.el, rtol=0, atol=1e-9)


def test_source_ces_default_emission_is_nominal_vacuum(site):
    """plan_source_ces emits vacuum geometry by default; for_fyst shifts azimuth.

    The boresight elevation is pinned to ``el_bore`` (nominal), so refraction
    cannot enter the emitted el. It instead shifts where the source reaches
    ``el_bore``, moving the solved azimuth track, so a silent default flip to
    for_fyst would not pass unnoticed.
    """
    kwargs = dict(
        ra=180.0,
        dec=-30.0,
        footprint="i1",
        el_bore=40.0,
        night=_SOURCE_NIGHT,
        mode="setting",
        site=site,
        timestep=0.5,
    )
    default = plan_source_ces(**kwargs).trajectory
    no_refraction = plan_source_ces(
        **kwargs, atmosphere=AtmosphericConditions.no_refraction()
    ).trajectory
    for_fyst = plan_source_ces(**kwargs, atmosphere=AtmosphericConditions.for_fyst()).trajectory

    # Default emission equals explicit vacuum, bit-for-bit.
    np.testing.assert_allclose(default.az, no_refraction.az, rtol=0, atol=1e-9)
    np.testing.assert_allclose(default.el, no_refraction.el, rtol=0, atol=1e-9)

    # El is the commanded nominal el_bore verbatim under any atmosphere.
    np.testing.assert_allclose(for_fyst.el, default.el, rtol=0, atol=1e-9)

    # But the atmosphere kwarg is live: for_fyst moves the azimuth track.
    max_az_shift_arcsec = float(np.max(np.abs(for_fyst.az - default.az))) * 3600.0
    assert max_az_shift_arcsec > 1.0


def test_plan_pong_scan_default_emission_is_nominal_vacuum(site):
    """plan_pong_scan's own atmosphere default emits vacuum az/el.

    The public wrapper carries its own ``atmosphere=None`` default and is the
    seam PCS dispatches through, so it is guarded directly: the pattern-seam
    and builder tests would not catch a flip made here.
    """

    def _plan(**atm):
        return plan_pong_scan(
            field=FieldRegion(ra_center=_RA, dec_center=_DEC, width=1.0, height=1.0),
            velocity=0.5,
            spacing=0.2,
            num_terms=4,
            site=site,
            start_time=_START,
            timestep=0.5,
            **atm,
        ).trajectory

    default = _plan()
    no_refraction = _plan(atmosphere=AtmosphericConditions.no_refraction())
    for_fyst = _plan(atmosphere=AtmosphericConditions.for_fyst())
    _assert_nominal_default(default, no_refraction, for_fyst, _el_refraction_bump_deg(site))


def test_plan_daisy_scan_default_emission_is_nominal_vacuum(site):
    """plan_daisy_scan's own atmosphere default emits vacuum az/el (see pong)."""

    def _plan(**atm):
        return plan_daisy_scan(
            ra=_RA,
            dec=_DEC,
            radius=0.4,
            velocity=0.3,
            turn_radius=0.2,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            site=site,
            start_time=_START,
            timestep=_TIMESTEP,
            duration=_DURATION,
            **atm,
        ).trajectory

    default = _plan()
    no_refraction = _plan(atmosphere=AtmosphericConditions.no_refraction())
    for_fyst = _plan(atmosphere=AtmosphericConditions.for_fyst())
    _assert_nominal_default(default, no_refraction, for_fyst, _el_refraction_bump_deg(site))


def test_trajectory_builder_default_emission_is_nominal_vacuum(site):
    """TrajectoryBuilder's default (no with_atmosphere) emits vacuum az/el."""
    config = _pong_config()

    def _build(atmosphere):
        builder = (
            TrajectoryBuilder(site)
            .at(ra=_RA, dec=_DEC)
            .with_config(config)
            .duration(_DURATION)
            .starting_at(_START)
        )
        if atmosphere is not None:
            builder = builder.with_atmosphere(atmosphere)
        return builder.build()

    default = _build(None)
    no_refraction = _build(AtmosphericConditions.no_refraction())
    for_fyst = _build(AtmosphericConditions.for_fyst())
    _assert_nominal_default(default, no_refraction, for_fyst, _el_refraction_bump_deg(site))
