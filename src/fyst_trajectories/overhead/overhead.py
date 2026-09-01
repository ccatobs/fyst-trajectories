"""Calibration state tracking and overhead injection.

Manages when calibrations are due based on cadence policies.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

from astropy.time import Time

from .models import CalibrationPolicy, CalibrationSpec, CalibrationType, OverheadModel

if TYPE_CHECKING:
    from ..coordinates import Coordinates

__all__ = [
    "CalibrationState",
]


@dataclass(frozen=True)
class CalibrationState:
    """Tracks when each calibration type was last performed.

    Used by the scheduler to determine which calibrations are due
    at a given time based on the calibration policy cadences.

    Parameters
    ----------
    last_retune : Time or None
        Last KID probe tone reset time.
    last_pointing_cal : Time or None
        Last pointing correction time.
    last_focus : Time or None
        Last focus check time.
    last_skydip : Time or None
        Last sky dip time.
    last_planet_cal : Time or None
        Last planet calibration time.
    last_beam_map : Time or None
        Last beam-map scan time. Only consulted when
        ``CalibrationPolicy.beam_map_cadence`` is set; otherwise beam
        maps must be injected manually.
    """

    last_retune: Time | None = None
    last_pointing_cal: Time | None = None
    last_focus: Time | None = None
    last_skydip: Time | None = None
    last_planet_cal: Time | None = None
    last_beam_map: Time | None = None

    def needs_calibration(
        self,
        current_time: Time,
        policy: CalibrationPolicy,
        overhead: OverheadModel,
        coords: Coordinates | None = None,
    ) -> list[CalibrationSpec]:
        """Determine which calibrations are due.

        Checks each calibration type against its cadence. A cadence of 0
        keeps that calibration permanently due; see
        :class:`CalibrationPolicy` for how the scheduler consumes a
        cadence-0 retune. Beam mapping has
        a nullable cadence; ``policy.beam_map_cadence is None`` (the
        default) keeps beam maps off the automatic schedule entirely.

        Returns calibrations in priority order: retune, pointing, focus,
        skydip, planet_cal, beam_map.

        Parameters
        ----------
        current_time : Time
            Current scheduling time.
        policy : CalibrationPolicy
            Cadence configuration.
        overhead : OverheadModel
            Duration information for calibration operations.
        coords : Coordinates or None
            Coordinate transformer. When provided, planet_cal and
            beam_map are only included if at least one planet target
            is above the horizon.

        Returns
        -------
        list of CalibrationSpec
            Calibrations that should be performed now.
        """
        needed = []

        checks: list[tuple[CalibrationType, Time | None, float | None]] = [
            (CalibrationType.RETUNE, self.last_retune, policy.retune_cadence),
            (CalibrationType.POINTING_CAL, self.last_pointing_cal, policy.pointing_cadence),
            (CalibrationType.FOCUS, self.last_focus, policy.focus_cadence),
            (CalibrationType.SKYDIP, self.last_skydip, policy.skydip_cadence),
            (CalibrationType.PLANET_CAL, self.last_planet_cal, policy.planet_cal_cadence),
            (CalibrationType.BEAM_MAP, self.last_beam_map, policy.beam_map_cadence),
        ]

        for cal_type, last_time, cadence in checks:
            # ``None`` cadence means "never automatically schedule"; it keeps
            # BEAM_MAP off the automatic schedule unless the operator opts in.
            if cadence is None:
                continue
            if self._is_due(current_time, last_time, cadence):
                target = None
                if cal_type in (CalibrationType.PLANET_CAL, CalibrationType.BEAM_MAP) and (
                    policy.planet_targets
                ):
                    target = self._find_visible_planet(
                        policy.planet_targets,
                        current_time,
                        coords,
                        min_elevation=policy.planet_min_elevation,
                    )
                    if target is None:
                        continue
                needed.append(
                    CalibrationSpec(
                        name=cal_type,
                        duration=overhead.get_calibration_duration(cal_type),
                        target=target,
                    )
                )

        return needed

    @staticmethod
    def _find_visible_planet(
        planet_targets: tuple[str, ...],
        current_time: Time,
        coords: Coordinates | None,
        min_elevation: float = 20.0,
    ) -> str | None:
        """Return the first visible planet target.

        Parameters
        ----------
        planet_targets : tuple of str
            Planet names to check.
        current_time : Time
            Current time.
        coords : Coordinates or None
            Coordinate transformer. If ``None`` the visibility check is
            skipped entirely and the first entry of ``planet_targets``
            is returned (caller takes responsibility for visibility).
            When a ``Coordinates`` instance is provided the function
            returns the first target above ``min_elevation``, or
            ``None`` if none are visible.
        min_elevation : float
            Minimum altitude in degrees for a planet to be considered
            visible. Default is 20.0 degrees.

        Returns
        -------
        str or None
            Name of a visible planet, or ``None`` when ``coords`` is
            supplied and no planet is above the minimum elevation.
            When ``coords`` is ``None`` and ``planet_targets`` is empty
            this returns ``None`` (no "first target" to fall back on).
        """
        if coords is None:
            return planet_targets[0] if planet_targets else None

        for planet in planet_targets:
            _, alt = coords.get_body_altaz(planet, current_time)
            if alt > min_elevation:
                return planet

        return None

    def update(self, cal_type: CalibrationType | str, time: Time) -> CalibrationState:
        """Return a new state with the given calibration type updated.

        Looks up :attr:`CalibrationType.state_field` to find the
        matching ``last_*`` attribute on this dataclass; both this
        method and :meth:`OverheadModel.get_calibration_duration` share
        a single mapping table so the two APIs cannot drift out of sync.

        Parameters
        ----------
        cal_type : CalibrationType or str
            Calibration type name.
        time : Time
            Time the calibration was performed.

        Returns
        -------
        CalibrationState
            New state with the updated timestamp.
        """
        cal_type = CalibrationType.coerce(cal_type)
        return dataclasses.replace(self, **{cal_type.state_field: time})

    @staticmethod
    def _is_due(current_time: Time, last_time: Time | None, cadence: float) -> bool:
        """Check whether a calibration is due.

        Parameters
        ----------
        current_time : Time
            Current time.
        last_time : Time or None
            Last time this calibration was performed (None = never).
        cadence : float
            Required interval in seconds. 0 means always due.

        Returns
        -------
        bool
            True if the calibration should be performed.
        """
        if last_time is None:
            return True
        if cadence == 0:
            return True
        elapsed = (current_time - last_time).sec
        return elapsed >= cadence
