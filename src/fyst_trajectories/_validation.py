"""Argument validators shared across the package.

Each validator raises ``ValueError`` naming the offending field, so a
config or planner rejects a bad number where it is passed instead of letting
it surface later in an unrelated computation.
"""

import math


def _require_positive(value: float, name: str) -> None:
    """Raise unless ``value`` is a finite, strictly positive number.

    ``value <= 0`` alone lets NaN and infinity through: NaN fails every
    comparison, and infinity is genuinely positive. Both then surface far
    from the config that accepted them (as ``cannot convert float NaN to
    integer`` from the leg quantiser, or as an empty sample array), so the
    finiteness test belongs here with the sign test.

    Parameters
    ----------
    value : float
        The field being validated.
    name : str
        Field name used in the message.

    Raises
    ------
    ValueError
        If ``value`` is not finite or is not strictly positive.
    """
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _require_non_negative(value: float, name: str) -> None:
    """Raise unless ``value`` is a finite, non-negative number.

    The non-negative counterpart of :func:`_require_positive`; see that
    function for why finiteness is checked alongside the sign.

    Parameters
    ----------
    value : float
        The field being validated.
    name : str
        Field name used in the message.

    Raises
    ------
    ValueError
        If ``value`` is not finite or is negative.
    """
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _require_finite(value: float, name: str) -> None:
    """Raise unless ``value`` is a finite number.

    For fields whose sign and magnitude are free (a start azimuth, a
    rotation angle, an offset) finiteness is the whole contract; see
    :func:`_require_positive` for why it is checked at construction.

    Parameters
    ----------
    value : float
        The field being validated.
    name : str
        Field name used in the message.

    Raises
    ------
    ValueError
        If ``value`` is not finite.
    """
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value}")
