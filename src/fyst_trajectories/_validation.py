"""Argument validators shared across the package.

Each validator raises ``ValueError`` naming the offending field, so a
config or planner rejects a bad number where it is passed instead of letting
it surface later in an unrelated computation. Each ``_require_*`` validator
returns the value it checked as a float, so a caller that rebinds the value
computes with the number it checked: a ``Decimal`` passes the checks, and
used as given it fails in its first float arithmetic, after the work before
it has run.
"""

import math

#: The largest ``az_padding`` the planners accept, in degrees: one full turn.
AZ_PADDING_MAX_DEG = 360.0

# The NumPy dtype kinds of a real number: signed integer, unsigned integer, float.
_REAL_DTYPE_KINDS = ("i", "u", "f")


def _is_finite(value: float, name: str) -> bool:
    """Return whether ``value`` is finite, refusing a value that is not a real number.

    A real number here is an ``int``, ``float``, ``Decimal``, ``Fraction``,
    NumPy integer or float scalar, or a 0-d NumPy array of one. A ``bool``
    is not one, though ``float`` reads it as 0 or 1, and neither is text,
    though ``float`` parses it.

    Parameters
    ----------
    value : float
        The field being validated.
    name : str
        Field name used in the message.

    Returns
    -------
    bool
        Whether ``value`` is finite.

    Raises
    ------
    ValueError
        If ``value`` is not a real number: a ``bool`` (Python or NumPy), an
        array of one or more dimensions, a complex number, text, ``None`` or
        an integer past the float range.
    """
    # float() would read a bool as 0 or 1, a one-element array as its element
    # and a NumPy complex as its real part, so those are refused before it is
    # asked, and a Python complex with them, so that every complex is refused
    # alike. The NumPy attributes are read by name, so no NumPy import is needed.
    kind = getattr(getattr(value, "dtype", None), "kind", "f")
    if (
        isinstance(value, (bool, complex))
        or kind not in _REAL_DTYPE_KINDS
        or getattr(value, "ndim", 0) != 0
    ):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    try:
        # math.isfinite takes exactly the real numbers float() converts by
        # value, refuses text, which float() would parse, and raises what
        # float() would for a number it cannot convert.
        return math.isfinite(value)
    except OverflowError:
        raise ValueError(
            f"{name} must be a real number within the float range, got {value!r}"
        ) from None
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number, got {value!r}") from None


def _require_positive(value: float, name: str) -> float:
    """Return ``value`` as a float, raising unless it is a finite, strictly positive number.

    ``value <= 0`` alone lets NaN and infinity through: NaN fails every
    comparison, and infinity is genuinely positive. Both then surface far
    from the config that accepted them (as ``cannot convert float NaN to
    integer`` from the leg quantiser, or as an empty sample array), so the
    finiteness test belongs here with the sign test.

    Parameters
    ----------
    value : float
        The field being validated: a real number as :func:`_is_finite`
        defines it.
    name : str
        Field name used in the message.

    Returns
    -------
    float
        ``value`` converted to a float.

    Raises
    ------
    ValueError
        If ``value`` is not a real number (a ``bool``, an array of one or more
        dimensions, a complex number, text, ``None``, an integer past the
        float range), is not finite or is not strictly positive.
    """
    finite = _is_finite(value, name)
    number = float(value)
    if not finite or number <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return number


def _require_non_negative(value: float, name: str, *, maximum: float | None = None) -> float:
    """Return ``value`` as a float, raising unless it is finite and non-negative.

    The non-negative counterpart of :func:`_require_positive`; see that
    function for why finiteness is checked alongside the sign.

    Parameters
    ----------
    value : float
        The field being validated: a real number as :func:`_is_finite`
        defines it.
    name : str
        Field name used in the message.
    maximum : float, optional
        Largest accepted value, inclusive. ``None`` (default) sets no bound.

    Returns
    -------
    float
        ``value`` converted to a float.

    Raises
    ------
    ValueError
        If ``value`` is not a real number (a ``bool``, an array of one or more
        dimensions, a complex number, text, ``None``, an integer past the
        float range), is not finite, is negative, or exceeds ``maximum``.
    """
    finite = _is_finite(value, name)
    number = float(value)
    if not finite or number < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be at most {maximum}, got {value}")
    return number


def _require_finite(value: float, name: str) -> float:
    """Return ``value`` as a float, raising unless it is a finite number.

    For fields whose sign and magnitude are free (a start azimuth, a
    rotation angle, an offset) finiteness is the whole contract; see
    :func:`_require_positive` for why it is checked at construction.

    Parameters
    ----------
    value : float
        The field being validated: a real number as :func:`_is_finite`
        defines it.
    name : str
        Field name used in the message.

    Returns
    -------
    float
        ``value`` converted to a float.

    Raises
    ------
    ValueError
        If ``value`` is not a real number (a ``bool``, an array of one or more
        dimensions, a complex number, text, ``None``, an integer past the
        float range) or is not finite.
    """
    if not _is_finite(value, name):
        raise ValueError(f"{name} must be a finite number, got {value}")
    return float(value)
