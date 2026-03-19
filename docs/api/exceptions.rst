Exceptions and Warnings
=======================

Custom exception hierarchy for graceful error handling.

.. automodule:: fyst_trajectories.exceptions
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

**Catch an unobservable target**::

    from fyst_trajectories import plan_pong_scan, FieldRegion, get_fyst_site
    from fyst_trajectories.exceptions import TargetNotObservableError
    from astropy.time import Time

    try:
        block = plan_pong_scan(
            field=FieldRegion(ra_center=180.0, dec_center=80.0, width=2.0, height=2.0),
            velocity=0.5,
            site=get_fyst_site(),
            start_time=Time("2024-06-15T12:00:00", scale="utc"),
        )
    except TargetNotObservableError as e:
        print(f"Target not observable: {e}")
        if e.bounds_error:
            print(f"  Axis: {e.bounds_error.axis}")
            print(f"  Actual: {e.bounds_error.actual_min:.1f} to {e.bounds_error.actual_max:.1f}")

**Catch pointing warnings**::

    import warnings
    from fyst_trajectories.exceptions import PointingWarning

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        block = plan_constant_el_scan(...)
        sun_warnings = [x for x in w if issubclass(x.category, PointingWarning)]
