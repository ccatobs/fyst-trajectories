"""Tests for the focal-plane offset projection and its inverse.

``boresight_to_detector``, ``detector_to_boresight`` and ``sky_to_focal_plane``:
round trips, known geometry, field-rotation effects, the inverse's thresholds,
zenith degeneracy and refusal of non-finite input, and the cross-package parity
check of the absolute
field-rotation sign against the scan_patterns oracle.
"""

import warnings

import numpy as np
import pytest

from fyst_trajectories.exceptions import (
    OffsetInversionError,
    PointingError,
)
from fyst_trajectories.offsets import (
    _INVERSE_EARLY_EXIT_THRESHOLD,
    _INVERSE_FAILURE_THRESHOLD,
    InstrumentOffset,
    apply_detector_offset,
    boresight_to_detector,
    detector_to_boresight,
    sky_to_focal_plane,
)
from fyst_trajectories.trajectory import Trajectory


class TestBoresightToDetector:
    """Forward projection: offset directions, field-rotation angles, and array input."""

    def test_zero_offset_no_change(self):
        offset = InstrumentOffset(dx=0.0, dy=0.0)
        det_az, det_el = boresight_to_detector(180.0, 45.0, offset, focal_plane_rotation=0.0)
        assert det_az == pytest.approx(180.0, abs=1e-12)
        assert det_el == pytest.approx(45.0, abs=1e-12)

    def test_x_offset_increases_azimuth(self):
        offset = InstrumentOffset(dx=60.0, dy=0.0)  # 1 degree in arcmin
        det_az, _det_el = boresight_to_detector(180.0, 45.0, offset, focal_plane_rotation=0.0)

        assert det_az > 180.0

    def test_field_rotation_90_degrees(self):
        """A 90 degree field rotation swaps x and y."""
        offset = InstrumentOffset(dx=60.0, dy=0.0)  # 1 degree x offset

        # With 90 degree rotation, x offset becomes y offset
        det_az_90, det_el_90 = boresight_to_detector(180.0, 45.0, offset, focal_plane_rotation=90.0)

        # x offset rotated by 90 deg -> pure elevation offset
        assert det_az_90 == pytest.approx(180.0, abs=1e-6)
        assert det_el_90 == pytest.approx(46.0, rel=1e-4)

    def test_field_rotation_180_degrees(self):
        """A 180 degree field rotation approximately inverts the offset.

        On the sphere, the inversion is not exact because great-circle
        offsets are nonlinear. Both azimuth and elevation components
        invert approximately, with small residuals due to the curvature.
        """
        offset = InstrumentOffset(dx=60.0, dy=30.0)

        det_az_0, det_el_0 = boresight_to_detector(180.0, 45.0, offset, focal_plane_rotation=0.0)
        det_az_180, det_el_180 = boresight_to_detector(
            180.0, 45.0, offset, focal_plane_rotation=180.0
        )

        el_diff_0 = det_el_0 - 45.0
        el_diff_180 = det_el_180 - 45.0
        az_diff_0 = det_az_0 - 180.0
        az_diff_180 = det_az_180 - 180.0

        # Both components approximately invert: azimuth within 2%, elevation within 4%.
        assert az_diff_180 == pytest.approx(-az_diff_0, rel=0.02)
        assert el_diff_180 == pytest.approx(-el_diff_0, rel=0.04)

    def test_array_input(self):
        offset = InstrumentOffset(dx=30.0, dy=30.0)
        az = np.array([100.0, 150.0, 200.0])
        el = np.array([30.0, 45.0, 60.0])

        det_az, det_el = boresight_to_detector(az, el, offset, focal_plane_rotation=0.0)

        # Each element equals the scalar call at the same pose.
        for i in range(3):
            az_i, el_i = boresight_to_detector(az[i], el[i], offset, focal_plane_rotation=0.0)
            assert det_az[i] == pytest.approx(az_i, abs=1e-12)
            assert det_el[i] == pytest.approx(el_i, abs=1e-12)

    def test_array_field_rotation(self):
        offset = InstrumentOffset(dx=60.0, dy=0.0)
        field_rotation = np.array([0.0, 90.0, 180.0])

        det_az, det_el = boresight_to_detector(
            180.0, 45.0, offset, focal_plane_rotation=field_rotation
        )

        # Each element equals the scalar call at its own rotation.
        for i, rotation in enumerate(field_rotation):
            az_i, el_i = boresight_to_detector(180.0, 45.0, offset, focal_plane_rotation=rotation)
            assert det_az[i] == pytest.approx(az_i, abs=1e-12)
            assert det_el[i] == pytest.approx(el_i, abs=1e-12)


class TestDetectorToBoresight:
    """The inverse at zero offset, at a 2 x 1 deg offset, and on arrays, to 0.01 arcsec."""

    def test_zero_offset_no_change(self):
        offset = InstrumentOffset(dx=0.0, dy=0.0)
        bore_az, bore_el = detector_to_boresight(180.0, 45.0, offset, focal_plane_rotation=0.0)
        assert bore_az == pytest.approx(180.0, abs=1e-12)
        assert bore_el == pytest.approx(45.0, abs=1e-12)

    def test_inverse_with_large_offset(self):
        offset = InstrumentOffset(dx=120.0, dy=60.0)  # 2 deg, 1 deg
        bore_az, bore_el = 200.0, 50.0

        det_az, det_el = boresight_to_detector(bore_az, bore_el, offset, focal_plane_rotation=0.0)
        bore_az_recovered, bore_el_recovered = detector_to_boresight(
            det_az, det_el, offset, focal_plane_rotation=0.0
        )

        assert bore_az_recovered == pytest.approx(bore_az, abs=0.01 / 3600.0)
        assert bore_el_recovered == pytest.approx(bore_el, abs=0.01 / 3600.0)

    def test_array_input_inverse(self):
        offset = InstrumentOffset(dx=30.0, dy=30.0)
        bore_az = np.array([100.0, 150.0, 200.0])
        bore_el = np.array([30.0, 45.0, 60.0])

        det_az, det_el = boresight_to_detector(bore_az, bore_el, offset, focal_plane_rotation=0.0)
        bore_az_recovered, bore_el_recovered = detector_to_boresight(
            det_az, det_el, offset, focal_plane_rotation=0.0
        )

        np.testing.assert_allclose(bore_az_recovered, bore_az, atol=0.01 / 3600.0)
        np.testing.assert_allclose(bore_el_recovered, bore_el, atol=0.01 / 3600.0)


class TestOffsetRoundTrips:
    """Forward then inverse returns the boresight to 0.01 arcsec over the whole grid."""

    @pytest.mark.parametrize(
        "dx,dy",
        [
            (0.0, 0.0),  # Zero offset
            (30.0, 0.0),  # X only
            (0.0, 30.0),  # Y only
            (30.0, 30.0),  # Both positive
            (-30.0, 30.0),  # Mixed signs
            (30.0, -30.0),  # Mixed signs
            (-30.0, -30.0),  # Both negative
            (60.0, 60.0),  # 1 degree offset
            (120.0, 60.0),  # Large offset
        ],
    )
    def test_round_trip_various_offsets(self, dx, dy):
        offset = InstrumentOffset(dx=dx, dy=dy)
        bore_az, bore_el = 180.0, 45.0

        det_az, det_el = boresight_to_detector(bore_az, bore_el, offset, focal_plane_rotation=0.0)

        bore_az_back, bore_el_back = detector_to_boresight(
            det_az, det_el, offset, focal_plane_rotation=0.0
        )

        assert bore_az_back == pytest.approx(bore_az, abs=0.01 / 3600.0)
        assert bore_el_back == pytest.approx(bore_el, abs=0.01 / 3600.0)

    @pytest.mark.parametrize(
        "az,el",
        [
            (0.0, 30.0),  # North
            (90.0, 30.0),  # East
            (180.0, 30.0),  # South
            (270.0, 30.0),  # West
            (180.0, 20.0),  # Low elevation
            (180.0, 60.0),  # High elevation
            (180.0, 85.0),  # Near zenith
            (45.0, 45.0),  # Intermediate
            (315.0, 50.0),  # Another quadrant
        ],
    )
    def test_round_trip_various_positions(self, az, el):
        offset = InstrumentOffset(dx=30.0, dy=20.0)

        det_az, det_el = boresight_to_detector(az, el, offset, focal_plane_rotation=0.0)

        az_back, el_back = detector_to_boresight(det_az, det_el, offset, focal_plane_rotation=0.0)

        assert az_back == pytest.approx(az, abs=0.01 / 3600.0)
        assert el_back == pytest.approx(el, abs=0.01 / 3600.0)

    @pytest.mark.parametrize(
        "field_rotation",
        [0.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0, 270.0, -45.0, -90.0],
    )
    def test_round_trip_various_field_rotations(self, field_rotation):
        offset = InstrumentOffset(dx=30.0, dy=20.0)
        bore_az, bore_el = 180.0, 45.0

        det_az, det_el = boresight_to_detector(
            bore_az,
            bore_el,
            offset,
            focal_plane_rotation=field_rotation,
        )

        bore_az_back, bore_el_back = detector_to_boresight(
            det_az,
            det_el,
            offset,
            focal_plane_rotation=field_rotation,
        )

        assert bore_az_back == pytest.approx(bore_az, abs=0.01 / 3600.0)
        assert bore_el_back == pytest.approx(bore_el, abs=0.01 / 3600.0)

    def test_round_trip_with_arrays(self):
        offset = InstrumentOffset(dx=30.0, dy=20.0)

        bore_az = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
        bore_el = np.array([25.0, 35.0, 45.0, 55.0, 65.0])
        field_rotation = np.array([0.0, 30.0, 60.0, 90.0, 120.0])

        det_az, det_el = boresight_to_detector(
            bore_az,
            bore_el,
            offset,
            focal_plane_rotation=field_rotation,
        )

        bore_az_back, bore_el_back = detector_to_boresight(
            det_az,
            det_el,
            offset,
            focal_plane_rotation=field_rotation,
        )

        np.testing.assert_allclose(bore_az_back, bore_az, atol=0.01 / 3600.0)
        np.testing.assert_allclose(bore_el_back, bore_el, atol=0.01 / 3600.0)

    @pytest.mark.parametrize(
        "offset_arcmin,el,field_rotation",
        [
            (6.0, 30.0, 0.0),  # Small offset, low el
            (60.0, 45.0, 45.0),  # 1 deg offset, mid el
            (106.8, 45.0, 90.0),  # PrimeCam inner ring
            (180.0, 60.0, 120.0),  # 3 deg offset
            (300.0, 45.0, 0.0),  # 5 deg offset
            (300.0, 80.0, 60.0),  # 5 deg offset, high el
            (60.0, 20.0, 270.0),  # 1 deg offset, low el
        ],
    )
    def test_round_trip_large_offsets(self, offset_arcmin, el, field_rotation):
        offset = InstrumentOffset(dx=offset_arcmin, dy=offset_arcmin * 0.5)
        bore_az = 200.0

        det_az, det_el = boresight_to_detector(
            bore_az,
            el,
            offset,
            focal_plane_rotation=field_rotation,
        )
        bore_az_back, bore_el_back = detector_to_boresight(
            det_az,
            det_el,
            offset,
            focal_plane_rotation=field_rotation,
        )

        # Round-trip should be accurate to < 0.01 arcsec for all cases
        assert bore_az_back == pytest.approx(bore_az, abs=0.01 / 3600.0)
        assert bore_el_back == pytest.approx(el, abs=0.01 / 3600.0)


class TestOffsetKnownGeometry:
    """Hand-checkable cases: 90 deg swaps the axes, 180 deg inverts, pure el is exact."""

    def test_90_degree_rotation_swaps_axes(self):
        offset = InstrumentOffset(dx=60.0, dy=0.0)
        az, el = 180.0, 0.0  # At horizon, cos(el)=1

        det_az, det_el = boresight_to_detector(az, el, offset, focal_plane_rotation=90.0)

        assert det_az == pytest.approx(az, abs=1e-10)
        assert det_el == pytest.approx(el + 1.0, rel=1e-6)

    def test_90_degree_rotation_with_y_offset(self):
        """A 90 degree rotation turns a y offset into a negative x offset."""
        offset = InstrumentOffset(dx=0.0, dy=60.0)
        az, el = 180.0, 0.0

        det_az, det_el = boresight_to_detector(az, el, offset, focal_plane_rotation=90.0)

        assert det_az == pytest.approx(az - 1.0, rel=1e-6)
        assert det_el == pytest.approx(el, abs=1e-6)

    def test_180_degree_rotation_inverts_offsets(self):
        offset = InstrumentOffset(dx=60.0, dy=30.0)
        az, el = 180.0, 0.0

        det_az_0, det_el_0 = boresight_to_detector(az, el, offset, focal_plane_rotation=0.0)
        det_az_180, det_el_180 = boresight_to_detector(az, el, offset, focal_plane_rotation=180.0)

        az_offset_0 = det_az_0 - az
        az_offset_180 = det_az_180 - az
        el_offset_0 = det_el_0 - el
        el_offset_180 = det_el_180 - el

        assert az_offset_180 == pytest.approx(-az_offset_0, rel=1e-4)
        assert el_offset_180 == pytest.approx(-el_offset_0, rel=1e-4)

    def test_pure_elevation_offset_is_exact(self):
        """A pure elevation offset adds directly to elevation, with no azimuth term."""
        offset = InstrumentOffset(dx=0.0, dy=60.0)
        az, el = 180.0, 45.0

        det_az, det_el = boresight_to_detector(az, el, offset, focal_plane_rotation=0.0)

        assert det_az == pytest.approx(az, abs=1e-10)
        assert det_el == pytest.approx(el + 1.0, abs=1e-10)

    def test_offset_direction_with_zero_field_rotation(self):
        offset = InstrumentOffset(dx=30.0, dy=0.0)
        az, el = 180.0, 0.0

        det_az, _det_el = boresight_to_detector(az, el, offset, focal_plane_rotation=0.0)

        assert det_az > az

    def test_offset_direction_with_zero_field_rotation_y(self):
        offset = InstrumentOffset(dx=0.0, dy=30.0)
        az, el = 180.0, 45.0

        _det_az, det_el = boresight_to_detector(az, el, offset, focal_plane_rotation=0.0)

        assert det_el > el


class TestFieldRotationEffects:
    """The offset sweeps a constant-separation circle, with a 360 deg period."""

    def test_offset_rotates_continuously(self):
        offset = InstrumentOffset(dx=60.0, dy=0.0)
        az, el = 180.0, 45.0

        results = []
        for fr in np.linspace(0, 360, 13)[:-1]:
            det_az, det_el = boresight_to_detector(az, el, offset, focal_plane_rotation=fr)
            results.append((fr, det_az - az, det_el - el))

        # Verify the offset magnitude is approximately constant
        # (on the sphere, it won't be exactly constant in projected coords,
        # but the angular separation should be constant)
        magnitudes = []
        for _, daz, de in results:
            # Approximate angular distance
            cos_el = np.cos(np.deg2rad(el))
            mag = np.sqrt((daz * cos_el) ** 2 + de**2)
            magnitudes.append(mag)

        np.testing.assert_allclose(magnitudes, magnitudes[0], rtol=5e-3)

    def test_field_rotation_period_360(self):
        offset = InstrumentOffset(dx=30.0, dy=20.0)
        az, el = 180.0, 45.0

        det_az_0, det_el_0 = boresight_to_detector(az, el, offset, focal_plane_rotation=0.0)
        det_az_360, det_el_360 = boresight_to_detector(az, el, offset, focal_plane_rotation=360.0)

        assert det_az_360 == pytest.approx(det_az_0, rel=1e-10)
        assert det_el_360 == pytest.approx(det_el_0, rel=1e-10)

    def test_negative_field_rotation(self):
        offset = InstrumentOffset(dx=30.0, dy=20.0)
        az, el = 180.0, 45.0

        det_az_neg, det_el_neg = boresight_to_detector(az, el, offset, focal_plane_rotation=-45.0)
        det_az_pos, det_el_pos = boresight_to_detector(az, el, offset, focal_plane_rotation=315.0)

        assert det_az_pos == pytest.approx(det_az_neg, rel=1e-10)
        assert det_el_pos == pytest.approx(det_el_neg, rel=1e-10)

    def test_vectorized_round_trip(self):
        offset = InstrumentOffset(dx=60.0, dy=30.0)
        az = np.array([100.0, 150.0, 200.0, 250.0])
        el = np.array([25.0, 35.0, 45.0, 55.0])
        fr = np.array([0.0, 30.0, 60.0, 90.0])

        det_az, det_el = boresight_to_detector(az, el, offset, focal_plane_rotation=fr)

        assert isinstance(det_az, np.ndarray)
        assert isinstance(det_el, np.ndarray)
        assert len(det_az) == 4
        assert len(det_el) == 4
        assert np.all(np.isfinite(det_az))
        assert np.all(np.isfinite(det_el))

        # Round-trip
        bore_az, bore_el = detector_to_boresight(det_az, det_el, offset, focal_plane_rotation=fr)
        np.testing.assert_allclose(bore_az, az, atol=0.01 / 3600.0)
        np.testing.assert_allclose(bore_el, el, atol=0.01 / 3600.0)


class TestInverseThresholdMagnitudes:
    """Pin the falsifiable magnitudes in the threshold comments.

    The values are deg, so deg*3600 = arcsec. Pin the true magnitudes so the
    comments cannot drift.
    """

    def test_early_exit_threshold_is_nanoarcsec(self):
        # 1e-12 deg * 3600 = 3.6e-9 arcsec = 3.6 nanoarcsec.
        arcsec = _INVERSE_EARLY_EXIT_THRESHOLD * 3600.0
        assert arcsec == pytest.approx(3.6e-9, rel=1e-9)

    def test_failure_threshold_is_milliarcsec(self):
        # 1e-6 deg * 3600 = 3.6e-3 arcsec = 3.6 milliarcsec.
        arcsec = _INVERSE_FAILURE_THRESHOLD * 3600.0
        assert arcsec == pytest.approx(3.6e-3, rel=1e-9)


class TestInverseZenithDegeneracy:
    """The inverse must not silently return a wrong azimuth at the pole.

    At the zenith pole, azimuth is degenerate: every boresight azimuth maps a
    pole-elevation detector to the same position, so the forward-residual
    convergence check reports success while the recovered azimuth is arbitrary.
    The pole guard raises a clear ``OffsetInversionError`` instead, a type
    inside the ``PointingError`` hierarchy the callers' Raises sections
    advertise, so geometric infeasibility is catchable with the rest.
    """

    def test_zenith_offset_raises_instead_of_wrong_azimuth(self):
        # bore=(180, 89), dx=60', fr=90 lands the detector at el=90 (the pole).
        offset = InstrumentOffset(dx=60.0, dy=0.0)
        det_az, det_el = boresight_to_detector(180.0, 89.0, offset, focal_plane_rotation=90.0)
        assert det_el == pytest.approx(90.0, abs=1e-3)

        with pytest.raises(OffsetInversionError, match="azimuth") as excinfo:
            detector_to_boresight(det_az, det_el, offset, focal_plane_rotation=90.0)
        # It is in the library's hierarchy, so a caller catching PointingError
        # (or ValueError) sees it.
        assert isinstance(excinfo.value, PointingError)

    def test_array_call_names_the_offending_samples(self):
        """An array call reports which samples tripped the pole guard.

        The whole call still refuses: a partially-inverted trajectory would be
        silently wrong at the degenerate samples. What the caller gains is the
        indices, so a long trajectory is diagnosable.
        """
        offset = InstrumentOffset(dx=60.0, dy=0.0)
        pole_az, pole_el = boresight_to_detector(180.0, 89.0, offset, focal_plane_rotation=90.0)
        safe_az, safe_el = boresight_to_detector(180.0, 45.0, offset, focal_plane_rotation=90.0)
        det_az = np.array([safe_az, pole_az, safe_az, pole_az])
        det_el = np.array([safe_el, pole_el, safe_el, pole_el])

        with pytest.raises(OffsetInversionError) as excinfo:
            detector_to_boresight(det_az, det_el, offset, focal_plane_rotation=90.0)
        assert excinfo.value.indices == (1, 3)
        assert "[1, 3]" in str(excinfo.value)

    def test_operational_envelope_still_round_trips(self):
        """The pole guard must not fire inside the real PrimeCam envelope."""
        offset = InstrumentOffset(dx=106.8, dy=0.0)  # inner ring ~1.78 deg
        for el in [20.0, 45.0, 70.0, 85.0]:
            for fr in [0.0, 90.0, 180.0, 270.0]:
                det_az, det_el = boresight_to_detector(200.0, el, offset, focal_plane_rotation=fr)
                bore_az, bore_el = detector_to_boresight(
                    det_az, det_el, offset, focal_plane_rotation=fr
                )
                assert bore_az == pytest.approx(200.0, abs=0.01 / 3600.0)
                assert bore_el == pytest.approx(el, abs=0.01 / 3600.0)


class TestDetectorToBoresightNonFinite:
    """Non-finite input is refused, and the convergence check fails closed.

    ``nan > threshold`` is False, so a threshold test written the other way
    round passes a NaN residual and, because one NaN makes the array maximum
    NaN, every other sample of the call unchecked.
    """

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    @pytest.mark.parametrize("name", ["det_az", "det_el", "focal_plane_rotation"])
    def test_scalar_non_finite_input_raises(self, name, bad):
        kwargs = {"det_az": 180.0, "det_el": 45.0, "focal_plane_rotation": 0.0}
        kwargs[name] = bad
        with pytest.raises(ValueError, match=name):
            detector_to_boresight(offset=InstrumentOffset(dx=5.0, dy=3.0), **kwargs)

    @pytest.mark.parametrize(
        ("offset", "name"),
        [
            (InstrumentOffset(dx=np.nan, dy=3.0), "offset.dx"),
            (InstrumentOffset(dx=5.0, dy=np.inf), "offset.dy"),
        ],
    )
    def test_non_finite_offset_raises(self, offset, name):
        with pytest.raises(ValueError, match=name):
            detector_to_boresight(180.0, 45.0, offset, focal_plane_rotation=0.0)

    def test_one_nan_sample_does_not_hide_an_unconverged_one(self):
        # 100 x 60 deg at el 60 does not converge on its own; beside a NaN it
        # was returned with an elevation of about 857 deg.
        offset = InstrumentOffset(dx=6000.0, dy=3600.0)
        with pytest.raises(OffsetInversionError, match="converge"):
            detector_to_boresight(180.0, 60.0, offset, focal_plane_rotation=0.0)
        with pytest.raises(ValueError, match="det_el"):
            detector_to_boresight(np.array([180.0, 180.0]), np.array([60.0, np.nan]), offset, 0.0)

    def test_nan_residual_from_finite_input_fails_closed(self):
        # A finite but absurd offset overflows inside the projection; the
        # residual is NaN and the inverse must refuse rather than return NaN.
        offset = InstrumentOffset(dx=1e308, dy=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises(OffsetInversionError, match="converge"):
                detector_to_boresight(180.0, 45.0, offset, focal_plane_rotation=0.0)

    def test_apply_detector_offset_names_the_non_finite_field(self, site):
        trajectory = Trajectory(
            times=np.array([0.0, 1.0, 2.0]),
            az=np.array([180.0, 180.1, 180.2]),
            el=np.full(3, 45.0),
            az_vel=np.full(3, 0.1),
            el_vel=np.zeros(3),
        )
        with pytest.raises(ValueError, match="offset.dx"):
            apply_detector_offset(trajectory, InstrumentOffset(dx=np.nan, dy=3.0), site=site)


class TestRotationKeyword:
    """The projections call their angle ``focal_plane_rotation``.

    The old ``field_rotation`` keyword shared its name with the celestial
    ``Coordinates.get_field_rotation``; no alias for it may come back.
    """

    @pytest.mark.parametrize(
        "call",
        [
            lambda **kw: boresight_to_detector(180.0, 45.0, InstrumentOffset(5.0, 3.0), **kw),
            lambda **kw: detector_to_boresight(180.0, 45.0, InstrumentOffset(5.0, 3.0), **kw),
            lambda **kw: sky_to_focal_plane(180.0, 45.0, 180.1, 45.1, **kw),
        ],
        ids=["boresight_to_detector", "detector_to_boresight", "sky_to_focal_plane"],
    )
    def test_old_keyword_is_rejected(self, call):
        with pytest.raises(TypeError, match="field_rotation"):
            call(field_rotation=30.0)


try:
    import scanning as _scanning  # noqa: F401

    HAS_SCANNING = True
except ImportError:
    HAS_SCANNING = False


@pytest.mark.slow
@pytest.mark.skipif(not HAS_SCANNING, reason="requires the scanning (scan_patterns) package")
class TestScanningModuleProjectionParity:
    """scanning's module projection matches ours, absolute sign included.

    The boresight-level parity tests never apply a module offset, so the
    field-rotation sign does not enter them; a round trip cancels a
    consistent sign error, and a flip-sensitivity test passes under
    either sign. This compares the two packages' detector placement
    directly, at an off-axis module and nonzero elevations, where a
    flipped mechanical rotation displaces the module by ``2 r sin(el)``
    (2.5 deg at el 45). Exactly that flip has shipped undetected once.
    """

    # The elevations below (25 and 80) sit outside the oracle's declared
    # 30-75 deg validity band, deliberately: the displacement under a flipped
    # sign scales as sin(el), so the ends of the range are where it is largest.
    # The comparison is exact (atol 1e-9), so an oracle that degraded outside
    # its band would fail this test rather than pass it silently.
    @pytest.mark.filterwarnings("ignore:elevation has values outside of 30 to 75 range")
    def test_from_boresight_matches_mechanical_rotation(self, site):
        import scanning

        # The same construction scanning's own behavioral tests use; its
        # private projection method is the exact seam under test.
        pong = scanning.Pong(num_term=4, width=2, height=2, spacing=0.1, velocity=0.5)
        tp = scanning.TelescopePattern(pong, start_ra=180, start_dec=-30, start_hrang=-1)

        dist, theta = 1.77985, 30.0
        az = np.array([0.0, 90.0, 180.0, 300.0])
        el = np.array([25.0, 45.0, 65.0, 80.0])

        their_az, their_el = tp._transform_from_boresight(az, el, dist, theta)

        offset = InstrumentOffset(
            dx=dist * 60.0 * np.cos(np.radians(theta)),
            dy=dist * 60.0 * np.sin(np.radians(theta)),
        )
        our_az, our_el = boresight_to_detector(
            az, el, offset, focal_plane_rotation=site.nasmyth_sign * el
        )

        az_delta = (np.asarray(their_az) - np.asarray(our_az) + 180.0) % 360.0 - 180.0
        np.testing.assert_allclose(az_delta, 0.0, atol=1e-9)
        np.testing.assert_allclose(their_el, our_el, atol=1e-9)

    @pytest.mark.filterwarnings("ignore:elevation has values outside of 30 to 75 range")
    def test_round_trip_through_both_seams(self):
        """The inverse seam of scanning calls ``detector_to_boresight`` correctly too.

        The test above reaches only the forward seam; this one also runs
        ``_transform_to_boresight``, the path scanning takes for a module
        offset, so an oracle that still spelled an old keyword in either
        call fails here.
        """
        import scanning

        pong = scanning.Pong(num_term=4, width=2, height=2, spacing=0.1, velocity=0.5)
        tp = scanning.TelescopePattern(pong, start_ra=180, start_dec=-30, start_hrang=-1)

        dist, theta = 1.77985, 30.0
        az = np.array([0.0, 90.0, 180.0, 300.0])
        el = np.array([25.0, 45.0, 65.0, 80.0])

        back_az, back_el = tp._transform_from_boresight(
            *tp._transform_to_boresight(az, el, dist, theta), dist, theta
        )

        az_delta = (np.asarray(back_az) - az + 180.0) % 360.0 - 180.0
        np.testing.assert_allclose(az_delta, 0.0, atol=1e-9)
        np.testing.assert_allclose(back_el, el, atol=1e-9)


class TestSkyToFocalPlane:
    """The inverse projection recovers the focal-plane offset a source sits at."""

    ARCSEC = 1.0 / 3600.0

    @pytest.mark.parametrize("el", [20.0, 45.0, 70.0, 89.0])
    @pytest.mark.parametrize("rotation", [0.0, 37.0, -120.0, 200.0])
    @pytest.mark.parametrize("dx, dy", [(5.0, 3.0), (-40.0, 10.0), (0.0, 0.0), (90.0, -90.0)])
    def test_round_trips_boresight_to_detector(self, el, rotation, dx, dy):
        offset = InstrumentOffset(dx=dx, dy=dy)
        det_az, det_el = boresight_to_detector(180.0, el, offset, focal_plane_rotation=rotation)
        xi, eta = sky_to_focal_plane(180.0, el, det_az, det_el, focal_plane_rotation=rotation)
        assert xi == pytest.approx(offset.dx_deg, abs=0.01 * self.ARCSEC)
        assert eta == pytest.approx(offset.dy_deg, abs=0.01 * self.ARCSEC)

    def test_axis_conventions_against_the_flat_sky(self):
        """An independent small-angle check: xi follows azimuth, eta follows elevation."""
        # Offsets small enough that the second-order spherical terms (which
        # scale as the offset squared) sit below the tolerance.
        bore_az, bore_el = 100.0, 40.0
        d_az, d_el = 0.02, 0.01
        xi, eta = sky_to_focal_plane(bore_az, bore_el, bore_az + d_az, bore_el + d_el, 0.0)
        assert xi == pytest.approx(d_az * np.cos(np.radians(bore_el)), abs=5e-6)
        assert eta == pytest.approx(d_el, abs=5e-6)
        # The rotation is applied the way the forward map applies it: a source
        # straight "up" in the horizon frame lands on the focal-plane axis that
        # the rotation carries onto the elevation direction.
        xi_r, eta_r = sky_to_focal_plane(bore_az, bore_el, bore_az, bore_el + d_el, 90.0)
        assert xi_r == pytest.approx(d_el, abs=5e-6)
        assert eta_r == pytest.approx(0.0, abs=5e-6)

    def test_coincident_positions_land_on_the_origin(self):
        assert sky_to_focal_plane(123.0, 33.0, 123.0, 33.0, 12.0) == (0.0, 0.0)

    def test_broadcasts_over_a_trajectory(self):
        offset = InstrumentOffset(dx=12.0, dy=-7.0)
        az = np.linspace(100.0, 140.0, 25)
        el = np.linspace(30.0, 60.0, 25)
        rotation = -el
        det_az, det_el = boresight_to_detector(az, el, offset, focal_plane_rotation=rotation)
        xi, eta = sky_to_focal_plane(az, el, det_az, det_el, rotation)
        np.testing.assert_allclose(xi, offset.dx_deg, atol=0.01 * self.ARCSEC)
        np.testing.assert_allclose(eta, offset.dy_deg, atol=0.01 * self.ARCSEC)
        assert isinstance(xi, np.ndarray) and xi.shape == az.shape

    def test_scalar_inputs_return_floats(self):
        xi, eta = sky_to_focal_plane(10.0, 50.0, 10.5, 50.2, 15.0)
        assert isinstance(xi, float) and isinstance(eta, float)
