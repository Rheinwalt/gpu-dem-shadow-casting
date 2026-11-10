"""Run with NUMBA_ENABLE_CUDASIM=1 for machines without a CUDA GPU."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "shadow", Path(__file__).resolve().parents[1] / "gpu-dem-shadow-casting.py"
)
shadow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shadow)


def cpu_shadow(z, pixel_size, azimuth, elevation, eps=0.0):
    """Vectorized CPU reference using the notebook's pixel-center sampling."""
    azimuth = np.broadcast_to(azimuth, z.shape)
    elevation = np.broadcast_to(elevation, z.shape)
    result = np.zeros(z.shape, dtype=np.int16)
    for y0, x0 in np.ndindex(z.shape):
        if elevation[y0, x0] == 90:
            continue
        phi = np.deg2rad(azimuth[y0, x0] % 360)
        dx, dy = np.sin(phi), np.cos(phi)
        step = 1 / max(abs(dx), abs(dy))
        # Enough samples to leave the raster in its dominant direction.
        t = np.arange(1, max(z.shape) + 1) * step
        x = np.floor(x0 + 0.5 + dx * t).astype(int)
        y = np.floor(y0 + 0.5 + dy * t).astype(int)
        inside = (x >= 0) & (x < z.shape[1]) & (y >= 0) & (y < z.shape[0])
        ray = z[y0, x0] + t[inside] * pixel_size * np.tan(np.deg2rad(elevation[y0, x0]))
        result[y0, x0] = np.any(z[y[inside], x[inside]] - ray > eps)
    return result


@pytest.mark.parametrize("azimuth", [0, 30, 45, 60, 90, 135, 180, 225, 270, 300, 315, 359])
@pytest.mark.parametrize("elevation,pixel_size", [(10, 1), (40, 2)])
def test_against_cpu_reference(azimuth, elevation, pixel_size):
    z = np.random.default_rng(42).uniform(0, 10, (5, 7))
    actual = shadow.cast_shadow(z, pixel_size, azimuth, elevation)
    np.testing.assert_array_equal(actual, cpu_shadow(z, pixel_size, azimuth, elevation))
    assert actual.dtype == np.int16


@pytest.mark.parametrize("azimuth", [0, 90, 180, 270])
def test_single_blocker_in_each_compass_direction(azimuth):
    z = np.zeros((3, 3))
    # Place the blocker on the boundary to exercise row/column zero too.
    y, x = {0: (2, 1), 90: (1, 2), 180: (0, 1), 270: (1, 0)}[azimuth]
    z[y, x] = 10
    assert shadow.cast_shadow(z, 1, azimuth, 30)[1, 1] == 1
    assert shadow.cast_shadow(z, 1, azimuth + 180, 30)[1, 1] == 0


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1), (5, 7), (16, 16)])
def test_flat_terrain_is_fully_initialized(shape):
    actual = shadow.cast_shadow(np.zeros(shape), 1, 225, 30)
    np.testing.assert_array_equal(actual, np.zeros(shape, dtype=np.int16))


def test_pixel_size_elevation_and_height_tolerance():
    z = np.array([[0, 2]], dtype=float)
    assert shadow.cast_shadow(z, 1, 90, 45)[0, 0] == 1
    assert shadow.cast_shadow(z, 3, 90, 45)[0, 0] == 0
    assert shadow.cast_shadow(z, 1, 90, 70)[0, 0] == 0
    assert shadow.cast_shadow(z, 1, 90, 45, eps=1.1)[0, 0] == 0
    assert shadow.cast_shadow(z, 1, 90, 0)[0, 0] == 1
    np.testing.assert_array_equal(shadow.cast_shadow(z, 1, 90, 90), [[0, 0]])


def test_azimuth_wraps():
    z = np.array([[0, 2]], dtype=float)
    expected = shadow.cast_shadow(z, 1, 90, 30)
    for angle in (-270, 450):
        np.testing.assert_array_equal(shadow.cast_shadow(z, 1, angle, 30), expected)


@pytest.mark.parametrize("kwargs", [
    {"z": np.zeros(3)},
    {"z": np.zeros((0, 3))},
    {"pixel_size": 0},
    {"pixel_size": np.inf},
    {"azimuth": np.nan},
    {"elevation": -1},
    {"elevation": 91},
    {"elevation": np.nan},
    {"eps": -1},
    {"eps": np.inf},
])
def test_invalid_inputs(kwargs):
    arguments = dict(z=np.zeros((2, 2)), pixel_size=1, azimuth=0, elevation=30)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        shadow.cast_shadow(**arguments)


@pytest.mark.parametrize("azimuth", [0, 90, 180, 270])
@pytest.mark.parametrize("raster_angles", ["both", "azimuth", "elevation"])
def test_constant_angle_rasters_match_parallel_rays(azimuth, raster_angles):
    z = np.random.default_rng(42).uniform(0, 10, (5, 7))
    az = np.full(z.shape, azimuth) if raster_angles != "elevation" else azimuth
    el = np.full(z.shape, 30) if raster_angles != "azimuth" else 30
    np.testing.assert_array_equal(
        shadow.cast_shadow(z, 2, az, el), shadow.cast_shadow(z, 2, azimuth, 30)
    )


@pytest.mark.parametrize("raster_angles", ["both", "azimuth", "elevation"])
def test_varying_angles_match_cpu_reference(raster_angles):
    rng = np.random.default_rng(5)
    z = rng.uniform(0, 30, (5, 7))
    azimuth = rng.uniform(-360, 720, z.shape) if raster_angles != "elevation" else 135
    elevation = rng.uniform(0, 90, z.shape) if raster_angles != "azimuth" else 25
    if np.ndim(elevation):
        elevation[0, 0] = 90
        elevation[-1, -1] = 0
    actual = shadow.cast_shadow(z, 2, azimuth, elevation, eps=0.25)
    expected = cpu_shadow(z, 2, azimuth, elevation, eps=0.25)
    np.testing.assert_array_equal(actual, expected)


def test_per_pixel_compass_directions_and_zenith():
    z = np.zeros((3, 3))
    z[1, 1] = 10
    azimuth = np.array([[45, 0, 315], [90, 0, 270], [135, 180, 225]])
    elevation = np.full(z.shape, 30.0)
    # West and east pixels look at the same obstacle; zenith overrides it.
    elevation[1, 2] = 90
    result = shadow.cast_shadow(z, 1, azimuth, elevation)
    assert result[1, 0] == 1
    assert result[1, 2] == 0
    assert result[0, 1] == 1
    assert result[2, 1] == 1
    np.testing.assert_array_equal(result, cpu_shadow(z, 1, azimuth, elevation))


def test_noncontiguous_angle_rasters():
    z = np.random.default_rng(0).uniform(0, 10, (3, 4))
    azimuth = np.arange(24).reshape(3, 8)[:, ::2] * 15
    elevation = np.full((4, 3), 30).T
    np.testing.assert_array_equal(
        shadow.cast_shadow(z, 1, azimuth, elevation),
        cpu_shadow(z, 1, azimuth, elevation),
    )


@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (5, 1)])
def test_angle_rasters_with_singleton_dimensions(shape):
    z = np.arange(np.prod(shape), dtype=float).reshape(shape)
    azimuth = np.full(shape, 90)
    elevation = np.full(shape, 10)
    np.testing.assert_array_equal(
        shadow.cast_shadow(z, 1, azimuth, elevation),
        cpu_shadow(z, 1, azimuth, elevation),
    )


def test_all_zenith_raster():
    z = np.array([[0, 10], [5, 2]])
    result = shadow.cast_shadow(z, 1, np.full(z.shape, 90), np.full(z.shape, 90))
    np.testing.assert_array_equal(result, np.zeros(z.shape, dtype=np.int16))


@pytest.mark.parametrize("angle", ["azimuth", "elevation"])
@pytest.mark.parametrize("value", [np.zeros((2, 1)), np.zeros(2),
                                   np.full((2, 2), np.nan), np.full((2, 2), np.inf)])
def test_invalid_angle_rasters(angle, value):
    arguments = dict(z=np.zeros((2, 2)), pixel_size=1, azimuth=0, elevation=30)
    arguments[angle] = value
    with pytest.raises(ValueError):
        shadow.cast_shadow(**arguments)


@pytest.mark.parametrize("value", [-1, 91])
def test_out_of_range_elevation_raster(value):
    elevation = np.full((2, 2), 30)
    elevation[0, 0] = value
    with pytest.raises(ValueError):
        shadow.cast_shadow(np.zeros((2, 2)), 1, 0, elevation)


@pytest.mark.parametrize("raster", [False, True])
def test_kernel_dispatch(monkeypatch, raster):
    def unexpected_kernel(*args, **kwargs):
        pytest.fail("Wrapper selected the wrong kernel")

    unused = "cuda_kern_cast_shadow" if raster else "cuda_kern_cast_shadow_angles"
    monkeypatch.setattr(shadow, unused, unexpected_kernel)
    azimuth = np.full((2, 2), 90) if raster else 90
    result = shadow.cast_shadow(np.zeros((2, 2)), 1, azimuth, 30)
    np.testing.assert_array_equal(result, np.zeros((2, 2), dtype=np.int16))
