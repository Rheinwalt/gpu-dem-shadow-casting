from math import cos, floor, pi, sin, tan

import numpy as np
from tqdm import trange
from numba import cuda


@cuda.jit
def cuda_kern_cast_shadow(shadow, z, ys, xs, pixelsize, tan_theta, dy_sun, dx_sun, eps):
    """Cast parallel rays using a single sun direction for the raster."""
    y0, x0 = cuda.grid(2)
    if y0 >= ys or x0 >= xs:
        return

    shadow[y0, x0] = _ray_shadow(
        z, ys, xs, y0, x0, pixelsize, tan_theta, dy_sun, dx_sun, eps
    )


@cuda.jit
def cuda_kern_cast_shadow_angles(shadow, z, ys, xs, pixelsize, azimuth, elevation, eps):
    """Read the source pixel's sun angles and cast its ray."""
    y0, x0 = cuda.grid(2)
    if y0 >= ys or x0 >= xs:
        return

    # A scalar angle is transferred as a 1x1 array for mixed inputs.
    if azimuth.size == 1:
        phi_deg = azimuth[0, 0]
    else:
        phi_deg = azimuth[y0, x0]
    if elevation.size == 1:
        theta_deg = elevation[0, 0]
    else:
        theta_deg = elevation[y0, x0]

    if theta_deg == 90.0:
        shadow[y0, x0] = 0
        return

    phi = phi_deg * (pi / 180.0)
    theta = theta_deg * (pi / 180.0)
    shadow[y0, x0] = _ray_shadow(
        z, ys, xs, y0, x0, pixelsize, tan(theta), cos(phi), sin(phi), eps
    )


@cuda.jit(device=True)
def _ray_shadow(z, ys, xs, y0, x0, pixelsize, tan_theta, dy_sun, dx_sun, eps):
    """Sample one fixed-direction ray from a pixel center to the boundary."""
    # dominant-axis stepping
    adx = abs(dx_sun)
    ady = abs(dy_sun)
    if adx > ady:
        step = 1.0 / adx
    else:
        step = 1.0 / ady

    x_start = x0 + 0.5
    y_start = y0 + 0.5

    # ray / boundary
    if dx_sun > 0.0:
        tx = (xs - x_start) / dx_sun
    elif dx_sun < 0.0:
        tx = (0. - x_start) / dx_sun
    else:
        tx = 1.0e30

    if dy_sun > 0.0:
        ty = (ys - y_start) / dy_sun
    elif dy_sun < 0.0:
        ty = (0. - y_start) / dy_sun
    else:
        ty = 1.0e30

    # Stop at the first raster boundary.
    if tx < ty:
        tmax = tx
    else:
        tmax = ty

    # make sure to stay inside
    tmax -= 1.0e-6

    # number of samples along the ray
    n = int(tmax / step) + 1

    z0 = z[y0, x0]

    # start at 1 to skip the source pixel
    for k in range(1, n):
        t = k * step
        x = int(floor(x_start + dx_sun * t))
        y = int(floor(y_start + dy_sun * t))
        r = t * pixelsize
        dz = z[y, x] - z0 - r * tan_theta
        if dz > eps:
            return 1

    return 0


def cast_shadow(z, pixel_size, azimuth, elevation, eps=0.0):
    """Return an int16 shadow mask (1 = shadow, 0 = illuminated).

    ``z`` is a 2D elevation raster with square pixels. ``pixel_size`` and
    ``eps`` use the same units as its heights (e.g., meters). Rows increase
    northward and columns eastward, as displayed with ``origin="lower"``.

    ``azimuth`` is in degrees clockwise from North: 0 = North, 90 = East,
    180 = South, 270 = West. ``elevation`` is in degrees above the horizon,
    from 0 (horizon) to 90 (zenith).

    Each angle may be a scalar or a 2D array with exactly the shape of ``z``.
    Angle rasters must be aligned with ``z`` and contain finite degree values.
    Scalar angles apply to every pixel. Raster angles set each source pixel's
    ray direction, which stays fixed along that ray.

    A pixel is shadowed when any sampled terrain toward the sun exceeds
    the ray height by more than ``eps``. Terrain outside the raster is ignored.
    """
    z = np.asarray(z)
    if z.ndim != 2 or z.size == 0:
        raise ValueError("z must be a nonempty 2D elevation raster")
    if not np.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError("pixel_size must be finite and positive")
    azimuth = np.asarray(azimuth, dtype=np.float64)
    elevation = np.asarray(elevation, dtype=np.float64)
    for name, angle in (("azimuth", azimuth), ("elevation", elevation)):
        if angle.ndim != 0 and angle.shape != z.shape:
            raise ValueError(f"{name} must be a scalar or a raster with shape {z.shape}")
    if not np.all(np.isfinite(azimuth)):
        raise ValueError("azimuth must be finite")
    if not np.all(np.isfinite(elevation) & (elevation >= 0) & (elevation <= 90)):
        raise ValueError("elevation must be between 0 and 90 degrees")
    if not np.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and nonnegative")
    if np.all(elevation == 90):
        return np.zeros(z.shape, dtype=np.int16)

    ys, xs = z.shape
    d_z = cuda.to_device(z.astype(np.float32))
    d_s = cuda.device_array((ys, xs), np.int16)
    nthreads = (16, 16)
    nblocks = (
        (ys + nthreads[0] - 1) // nthreads[0],
        (xs + nthreads[1] - 1) // nthreads[1],
    )
    if azimuth.ndim == 0 and elevation.ndim == 0:
        tan_theta = np.tan(np.deg2rad(elevation.item()))
        phi = np.deg2rad(azimuth.item() % 360)
        cuda_kern_cast_shadow[nblocks, nthreads](
            d_s, d_z, ys, xs, pixel_size, tan_theta, np.cos(phi), np.sin(phi), eps
        )
    else:
        d_azimuth = cuda.to_device(np.ascontiguousarray(np.atleast_2d(azimuth % 360)))
        d_elevation = cuda.to_device(np.ascontiguousarray(np.atleast_2d(elevation)))
        cuda_kern_cast_shadow_angles[nblocks, nthreads](
            d_s, d_z, ys, xs, pixel_size, d_azimuth, d_elevation, eps
        )
    return d_s.copy_to_host()


def draw_dome(z, xg, yg, x0, y0, r0):
    """
    Some half spheres for a synthetic elevation model.
    """
    dx = xg - x0
    dy = yg - y0
    sl = np.where(dx*dx + dy*dy < r0*r0)
    dx, dy = dx[sl], dy[sl]
    dz = np.sqrt(r0*r0 - dx*dx - dy*dy)
    z[sl] = np.max((z[sl], dz), axis=0)
    return z


if __name__ == "__main__":
    # example usage
    n = 3 * 1080
    xb = np.arange(16/9*n)
    yb = np.arange(n)
    xr = (xb[1:]+xb[:-1]) / 2
    yr = (yb[1:]+yb[:-1]) / 2
    xg, yg = np.meshgrid(xr, yr)
    z = np.zeros(xg.shape)
    for i in trange(300):
        x0 = np.random.random() * 16 / 9 * n
        y0 = np.random.random() * n
        r0 = np.random.random() * n / 10 + 1
        draw_dome(z, xg, yg, x0, y0, r0)

    height_image = z

    azimuth = 330  # clockwise from North
    elevation = 30  # above the horizon
    shadow_image = cast_shadow(height_image, xb[1]-xb[0], azimuth, elevation)
    np.save("height_image.npy", height_image)
    np.save("shadow_image.npy", shadow_image)
