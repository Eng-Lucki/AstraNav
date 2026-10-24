"""
3D attitude mathematics for the star tracker.

Frames
  Inertial (celestial): x -> RA 0h on the equator, z -> north celestial pole.
  Camera: x -> image right, y -> image down, z -> out through the lens (boresight).

The attitude matrix C rotates camera vectors into the inertial frame
(its columns are the camera axes written in inertial coordinates).
A pinhole camera with focal length f (pixels) maps a camera-frame
direction (X, Y, Z) to pixel (cx + f*X/Z, cy + f*Y/Z), which is the
gnomonic projection.

Roll is measured like in the image: 0 deg means north is up and east is
left (the normal view of the sky); positive roll turns the sky
clockwise on screen.
"""
import math

import numpy as np


def radec_to_vec(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=-1)


def vec_to_radec(v):
    v = np.asarray(v, dtype=np.float64)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    ra = np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360.0
    dec = np.degrees(np.arcsin(np.clip(v[..., 2], -1, 1)))
    return ra, dec


def local_axes(ra_deg, dec_deg):
    """East, north and outward unit vectors on the sky at (ra, dec)."""
    ra, dec = math.radians(ra_deg), math.radians(dec_deg)
    east = np.array([-math.sin(ra), math.cos(ra), 0.0])
    north = np.array([-math.sin(dec) * math.cos(ra), -math.sin(dec) * math.sin(ra), math.cos(dec)])
    up = np.cross(east, north)
    return east, north, up


def attitude_matrix(ra_deg, dec_deg, roll_deg):
    """Camera pointed at (ra, dec) with the given roll -> C (camera -> inertial)."""
    e, n, u = local_axes(ra_deg, dec_deg)
    c, s = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    x = -c * e + s * n
    y = -s * e - c * n
    return np.column_stack([x, y, u])


def attitude_to_pointing(C):
    """C -> (ra, dec, roll) in degrees."""
    ra, dec = vec_to_radec(C[:, 2])
    e, n, _ = local_axes(float(ra), float(dec))
    x = C[:, 0]
    roll = math.degrees(math.atan2(x @ n, -(x @ e)))
    return float(ra), float(dec), roll


PROJECTIONS = ("pinhole", "stereographic")


def project(C, f, cx, cy, vecs, projection="pinhole"):
    """Inertial unit vectors -> pixel coordinates. Returns (xy, in_front_of_camera).

    pinhole:       r = f * tan(angle from boresight)      (real camera lenses)
    stereographic: r = 2f * tan(angle / 2)                (Stellarium's default view)
    """
    cam = np.asarray(vecs) @ C          # row form of C.T @ v
    z = cam[:, 2]
    if projection == "stereographic":
        front = z > -0.5
        k = 2 * f / (1 + np.where(front, z, 1.0))
    else:
        front = z > 1e-6
        k = f / np.where(front, z, 1.0)
    xy = np.column_stack([cx + k * cam[:, 0], cy + k * cam[:, 1]])
    return xy, front


def pixels_to_camera(xy, f, cx, cy, projection="pinhole"):
    """Pixel coordinates -> unit direction vectors in the camera frame."""
    xy = np.atleast_2d(xy)
    dx, dy = (xy[:, 0] - cx) / f, (xy[:, 1] - cy) / f
    if projection == "stereographic":
        rho = np.hypot(dx, dy)
        theta = 2 * np.arctan(rho / 2)
        s = np.where(rho > 1e-12, np.sin(theta) / np.maximum(rho, 1e-12), 1.0)
        return np.column_stack([dx * s, dy * s, np.cos(theta)])
    v = np.column_stack([dx, dy, np.ones(len(xy))])
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def angle_from_center(r_px, f, projection="pinhole"):
    """Angle (radians) between the boresight and a point r_px pixels from the image centre."""
    return 2 * math.atan(r_px / (2 * f)) if projection == "stereographic" else math.atan(r_px / f)


def solve_wahba(v_cam, u_inertial, weights=None):
    """Best rotation C with C @ v_cam ~= u_inertial (Wahba's problem, SVD solution).

    This is the same problem real spacecraft star trackers solve: given star
    directions measured in the camera and the same stars' catalogue directions,
    find the rotation between the two frames that fits all of them best.
    """
    w = np.ones(len(v_cam)) if weights is None else np.asarray(weights)
    B = (u_inertial * w[:, None]).T @ v_cam
    U, _, Vt = np.linalg.svd(B)
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    return U @ np.diag([1.0, 1.0, d]) @ Vt


def angle_between(a, b):
    return math.acos(max(-1.0, min(1.0, float(np.dot(a, b)))))
