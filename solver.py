"""
Phase 3: Star identification (plate solving) and attitude determination.

Two modes, like a real spacecraft star tracker:

LOST-IN-SPACE  - no idea where the camera points. Every group of three
    bright detected stars forms a triangle. Its side lengths divided by the
    longest side give two ratios that do not change with rotation or zoom,
    so they are looked up in a pre-built index of ~200k triangles made from
    the Yale Bright Star Catalogue, covering the whole sky. Each match gives
    three star correspondences, from which the full 3D camera attitude is
    solved (Wahba's problem). The hypothesis is then checked by projecting
    every catalogue star into the image and counting how many land on a
    detected star; the best-supported hypothesis wins.

TRACKING       - when the previous frame was solved, its attitude is used
    as a prediction, so the stars only need to be re-matched nearby. Much
    faster, and what a tracker does most of the time.

If the focal length is unknown (a photo with unknown lens) it is estimated
from the matched stars as part of the solution.
"""
import itertools
import math
import os
import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from attitude import angle_from_center, attitude_to_pointing, pixels_to_camera, project, solve_wahba
from star_catalog import DATA_DIR, get_catalog

INDEX_VERSION = 1


@dataclass
class Solution:
    attitude: np.ndarray          # 3x3, camera -> inertial
    focal_px: float
    image_size: tuple             # (w, h)
    mirrored: bool                # image was left-right flipped
    roll_deg: float
    boresight_radec: tuple
    fov_deg: tuple                # (horizontal, vertical)
    matches: list                 # [(catalog index, detection index), ...]
    residual_px: float            # RMS distance between matched stars and catalogue prediction
    constellation: str
    triangle_xy: np.ndarray       # detected triangle that produced the identification
    mode: str                     # "LOST-IN-SPACE" or "TRACKING"
    hypotheses: int
    elapsed_ms: float
    projection: str = "pinhole"   # lens model the solution was computed with

    @property
    def center(self):
        return ((self.image_size[0] - 1) / 2.0, (self.image_size[1] - 1) / 2.0)

    def project(self, vecs):
        """Catalogue unit vectors -> pixel positions in the original (unflipped) image."""
        xy, front = project(self.attitude, self.focal_px, *self.center, vecs, self.projection)
        if self.mirrored:
            xy[:, 0] = self.image_size[0] - 1 - xy[:, 0]
        return xy, front


def tangent_triangles(A):
    """Unit vectors (n, 3, 3) -> triangle corners (n, 3, 2) on the plane touching the
    sphere at each triangle's centre, using image-like axes (x = west, y = south)."""
    c = A.sum(axis=1)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    ra = np.arctan2(c[:, 1], c[:, 0])
    dec = np.arcsin(np.clip(c[:, 2], -1, 1))
    east = np.stack([-np.sin(ra), np.cos(ra), np.zeros_like(ra)], axis=1)
    north = np.stack([-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)], axis=1)
    t = A / np.einsum("nij,nj->ni", A, c)[:, :, None]
    xi = np.einsum("nij,nj->ni", t, east)
    eta = np.einsum("nij,nj->ni", t, north)
    return np.stack([-xi, -eta], axis=2)


def triangle_signatures(P):
    """Rotation/scale-invariant description of triangles P (n, 3, 2).

    Vertices are ordered by the length of their opposite side (shortest first),
    so vertex i in an image triangle corresponds to vertex i in the matching
    catalogue triangle. Triangles with two near-equal sides (ambiguous order)
    or nearly collinear corners are marked invalid.
    """
    d = np.stack([np.linalg.norm(P[:, 1] - P[:, 2], axis=1),
                  np.linalg.norm(P[:, 0] - P[:, 2], axis=1),
                  np.linalg.norm(P[:, 0] - P[:, 1], axis=1)], axis=1)
    order = np.argsort(d, axis=1)
    s = np.take_along_axis(d, order, axis=1)
    longest = np.maximum(s[:, 2], 1e-12)
    r0, r1 = s[:, 0] / longest, s[:, 1] / longest
    Po = np.take_along_axis(P, order[:, :, None], axis=1)
    a, b, c = Po[:, 0], Po[:, 1], Po[:, 2]
    cross = (b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0]
    valid = ((r1 - r0) >= 0.03) & ((1 - r1) >= 0.03) & (np.abs(cross) >= 0.02 * longest ** 2)
    return r0, r1, longest, np.sign(cross).astype(np.int8), order, valid


def focal_from_pair(pa, pb, theta, projection="pinhole"):
    """Focal length (px) that makes the rays through pixels pa, pb (relative to the
    image centre) separated by angle theta. Pinhole: closed form, a quadratic in f^2.
    Other projections: bisection (the separation shrinks steadily as f grows)."""
    if projection != "pinhole":
        sep = lambda f: math.acos(min(1.0, float(np.dot(*pixels_to_camera(np.array([pa, pb]), f, 0, 0, projection)))))
        # Search from the shortest focal length that keeps both stars within ~127 deg of the
        # axis: below that the separation stops shrinking steadily with f
        lo = math.log(max(math.hypot(*pa), math.hypot(*pb), 1.0) / 4)
        hi = math.log(1e6)
        if not (sep(math.exp(hi)) < theta < sep(math.exp(lo))):
            return None
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if sep(math.exp(mid)) > theta else (lo, mid)
        return math.exp((lo + hi) / 2)
    k = math.cos(theta) ** 2
    p = pa[0] * pb[0] + pa[1] * pb[1]
    A = pa[0] ** 2 + pa[1] ** 2
    B = pb[0] ** 2 + pb[1] ** 2
    qa, qb, qc = k - 1.0, k * (A + B) - 2 * p, k * A * B - p * p
    disc = qb * qb - 4 * qa * qc
    if abs(qa) < 1e-15 or disc < 0:
        return None
    for F in sorted(((-qb + math.sqrt(disc)) / (2 * qa), (-qb - math.sqrt(disc)) / (2 * qa)), reverse=True):
        if F > 0 and p + F > 0:
            return math.sqrt(F)
    return None


class StarIdentifier:
    def __init__(self, index_mag=5.5, verify_mag=6.0, levels_deg=(3, 6, 12, 24, 40), neighbours=7,
                 ratio_tol=0.012, max_image_stars=15, min_matches=6):
        self.catalog = get_catalog()
        self.index_mag = index_mag
        self.levels_deg = levels_deg
        self.neighbours = neighbours
        self.ratio_tol = ratio_tol
        self.max_image_stars = max_image_stars
        self.min_matches = min_matches
        self.verify_idx = np.flatnonzero(self.catalog.mag <= verify_mag)
        self.verify_vec = self.catalog.vec[self.verify_idx]
        self._load_or_build_index()

    # ------------------------------------------------------------------ index
    def _index_path(self):
        lv = "-".join(str(x) for x in self.levels_deg)
        return os.path.join(DATA_DIR, f"triangle_index_v{INDEX_VERSION}_m{self.index_mag}_k{self.neighbours}_l{lv}.npz")

    def _load_or_build_index(self):
        path = self._index_path()
        if os.path.exists(path):
            z = np.load(path)
            if int(z["n_catalog"]) == len(self.catalog):
                self.idx_r0, self.idx_r1, self.idx_len = z["r0"], z["r1"], z["length"]
                self.idx_chir, self.idx_verts = z["chir"], z["verts"]
                return
        self._build_index()
        np.savez_compressed(path, r0=self.idx_r0, r1=self.idx_r1, length=self.idx_len, chir=self.idx_chir,
                            verts=self.idx_verts, n_catalog=len(self.catalog))

    def _build_index(self):
        """For every bright star, form triangles with its few brightest neighbours, at
        several angular scales so both narrow and wide fields of view are covered."""
        cat = self.catalog
        sel = np.flatnonzero(cat.mag <= self.index_mag)
        V = cat.vec[sel].astype(np.float32)
        mags = cat.mag[sel]
        dots = V @ V.T
        tris = set()
        for radius in self.levels_deg:
            cos_r = math.cos(math.radians(radius))
            for a in range(len(sel)):
                nb = np.flatnonzero(dots[a] > cos_r)
                nb = nb[nb != a]
                if len(nb) < 2:
                    continue
                nb = nb[np.argsort(mags[nb])[: self.neighbours]]
                for j, k in itertools.combinations(nb.tolist(), 2):
                    tris.add(tuple(sorted((a, j, k))))
        T = sel[np.array(sorted(tris))]
        P = tangent_triangles(cat.vec[T])
        r0, r1, length, chir, order, valid = triangle_signatures(P)
        verts = np.take_along_axis(T, order, axis=1)
        keep = np.flatnonzero(valid)
        srt = keep[np.argsort(r0[keep])]
        self.idx_r0, self.idx_r1, self.idx_len = r0[srt], r1[srt], length[srt]
        self.idx_chir, self.idx_verts = chir[srt], verts[srt]

    @property
    def index_size(self):
        return len(self.idx_r0)

    # ------------------------------------------------------------------ public
    def solve(self, det_xy, image_size, focal_px=None, focal_tol=0.2, prior=None,
              allow_mirror=False, time_budget=2.0, match_radius=None, projection="pinhole"):
        """Identify stars and return a Solution, or None.

        det_xy      detected star centroids, brightest first
        focal_px    focal length if known (exactly for a star tracker, roughly from a FOV hint)
        focal_tol   relative uncertainty of focal_px; small values keep it fixed
        prior       previous Solution, enables tracking mode
        projection  lens model: "pinhole" (real cameras) or "stereographic" (Stellarium)
        """
        t0 = time.perf_counter()
        det = np.asarray(det_xy, dtype=np.float64).reshape(-1, 2)
        if len(det) < 3:
            return None
        w, h = image_size
        radius = match_radius or max(4.0, 0.005 * max(w, h))
        ctx = dict(w=w, h=h, cx=(w - 1) / 2.0, cy=(h - 1) / 2.0, radius=radius, proj=projection,
                   fit_focal=focal_px is None or focal_tol > 0.05)

        if prior is not None and tuple(prior.image_size) == (w, h) and prior.projection == projection:
            d = self._flip(det, w) if prior.mirrored else det
            sol = self._track(d, prior, ctx, t0)
            if sol is not None:
                return sol

        deadline = t0 + time_budget
        sol = self._lost_in_space(det, focal_px, focal_tol, ctx, False, t0, deadline)
        if sol is None and allow_mirror:
            sol = self._lost_in_space(self._flip(det, w), focal_px, focal_tol, ctx, True, t0,
                                      time.perf_counter() + time_budget)
        return sol

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _flip(det, w):
        d = det.copy()
        d[:, 0] = w - 1 - d[:, 0]
        return d

    def _match(self, C, f, det, tree, ctx, radius):
        """Project catalogue stars with attitude C and pair them with detections.
        Returns (matches [(cat, det, dist)], number of catalogue stars in view)."""
        w, h = ctx["w"], ctx["h"]
        half_diag = angle_from_center(math.hypot(w, h) / 2, f, ctx["proj"])
        widest = math.radians(89 if ctx["proj"] == "pinhole" else 130)
        b = C[:, 2]
        near = self.verify_vec @ b > math.cos(min(half_diag * 1.05, widest))
        idx = self.verify_idx[near]
        xy, front = project(C, f, ctx["cx"], ctx["cy"], self.catalog.vec[idx], ctx["proj"])
        inside = front & (xy[:, 0] > -radius) & (xy[:, 0] < w + radius) & (xy[:, 1] > -radius) & (xy[:, 1] < h + radius)
        idx, xy = idx[inside], xy[inside]
        if len(idx) == 0:
            return [], 0
        dist, nn = tree.query(xy, distance_upper_bound=radius)
        ok = np.flatnonzero(np.isfinite(dist))
        matches, used = [], set()
        for k in ok[np.argsort(dist[ok])]:
            if nn[k] not in used:
                used.add(nn[k])
                matches.append((int(idx[k]), int(nn[k]), float(dist[k])))
        return matches, len(idx)

    def _chance_level(self, n_view, n_det, ctx, radius):
        """Expected number of matches if the attitude were random (stars landing on noise)."""
        return n_view * n_det * math.pi * radius ** 2 / (ctx["w"] * ctx["h"])

    def _accept(self, matches, n_view, n_det, ctx, radius):
        """A true solution beats chance by a wide margin AND explains the brightest
        stars in the image (a chance alignment of faint blobs does not)."""
        lam = self._chance_level(n_view, n_det, ctx, radius)
        if len(matches) < max(self.min_matches, lam + 4 * math.sqrt(lam) + 3):
            return False
        top = min(10, n_det)
        bright_hits = sum(1 for m in matches if m[1] < top)
        return bright_hits >= 0.4 * top

    def _best_focal(self, det_xy, u, f, ctx):
        """Golden-section search for the focal length that best fits the matched stars."""
        def cost(ff):
            v = pixels_to_camera(det_xy, ff, ctx["cx"], ctx["cy"], ctx["proj"])
            C = solve_wahba(v, u)
            return float(np.sum((v @ C.T - u) ** 2))
        lo, hi = math.log(f * 0.85), math.log(f * 1.18)
        g = (math.sqrt(5) - 1) / 2
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
        fa, fb = cost(math.exp(a)), cost(math.exp(b))
        for _ in range(30):
            if fa < fb:
                hi, b, fb = b, a, fa
                a = hi - g * (hi - lo)
                fa = cost(math.exp(a))
            else:
                lo, a, fa = a, b, fb
                b = lo + g * (hi - lo)
                fb = cost(math.exp(b))
        return math.exp((lo + hi) / 2)

    def _refine(self, C, f, det, tree, ctx, radii):
        """Re-match with the current attitude and re-solve using every matched star."""
        matches = []
        for r in radii:
            matches, _ = self._match(C, f, det, tree, ctx, r)
            if len(matches) < 3:
                return None
            ci = [m[0] for m in matches]
            di = [m[1] for m in matches]
            u = self.catalog.vec[ci]
            if ctx["fit_focal"] and len(matches) >= 5:
                f = self._best_focal(det[di], u, f, ctx)
            C = solve_wahba(pixels_to_camera(det[di], f, ctx["cx"], ctx["cy"], ctx["proj"]), u)
        matches, n_view = self._match(C, f, det, tree, ctx, ctx["radius"])
        return C, f, matches, n_view

    def _finish(self, C, f, matches, ctx, mirrored, tri_xy, mode, hyps, t0):
        cat = self.catalog
        ci = [m[0] for m in matches]
        di = [m[1] for m in matches]
        sol = Solution(attitude=C, focal_px=f, image_size=(ctx["w"], ctx["h"]), mirrored=mirrored,
                       roll_deg=0.0, boresight_radec=(0.0, 0.0), fov_deg=(0.0, 0.0),
                       matches=list(zip(ci, di)), residual_px=0.0, constellation="",
                       triangle_xy=tri_xy, mode=mode, hypotheses=hyps, elapsed_ms=0.0,
                       projection=ctx["proj"])
        ra, dec, roll = attitude_to_pointing(C)
        if mirrored:
            roll = -roll
        sol.roll_deg, sol.boresight_radec = roll, (ra, dec)
        sol.fov_deg = tuple(math.degrees(2 * angle_from_center(s / 2, f, ctx["proj"])) for s in (ctx["w"], ctx["h"]))
        sol.residual_px = float(np.sqrt(np.mean([m[2] ** 2 for m in matches])))
        # Constellation of the brightest identified stars
        bright = sorted(ci, key=lambda i: cat.mag[i])[:8]
        sol.constellation = cat.dominant_constellation(bright)
        sol.elapsed_ms = (time.perf_counter() - t0) * 1000
        return sol

    def _track(self, det, prior, ctx, t0):
        tree = cKDTree(det)
        ctx = dict(ctx, fit_focal=False)
        r = ctx["radius"]
        out = self._refine(prior.attitude, prior.focal_px, det, tree, ctx, (8 * r, 3 * r, r))
        if out is None:
            return None
        C, f, matches, n_view = out
        if not self._accept(matches, n_view, len(det), ctx, r):
            return None
        return self._finish(C, f, matches, ctx, prior.mirrored, np.zeros((0, 2)), "TRACKING", 0, t0)

    def _lost_in_space(self, det, focal_px, focal_tol, ctx, mirrored, t0, deadline):
        cand = det[: self.max_image_stars]
        tree = cKDTree(det)
        cx, cy, radius = ctx["cx"], ctx["cy"], ctx["radius"]
        combos = np.array(sorted(itertools.combinations(range(len(cand)), 3), key=sum))
        if len(combos) == 0:
            return None
        if focal_px:
            # Known (or roughly known) lens: measure triangles on the sphere, like the catalogue
            rays = pixels_to_camera(cand, focal_px, cx, cy, ctx["proj"])
            P = tangent_triangles(rays[combos])
        else:
            P = cand[combos]
        r0, r1, length, chir, order, valid = triangle_signatures(P)
        tol = self.ratio_tol if focal_px else 2 * self.ratio_tol

        best, best_score, hyps = None, 0.0, 0
        for t in np.flatnonzero(valid):
            if time.perf_counter() > deadline:
                break
            lo = np.searchsorted(self.idx_r0, r0[t] - tol)
            hi = np.searchsorted(self.idx_r0, r0[t] + tol)
            m = (np.abs(self.idx_r1[lo:hi] - r1[t]) < tol) & (self.idx_chir[lo:hi] == chir[t])
            if focal_px:
                m &= np.abs(length[t] / self.idx_len[lo:hi] - 1) < focal_tol + 0.03
            det3 = combos[t][order[t]]
            pts = cand[det3]
            for j in lo + np.flatnonzero(m):
                hyps += 1
                cat3 = self.idx_verts[j]
                u = self.catalog.vec[cat3]
                f = focal_px
                if not focal_px or ctx["fit_focal"]:
                    # Longest side joins ordered vertices 0 and 1
                    theta = math.acos(min(1.0, float(u[0] @ u[1])))
                    f = focal_from_pair(pts[0] - (cx, cy), pts[1] - (cx, cy), theta, ctx["proj"])
                    if f is None or f < 0.2 * max(ctx["w"], ctx["h"]):
                        continue
                v = pixels_to_camera(pts, f, cx, cy, ctx["proj"])
                C = solve_wahba(v, u)
                if np.max(np.linalg.norm(v @ C.T - u, axis=1)) > 0.03 * self.idx_len[j] + 2e-4:
                    continue   # shapes do not really agree
                matches, n_view = self._match(C, f, det, tree, ctx, radius)
                score = len(matches) - self._chance_level(n_view, len(det), ctx, radius)
                if score > best_score:
                    best_score, best = score, (C, f, pts, n_view, len(matches))
            if best is not None and best[4] >= 8 and best_score >= 0.5 * min(len(det), 40):
                break   # overwhelming evidence, no need to keep searching

        if best is None:
            return None
        C, f, pts, _, _ = best
        out = self._refine(C, f, det, tree, ctx, (radius, radius))
        if out is None:
            return None
        C, f, matches, n_view = out
        if not self._accept(matches, n_view, len(det), ctx, radius):
            return None
        tri = pts.copy()
        if mirrored:
            tri[:, 0] = ctx["w"] - 1 - tri[:, 0]
        return self._finish(C, f, matches, ctx, mirrored, tri, "LOST-IN-SPACE", hyps, t0)


def wrap_angle(a):
    return (a + 180.0) % 360.0 - 180.0
