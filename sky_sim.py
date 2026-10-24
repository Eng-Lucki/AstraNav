"""
Phase 1: Synthetic low-light night sky generator.

Renders what the star camera sensor would see: every star of the Yale
Bright Star Catalogue that falls in the field of view, drawn with a
Gaussian point spread function (PSF) at its exact sub-pixel position,
plus a sky background with Gaussian read noise and fixed hot pixels.
Optional harder conditions: light pollution near the horizon, scattered
moonlight, and motion streaks.

The camera points at a target constellation (or a random part of the
sky), rolled by the vehicle heading and with a slow pointing wobble, so
the solver has to recover the full 3D attitude.
"""
import math

import cv2
import numpy as np

from attitude import attitude_matrix, pixels_to_camera, project, vec_to_radec
from catalog import CONSTELLATIONS, FOCAL_PX
from star_catalog import get_catalog

RANDOM_SKY = "Random sky (lost in space)"


class SkySimulator:
    def __init__(self, width=950, height=650, seed=None):
        self.width = width
        self.height = height
        self.focal = FOCAL_PX
        self.cx, self.cy = (width - 1) / 2.0, (height - 1) / 2.0
        self.center = np.array([self.cx, self.cy])
        self.rng = np.random.default_rng(seed)
        self.catalog = get_catalog()

        # Tunable sensor / scene parameters (the GUI sliders change these)
        self.noise_sigma = 15.0      # read-noise standard deviation (grey levels)
        self.sky_level = 25.0        # sky background pedestal (grey levels)
        self.hot_pixel_count = 40
        self.mag_limit = 6.0         # faintest catalogue star drawn
        self.psf_sigma = 1.4         # star blur radius (px)
        self.jitter = 0.25           # platform vibration (px)
        self.obscured = False        # clouds blocking the view
        self.light_pollution = 0.0   # horizon glow strength at the bottom edge (grey levels)
        self.moon_glow = 0.0         # scattered moonlight peak (grey levels)
        self.blur_length = 0.0       # motion streak length during the exposure (px)
        self.blur_angle = 30.0       # motion streak direction (deg)

        self.offset = np.zeros(2)    # slow pointing wobble (px)
        self.new_random_pointing()

        # Hot pixels are sensor defects, so they stay at the same place every frame
        n_hot = 200
        self._hot_xy = np.column_stack([
            self.rng.integers(0, width, n_hot),
            self.rng.integers(0, height, n_hot),
        ])
        self._hot_val = self.rng.uniform(90, 200, n_hot)

        # Catalogue entry of every stick-figure star (nearest catalogue direction)
        self._figure_idx = {name: np.argmax(c.vec @ self.catalog.vec.T, axis=1)
                            for name, c in CONSTELLATIONS.items()}

        # Normalised shapes of the smooth sky-brightness patterns
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
        self._horizon_map = (yy / (height - 1)) ** 2 * (0.7 + 0.3 * xx / (width - 1))
        moon_xy, moon_r = (0.85 * width, 0.12 * height), 0.25 * height
        self._moon_map = 1.0 / (1.0 + ((xx - moon_xy[0]) ** 2 + (yy - moon_xy[1]) ** 2) / moon_r ** 2)

    def new_random_pointing(self):
        """Pick a uniformly random direction on the sky for the 'random sky' target."""
        self.random_pointing = (self.rng.uniform(0, 360), math.degrees(math.asin(self.rng.uniform(-1, 1))))
        self.offset = np.zeros(2)

    @staticmethod
    def magnitude_to_peak(mag):
        """Star brightness on the 8-bit sensor. Real magnitudes span a huge range,
        so the scale is compressed to keep faint stars visible in 8 bits."""
        return 300.0 * 10 ** (-0.2 * (np.asarray(mag) - 0.5))

    @staticmethod
    def motion_kernel(length, angle_deg):
        """Line-shaped blur kernel, centred so the streak's centre stays on the true star position."""
        size = (int(math.ceil(length)) + 2) | 1
        k = np.zeros((size, size), dtype=np.uint8)
        c = size // 2
        a = math.radians(angle_deg)
        dx, dy = 0.5 * length * math.cos(a), 0.5 * length * math.sin(a)
        to16 = lambda v: int(round(v * 16))   # cv2 sub-pixel drawing uses 4 fractional bits
        cv2.line(k, (to16(c - dx), to16(c - dy)), (to16(c + dx), to16(c + dy)), 255, 1, cv2.LINE_AA, 4)
        k = k.astype(np.float64)
        return k / k.sum()

    def configure(self, **settings):
        for key, value in settings.items():
            if not hasattr(self, key):
                raise AttributeError(f"SkySimulator has no setting '{key}'")
            setattr(self, key, value)

    def _offset_limits(self, rel, margin=25):
        """Allowed image shift that keeps the points rel (relative to centre) inside the frame."""
        lo = margin - self.center - rel.min(axis=0)
        hi = np.array([self.width, self.height]) - margin - self.center - rel.max(axis=0)
        bad = lo > hi
        mid = (lo + hi) / 2
        lo[bad] = mid[bad]
        hi[bad] = mid[bad]
        return lo, hi

    def _draw_star(self, img, x, y, peak):
        s = self.psf_sigma
        r = int(math.ceil(4 * s))
        x0, y0 = int(round(x)), int(round(y))
        xa, xb = max(x0 - r, 0), min(x0 + r + 1, self.width)
        ya, yb = max(y0 - r, 0), min(y0 + r + 1, self.height)
        if xa >= xb or ya >= yb:
            return
        gx = np.arange(xa, xb) - x
        gy = np.arange(ya, yb) - y
        patch = np.exp(-(gy[:, None] ** 2 + gx[None, :] ** 2) / (2 * s * s))
        img[ya:yb, xa:xb] += peak * patch

    def render(self, target, roll_deg):
        """Render one sensor frame looking at `target` (a constellation name or RANDOM_SKY).

        Returns (frame, truth) where frame is a uint8 greyscale image and truth
        holds the exact star positions and attitude used to make it.
        """
        const = CONSTELLATIONS.get(target)
        ra0, dec0 = (const.ra0, const.dec0) if const else self.random_pointing
        C0 = attitude_matrix(ra0, dec0, roll_deg)

        # Pointing wobble: slow random walk, kept so the target constellation stays in frame
        if const:
            rel = project(C0, self.focal, self.cx, self.cy, const.vec)[0] - self.center
        else:
            rel = np.array([[-150.0, -150.0], [150.0, 150.0]])
        self.offset += self.rng.normal(0, 2.0, 2)
        lo, hi = self._offset_limits(rel)
        self.offset = np.clip(self.offset, lo, hi)

        # Moving the image content by +offset = pointing at what pixel (centre - offset) showed
        b = C0 @ pixels_to_camera(self.center - self.offset, self.focal, self.cx, self.cy)[0]
        ra_b, dec_b = (float(v) for v in vec_to_radec(b))
        C = attitude_matrix(ra_b, dec_b, roll_deg)

        # Every catalogue star bright enough and inside the frame
        cat = self.catalog
        bright = np.flatnonzero(cat.mag <= self.mag_limit)
        xy, front = project(C, self.focal, self.cx, self.cy, cat.vec[bright])
        m = 12
        keep = front & (xy[:, 0] > -m) & (xy[:, 0] < self.width + m) & (xy[:, 1] > -m) & (xy[:, 1] < self.height + m)
        rendered_idx = bright[keep]
        rendered_xy = xy[keep] + self.rng.normal(0, self.jitter, (keep.sum(), 2))

        # Stars are drawn on their own layer so motion blur only smears starlight
        stars = np.zeros((self.height, self.width), dtype=np.float64)
        attenuation = 0.08 if self.obscured else 1.0
        for (x, y), i in zip(rendered_xy, rendered_idx):
            self._draw_star(stars, x, y, self.magnitude_to_peak(cat.mag[i]) * attenuation)
        if self.blur_length >= 1:
            stars = cv2.filter2D(stars, -1, self.motion_kernel(self.blur_length, self.blur_angle))

        img = (self.sky_level + self.light_pollution * self._horizon_map
               + self.moon_glow * self._moon_map + stars)

        n_hot = int(self.hot_pixel_count)
        hx, hy = self._hot_xy[:n_hot, 0], self._hot_xy[:n_hot, 1]
        img[hy, hx] = np.maximum(img[hy, hx], self._hot_val[:n_hot])

        img += self.rng.normal(0, self.noise_sigma, img.shape)
        frame = np.clip(img, 0, 255).astype(np.uint8)

        in_frame = ((rendered_xy[:, 0] >= 0) & (rendered_xy[:, 0] < self.width) &
                    (rendered_xy[:, 1] >= 0) & (rendered_xy[:, 1] < self.height))

        # Target constellation stars (for the "catalog found" metric)
        if const:
            pos = {int(i): p for i, p in zip(rendered_idx, rendered_xy)}
            fig = self._figure_idx[target]
            star_xy = np.array([pos.get(int(i), (np.nan, np.nan)) for i in fig])
            visible = np.isfinite(star_xy[:, 0])
            visible[visible] &= ((star_xy[visible, 0] >= 0) & (star_xy[visible, 0] < self.width) &
                                 (star_xy[visible, 1] >= 0) & (star_xy[visible, 1] < self.height))
        else:
            star_xy, visible = np.zeros((0, 2)), np.zeros(0, bool)

        truth = {
            "constellation": target,
            "roll_deg": roll_deg,
            "offset": self.offset.copy(),
            "attitude": C,
            "star_xy": star_xy,
            "star_visible": visible,
            "bg_xy": rendered_xy[in_frame],
            "bg_idx": rendered_idx[in_frame],
            "hot_xy": self._hot_xy[:n_hot].astype(float),
            "boresight_radec": (ra_b, dec_b),
            "obscured": self.obscured,
        }
        return frame, truth
