"""
Pointing targets / constellation stick figures, plus sky-coordinate helpers.

The stars themselves come from the Yale Bright Star Catalogue (star_catalog.py).
The constellations here are used as places to point the simulated camera
and as stick figures drawn on the overlay once the field is identified.
"""
import math

import numpy as np

from attitude import radec_to_vec

# Simulated star-tracker lens: how many sensor pixels one degree of sky covers.
# A real star tracker knows this from its lens focal length and pixel pitch.
PLATE_SCALE = 20.0  # px / degree  (950x650 px sensor -> ~45 x 32 deg field of view)
FOCAL_PX = PLATE_SCALE * 180.0 / math.pi

# name, RA (deg), Dec (deg), visual magnitude
_RAW_CATALOG = {
    "Orion": {
        "stars": [
            ("Betelgeuse", 88.79, 7.41, 0.50),
            ("Bellatrix", 81.28, 6.35, 1.64),
            ("Alnitak", 85.19, -1.94, 1.77),
            ("Alnilam", 84.05, -1.20, 1.69),
            ("Mintaka", 83.00, -0.30, 2.23),
            ("Saiph", 86.94, -9.67, 2.06),
            ("Rigel", 78.63, -8.20, 0.13),
        ],
        "lines": [(0, 1), (0, 2), (1, 4), (2, 3), (3, 4), (2, 5), (4, 6)],
    },
    "Big Dipper": {
        "stars": [
            ("Dubhe", 165.93, 61.75, 1.79),
            ("Merak", 165.46, 56.38, 2.37),
            ("Phecda", 178.46, 53.69, 2.44),
            ("Megrez", 183.86, 57.03, 3.31),
            ("Alioth", 193.51, 55.96, 1.77),
            ("Mizar", 200.98, 54.93, 2.27),
            ("Alkaid", 206.89, 49.31, 1.86),
        ],
        "lines": [(0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (4, 5), (5, 6)],
    },
    "Cassiopeia": {
        "stars": [
            ("Caph", 2.29, 59.15, 2.27),
            ("Schedar", 10.13, 56.54, 2.24),
            ("Navi", 14.18, 60.72, 2.47),
            ("Ruchbah", 21.45, 60.24, 2.68),
            ("Segin", 28.60, 63.67, 3.37),
        ],
        "lines": [(0, 1), (1, 2), (2, 3), (3, 4)],
    },
    "Cygnus": {
        "stars": [
            ("Deneb", 310.36, 45.28, 1.25),
            ("Sadr", 305.56, 40.26, 2.23),
            ("Albireo", 292.68, 27.96, 3.05),
            ("Gienah", 311.55, 33.97, 2.48),
            ("Fawaris", 296.24, 45.13, 2.87),
        ],
        "lines": [(0, 1), (1, 2), (3, 1), (1, 4)],
    },
}


def angular_distance(ra1, dec1, ra2, dec2):
    """Great-circle distance between two sky positions (degrees)."""
    ra1, dec1, ra2, dec2 = map(math.radians, (ra1, dec1, ra2, dec2))
    s = math.sin(dec1) * math.sin(dec2) + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2)
    return math.degrees(math.acos(max(-1.0, min(1.0, s))))


def format_ra(ra_deg):
    # Round to the displayed precision first so 59.96 s never shows as "60.0s"
    tenths = round((ra_deg % 360.0) / 15.0 * 36000) % (24 * 36000)
    h, rem = divmod(tenths, 36000)
    m, s = divmod(rem, 600)
    return f"{h:02d}h {m:02d}m {s / 10:04.1f}s"


def format_dec(dec_deg):
    sign = "+" if dec_deg >= 0 else "-"
    arcsec = round(abs(dec_deg) * 3600)
    deg, rem = divmod(arcsec, 3600)
    m, s = divmod(rem, 60)
    return f"{sign}{deg:02d}° {m:02d}' {s:02d}\""


class Constellation:
    """A stick figure: named stars with sky directions, and lines between them."""

    def __init__(self, name, data):
        self.name = name
        self.star_names = [s[0] for s in data["stars"]]
        self.mags = np.array([s[3] for s in data["stars"]])
        self.lines = data["lines"]
        ra = np.array([s[1] for s in data["stars"]])
        dec = np.array([s[2] for s in data["stars"]])
        self.vec = radec_to_vec(ra, dec)
        centre = self.vec.mean(axis=0)
        centre /= np.linalg.norm(centre)
        self.ra0 = math.degrees(math.atan2(centre[1], centre[0])) % 360
        self.dec0 = math.degrees(math.asin(centre[2]))

    def __len__(self):
        return len(self.vec)


CONSTELLATIONS = {name: Constellation(name, data) for name, data in _RAW_CATALOG.items()}
