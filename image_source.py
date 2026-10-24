"""
Loading real sky images (Stellarium screenshots, astrophotos, phone night shots).
"""
import math
import os

import cv2
import numpy as np

IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
MAX_SIDE = 1600   # larger photos are shrunk: faster, and stars still span a few pixels


def list_images(folder=IMAGES_DIR):
    """All image files in a folder and its subfolders, sorted by path."""
    found = []
    if os.path.isdir(folder):
        for root, _, files in os.walk(folder):
            found += [os.path.join(root, f) for f in files if f.lower().endswith(IMAGE_EXTS)]
    return sorted(found)


def load_image(path, max_side=MAX_SIDE):
    """Read an image file -> (BGR colour image, scale factor applied).

    np.fromfile + imdecode instead of cv2.imread so paths with non-ASCII
    characters (common in OneDrive folders) work on Windows.
    """
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return img, scale


def focal_from_fov(fov_deg, size_px, projection="pinhole"):
    """Focal length (px) for a field of view measured across size_px pixels."""
    half = math.radians(fov_deg) / 2.0
    if projection == "stereographic":
        return (size_px / 2.0) / (2 * math.tan(half / 2))
    return (size_px / 2.0) / math.tan(half)
