"""
Phase 2: Classical digital image processing pipeline.

  A. Greyscale input
  B. Background removal       -> flattens light pollution / moon glow
  C. Spatial noise filter     -> removes hot pixels (salt noise) and read noise
  D. Thresholding             -> separates starlight from background
  E. Connected components     -> one blob per star
  F. Sub-pixel centroiding    -> precise star position from image moments
"""
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

BACKGROUND_METHODS = ["None", "Morphological top-hat", "Median background subtraction"]
NOISE_FILTERS = ["None", "Median", "Gaussian", "Bilateral"]
THRESHOLD_METHODS = ["Adaptive (median + k·σ)", "Otsu", "Fixed"]
CENTROID_METHODS = ["Intensity-weighted moments", "Binary shape moments"]

# Grey level the flattened background is shifted to, so negative noise is not clipped away
PEDESTAL = 20


@dataclass
class Star:
    x: float
    y: float
    flux: float   # background-subtracted brightness sum
    peak: float
    area: int


@dataclass
class DipResult:
    gray: np.ndarray
    corrected: np.ndarray
    filtered: np.ndarray
    binary: np.ndarray
    threshold: float
    background: float
    noise_sigma: float
    stars: list = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def xy(self):
        if not self.stars:
            return np.zeros((0, 2))
        return np.array([(s.x, s.y) for s in self.stars])


def estimate_background(img):
    """Robust background level and noise via 3-sigma clipping, so the few bright
    stars do not inflate the estimate (every 4th pixel is plenty of samples)."""
    f = img.ravel()[::4].astype(np.float32)
    bg = float(np.median(f))
    sigma = max(1.4826 * float(np.median(np.abs(f - bg))), 1.0)
    for _ in range(3):
        sel = f[np.abs(f - bg) < 3 * sigma]
        bg, sigma = float(np.median(sel)), max(float(sel.std()), 0.5)
    return bg, sigma


def remove_background(gray, method, kernel_size=21):
    """Flatten slowly-varying sky brightness while keeping the small, sharp stars."""
    k = int(kernel_size) | 1
    if method == "Morphological top-hat":
        # Opening (erode then dilate) with a disk larger than any star wipes the
        # stars out and leaves only the smooth background; image - opening = stars.
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, disk)
        bg = cv2.blur(opened, (k, k))   # smooth the blocky opening so noise stays centred
    elif method == "Median background subtraction":
        # A large median ignores the few star pixels in each window, so it follows
        # the sky glow but not the stars.
        bg = cv2.medianBlur(gray, min(k, 255))
    else:
        return gray
    out = gray.astype(np.int16) - bg.astype(np.int16) + PEDESTAL
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_noise_filter(img, method, ksize=3):
    if method == "Median":
        # A hot pixel is a single bright pixel, so it is never the median of its
        # neighbourhood and gets removed, while stars (several pixels wide) survive.
        return cv2.medianBlur(img, ksize)
    if method == "Gaussian":
        # Averages noise down, but only spreads hot pixels out instead of removing them
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    if method == "Bilateral":
        # Edge-preserving smoothing: only averages pixels of similar brightness
        return cv2.bilateralFilter(img, ksize, 40, ksize)
    return img.copy()


def process_frame(frame, background_method="None", background_kernel=21,
                  noise_filter="Median", filter_ksize=3,
                  threshold_method=THRESHOLD_METHODS[0], k_sigma=5.0,
                  fixed_threshold=135, min_area=3, max_area=800,
                  centroid_method=CENTROID_METHODS[0], max_stars=200):
    t0 = time.perf_counter()

    # Step A: greyscale (the simulated sensor is already mono)
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step B: background removal
    corrected = remove_background(gray, background_method, background_kernel)

    # Step C: spatial noise filter
    filtered = apply_noise_filter(corrected, noise_filter, filter_ksize)

    # Step D: thresholding
    background, sigma = estimate_background(filtered)
    if threshold_method == "Otsu":
        thr, _ = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold_method == "Fixed":
        thr = float(fixed_threshold)
    else:
        # Adaptive: the threshold follows the measured noise, so it keeps working
        # when the sky gets brighter or the sensor noisier.
        thr = background + k_sigma * sigma
    thr = float(min(thr, 254))
    binary = np.where(filtered > thr, 255, 0).astype(np.uint8)

    # Step E: connected components (8-connectivity), one label per blob
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Step F: centroiding on the background-corrected (but not smoothed) image
    raw_bg, _ = estimate_background(corrected)
    stars = []
    h, w = gray.shape
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = (int(v) for v in stats[i, :4])

        if centroid_method == CENTROID_METHODS[1]:
            # Binary shape moments: centre of the blob outline. Every pixel counts
            # the same, so the result is biased towards the blob's shape.
            mask = (labels[y:y + bh, x:x + bw] == i).astype(np.float64)
            m = cv2.moments(mask, binaryImage=True)
            cx, cy = x + m["m10"] / m["m00"], y + m["m01"] / m["m00"]
            win = corrected[y:y + bh, x:x + bw].astype(np.float64) - raw_bg
            flux = float(np.clip(win, 0, None)[mask > 0].sum())
            peak = float(corrected[y:y + bh, x:x + bw][mask > 0].max())
        else:
            # Intensity-weighted moments: centre of brightness. Weights are the
            # background-subtracted pixel values in a window around the blob, so
            # brighter pixels pull the centroid harder (true sub-pixel position).
            pad = 2
            xa, ya = max(x - pad, 0), max(y - pad, 0)
            xb, yb = min(x + bw + pad, w), min(y + bh + pad, h)
            win = corrected[ya:yb, xa:xb].astype(np.float64) - raw_bg
            weights = np.clip(win, 0, None)
            m00 = weights.sum()
            if m00 <= 0:
                continue
            gy, gx = np.mgrid[ya:yb, xa:xb]
            cx = float((weights * gx).sum() / m00)
            cy = float((weights * gy).sum() / m00)
            flux = float(m00)
            peak = float(corrected[ya:yb, xa:xb].max())

        stars.append(Star(cx, cy, flux, peak, area))

    stars.sort(key=lambda s: s.flux, reverse=True)
    stars = stars[:max_stars]

    return DipResult(gray=gray, corrected=corrected, filtered=filtered, binary=binary, threshold=thr,
                     background=background, noise_sigma=sigma, stars=stars,
                     elapsed_ms=(time.perf_counter() - t0) * 1000)
