"""
Accuracy metrics: compares the pipeline's output with the simulator's ground truth.
"""
import numpy as np

from catalog import angular_distance
from nav import wrap_angle

CORRECT_POINTING_DEG = 0.5


def true_star_positions(truth):
    """All stars really present in the frame: visible catalog stars first, then background stars."""
    cat_xy = truth["star_xy"][truth["star_visible"]]
    if len(truth["bg_xy"]):
        return cat_xy, np.vstack([cat_xy, truth["bg_xy"]])
    return cat_xy, cat_xy


def classify_detections(truth, det, match_radius=3.0):
    """Distance from each detection to the nearest real star, and whether it is a genuine star."""
    _, true_xy = true_star_positions(truth)
    if len(det) == 0:
        return np.zeros(0), np.zeros(0, bool)
    if len(true_xy) == 0:
        return np.full(len(det), np.inf), np.zeros(len(det), bool)
    d = np.linalg.norm(det[:, None, :] - true_xy[None, :, :], axis=2)
    nearest = d.min(axis=1)
    return nearest, nearest < match_radius


def evaluate(truth, dip, solution, match_radius=3.0):
    """Return a dict of per-frame metrics."""
    det = dip.xy
    cat_xy, true_xy = true_star_positions(truth)

    out = {
        "detections": len(det),
        "catalog_visible": int(len(cat_xy)),
        "catalog_detected": 0,
        "false_detections": 0,
        "centroid_rmse_px": float("nan"),
        "roll_error_deg": float("nan"),
        "boresight_error_deg": float("nan"),
        "solved": solution is not None,
        "correct_id": False,
    }

    nearest, good = classify_detections(truth, det, match_radius)
    out["false_detections"] = int((~good).sum())
    if good.any():
        out["centroid_rmse_px"] = float(np.sqrt(np.mean(nearest[good] ** 2)))
    if len(det) and len(cat_xy):
        d = np.linalg.norm(det[:, None, :] - cat_xy[None, :, :], axis=2)
        out["catalog_detected"] = int((d.min(axis=0) < match_radius).sum())

    if solution is not None:
        # A solution counts as correct when it points where the camera really points
        err = angular_distance(*solution.boresight_radec, *truth["boresight_radec"])
        out["correct_id"] = err < CORRECT_POINTING_DEG
        if out["correct_id"]:
            out["roll_error_deg"] = wrap_angle(solution.roll_deg - truth["roll_deg"])
            out["boresight_error_deg"] = err
    return out
