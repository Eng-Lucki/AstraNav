"""
Automated Monte-Carlo benchmark.

Sweeps one sky/sensor parameter over a range of values, renders many frames
at random camera roll for each value, and measures how well the pipeline
performs. Several pipeline variants (e.g. different noise filters) can be
compared on exactly the same frames.

Run from the command line for a quick report:
    python benchmark.py                       # noise sweep, comparing noise filters
    python benchmark.py --sweep "Hot pixels" --compare "Noise filters" --frames 100
"""
import argparse
import csv
import math
import os

import numpy as np

from catalog import CONSTELLATIONS
from dip_pipeline import BACKGROUND_METHODS, CENTROID_METHODS, NOISE_FILTERS, THRESHOLD_METHODS, process_frame
from metrics import evaluate
from sky_sim import SkySimulator
from solver import StarIdentifier

# Display name -> (SkySimulator attribute, values to test)
SWEEPS = {
    "Sensor noise σ": ("noise_sigma", [0, 5, 10, 15, 20, 25, 30, 35, 40, 50]),
    "Hot pixels": ("hot_pixel_count", [0, 25, 50, 75, 100, 125, 150, 200]),
    "PSF blur σ": ("psf_sigma", [0.8, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0]),
    "Motion blur length": ("blur_length", [0, 3, 6, 9, 12, 15, 20, 25]),
    "Light pollution": ("light_pollution", [0, 20, 40, 60, 80, 100, 120]),
    "Moon glow": ("moon_glow", [0, 25, 50, 75, 100, 150, 200]),
    "Faintest star (magnitude)": ("mag_limit", [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]),
}

# Algorithm-parameter sweeps: display name -> (process_frame argument, values,
# scene that makes the parameter matter, other DIP settings it needs).
# The scene and DIP settings here are applied on top of the caller's settings.
DIP_SWEEPS = {
    "Filter kernel size (px)": ("filter_ksize", [3, 5, 7],
                                {"noise_sigma": 25, "hot_pixel_count": 150}, {"min_area": 1}),
    "Adaptive threshold k (σ)": ("k_sigma", [2, 2.5, 3, 4, 5, 6, 8, 10, 12],
                                      {"noise_sigma": 15}, {}),
    "Top-hat disk size (px)": ("background_kernel", [3, 5, 7, 11, 15, 21, 31, 45, 61],
                               {"moon_glow": 120, "light_pollution": 60},
                               {"background_method": "Morphological top-hat"}),
    "Min blob area (px)": ("min_area", [1, 2, 3, 4, 6, 9, 12, 16],
                           {"noise_sigma": 15, "hot_pixel_count": 150}, {"noise_filter": "None"}),
}
SWEEPS.update({name: spec[:2] for name, spec in DIP_SWEEPS.items()})

# Display name -> {variant label: DIP setting overrides}
COMPARISONS = {
    "Current settings only": {"Current settings": {}},
    "Noise filters": {f: {"noise_filter": f} for f in NOISE_FILTERS},
    "Background removal": {b: {"background_method": b} for b in BACKGROUND_METHODS},
    "Threshold methods": {t: {"threshold_method": t} for t in THRESHOLD_METHODS},
    "Centroid methods": {c: {"centroid_method": c} for c in CENTROID_METHODS},
}

RESULT_FIELDS = ["variant", "parameter", "value", "frames", "solve_rate", "wrong_id_rate",
                 "catalog_detect_rate", "false_per_frame", "centroid_rmse_px",
                 "roll_rms_arcmin", "dip_ms"]


def run_sweep(sweep, comparison, frames=50, sim_settings=None, dip_settings=None,
              progress=None, should_stop=None, seed=0):
    """Run a parameter sweep. Returns a list of result dicts (one per variant per value).

    progress(done, total) is called after every frame; should_stop() lets a GUI cancel.
    """
    attr, values = SWEEPS[sweep]
    variants = COMPARISONS[comparison]
    sim_settings = dict(sim_settings or {})
    dip_settings = dict(dip_settings or {})
    is_dip = sweep in DIP_SWEEPS
    if is_dip:
        sim_settings.update(DIP_SWEEPS[sweep][2])
        dip_settings.update(DIP_SWEEPS[sweep][3])
    solver = StarIdentifier()
    names = list(CONSTELLATIONS)
    total, done = len(values) * frames, 0
    results = []

    for value in values:
        # Same seeds for every value, so the frames are identical and only the swept parameter changes
        sim = SkySimulator(seed=seed)
        rng = np.random.default_rng(seed)
        sim.configure(**sim_settings)
        if is_dip:
            dip_settings[attr] = value
        else:
            sim.configure(**{attr: value})
        acc = {label: [] for label in variants}

        for i in range(frames):
            if should_stop and should_stop():
                return results
            frame, truth = sim.render(names[i % len(names)], rng.uniform(-180, 180))
            for label, overrides in variants.items():
                dip = process_frame(frame, **{**dip_settings, **overrides})
                # Every frame is solved lost-in-space (no tracking), with the known lens
                sol = solver.solve(dip.xy, (sim.width, sim.height), focal_px=sim.focal, focal_tol=0.01)
                m = evaluate(truth, dip, sol)
                m["dip_ms"] = dip.elapsed_ms
                acc[label].append(m)
            done += 1
            if progress:
                progress(done, total)

        for label, ms in acc.items():
            results.append(summarise(label, sweep, value, ms))
    return results


def summarise(label, sweep, value, ms):
    def nanmean(xs):
        xs = [x for x in xs if not math.isnan(x)]
        return float(np.mean(xs)) if xs else float("nan")

    roll = [m["roll_error_deg"] for m in ms if not math.isnan(m["roll_error_deg"])]
    return {
        "variant": label,
        "parameter": sweep,
        "value": value,
        "frames": len(ms),
        "solve_rate": float(np.mean([m["correct_id"] for m in ms])),
        "wrong_id_rate": float(np.mean([m["solved"] and not m["correct_id"] for m in ms])),
        "catalog_detect_rate": float(np.mean([m["catalog_detected"] / max(m["catalog_visible"], 1) for m in ms])),
        "false_per_frame": float(np.mean([m["false_detections"] for m in ms])),
        "centroid_rmse_px": nanmean([m["centroid_rmse_px"] for m in ms]),
        "roll_rms_arcmin": float(np.sqrt(np.mean(np.square(roll))) * 60) if roll else float("nan"),
        "dip_ms": float(np.mean([m["dip_ms"] for m in ms])),
    }


def save_csv(results, path):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})


# The full set of experiments reported in the project report: (sweep, comparison, output file)
REPORT_RUNS = [
    ("Sensor noise σ", "Noise filters", "bench_noise_vs_filter.csv"),
    ("Moon glow", "Background removal", "bench_moon_vs_background.csv"),
    ("Light pollution", "Threshold methods", "bench_lightpollution_vs_threshold.csv"),
    ("Motion blur length", "Centroid methods", "bench_motionblur_vs_centroid.csv"),
    ("Filter kernel size (px)", "Noise filters", "param_filter_kernel_size.csv"),
    ("Adaptive threshold k (σ)", "Current settings only", "param_threshold_k.csv"),
    ("Top-hat disk size (px)", "Current settings only", "param_tophat_disk_size.csv"),
    ("Min blob area (px)", "Current settings only", "param_min_blob_area.csv"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default="Sensor noise σ", choices=list(SWEEPS))
    ap.add_argument("--compare", default="Noise filters", choices=list(COMPARISONS))
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--out", default="benchmark_results.csv")
    ap.add_argument("--all", metavar="FOLDER", help="run every experiment in the report, CSVs into FOLDER")
    args = ap.parse_args()

    def progress(done, total):
        if done % 25 == 0 or done == total:
            print(f"\r  {done}/{total} frames", end="", flush=True)

    if args.all:
        os.makedirs(args.all, exist_ok=True)
        for sweep, comp, fname in REPORT_RUNS:
            print(f"{sweep} | {comp}")
            save_csv(run_sweep(sweep, comp, args.frames, progress=progress), os.path.join(args.all, fname))
            print(f"  -> {fname}")
        return

    print(f"Sweep: {args.sweep}  |  Compare: {args.compare}  |  {args.frames} frames per point")
    results = run_sweep(args.sweep, args.compare, args.frames, progress=progress)
    print()
    print(f"{'variant':32s} {'value':>7s} {'solve':>6s} {'detect':>7s} {'false/fr':>8s} {'rmse px':>8s} {'roll′':>7s} {'ms':>6s}")
    for label in dict.fromkeys(r["variant"] for r in results):
        for r in (r for r in results if r["variant"] == label):
            print(f"{r['variant'][:32]:32s} {r['value']:7.1f} {r['solve_rate']:6.0%} {r['catalog_detect_rate']:7.0%} "
                  f"{r['false_per_frame']:8.2f} {r['centroid_rmse_px']:8.3f} {r['roll_rms_arcmin']:7.2f} {r['dip_ms']:6.1f}")
    save_csv(results, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
