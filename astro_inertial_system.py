"""
AstraNav - Astro-Inertial Navigation System
Launches the mission-control GUI.

Project modules:
  star_catalog.py  Yale Bright Star Catalogue loader (data/bsc5_catalog.gz, 9096 stars)
  attitude.py      3D attitude maths: projection, Wahba's problem, roll / pointing
  catalog.py       constellation targets + stick figures, sky-coordinate formatting
  sky_sim.py       Phase 1: synthetic noisy night sky rendered from the real catalogue
  dip_pipeline.py  Phase 2: background removal, noise filter, thresholding, sub-pixel centroiding
  solver.py        Phase 3: lost-in-space triangle star identification + tracking -> attitude
  nav.py           Phase 4: gyro dead reckoning + star-tracker drift correction
  metrics.py       accuracy against simulator ground truth
  benchmark.py     Monte-Carlo parameter sweeps (also runs from the command line)
  image_source.py  loading real sky images (images/ folder)
  gui.py           PySide6 interface

The first start builds a triangle index of the catalogue (a few seconds) and
caches it in data/ for later runs.
"""
from gui import main

if __name__ == "__main__":
    main()
