"""
AstraNav mission-control GUI (PySide6 + pyqtgraph).

Shortcuts:  Space = play/pause   S = single step
            N = next target (simulator) / next image (image mode)   P = previous image
"""
import csv
import math
import os
import sys
from datetime import datetime, timezone

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from benchmark import COMPARISONS, SWEEPS, run_sweep, save_csv
from catalog import CONSTELLATIONS, format_dec, format_ra
from dip_pipeline import (BACKGROUND_METHODS, CENTROID_METHODS, NOISE_FILTERS, THRESHOLD_METHODS,
                          process_frame)
from image_source import IMAGE_EXTS, IMAGES_DIR, focal_from_fov, list_images, load_image
from metrics import classify_detections, evaluate
from nav import InertialNavigator
from sky_sim import RANDOM_SKY, SkySimulator
from solver import StarIdentifier
from star_catalog import get_catalog

ACCENT = "#00e0b0"
WARN = "#ffa040"
BAD = "#ff5060"
PLOT_WINDOW = 600  # seconds of history shown in the time plots


# --------------------------------------------------------------------------- widgets

class ImageView(QtWidgets.QLabel):
    """QLabel that keeps its image scaled to fit while preserving aspect ratio."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(240, 160)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.setStyleSheet("background:#050608; border:1px solid #222;")
        self._pix = None

    def set_image(self, img):
        img = np.ascontiguousarray(img)
        h, w = img.shape[:2]
        if img.ndim == 2:
            qimg = QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format_Grayscale8)
        else:
            qimg = QtGui.QImage(img.data, w, h, 3 * w, QtGui.QImage.Format_BGR888)
        self._pix = QtGui.QPixmap.fromImage(qimg.copy())
        self._rescale()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rescale()

    def _rescale(self):
        if self._pix is not None:
            self.setPixmap(self._pix.scaled(self.size(), QtCore.Qt.KeepAspectRatio,
                                            QtCore.Qt.SmoothTransformation))


class LabeledSlider(QtWidgets.QWidget):
    """Horizontal slider with a caption and live value readout, supporting float steps."""
    valueChanged = QtCore.Signal(float)

    def __init__(self, label, lo, hi, value, step=1.0, fmt="{:.0f}", suffix=""):
        super().__init__()
        self.step, self.fmt, self.suffix = step, fmt, suffix
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(round(lo / step), round(hi / step))
        self.slider.setValue(round(value / step))
        self.readout = QtWidgets.QLabel()
        self.readout.setMinimumWidth(62)
        self.readout.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.readout.setStyleSheet(f"color:{ACCENT}; font-family:Consolas;")
        caption = QtWidgets.QLabel(label)

        lay = QtWidgets.QGridLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.addWidget(caption, 0, 0)
        lay.addWidget(self.readout, 0, 1)
        lay.addWidget(self.slider, 1, 0, 1, 2)
        self.slider.valueChanged.connect(self._changed)
        self._changed()

    def value(self):
        return self.slider.value() * self.step

    def _changed(self, *_):
        self.readout.setText(self.fmt.format(self.value()) + self.suffix)
        self.valueChanged.emit(self.value())


def group(title, *widgets):
    box = QtWidgets.QGroupBox(title)
    lay = QtWidgets.QVBoxLayout(box)
    lay.setSpacing(4)
    for w in widgets:
        if isinstance(w, QtWidgets.QLayout):
            lay.addLayout(w)
        else:
            lay.addWidget(w)
    return box


def labeled(label, widget):
    row = QtWidgets.QHBoxLayout()
    row.addWidget(QtWidgets.QLabel(label))
    row.addWidget(widget, 1)
    return row


class BenchmarkWorker(QtCore.QThread):
    """Runs a benchmark sweep in the background so the window stays responsive."""
    progressed = QtCore.Signal(int, int)
    finished_with = QtCore.Signal(list)

    def __init__(self, sweep, comparison, frames, sim_settings, dip_settings):
        super().__init__()
        self.args = (sweep, comparison, frames, sim_settings, dip_settings)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        sweep, comparison, frames, sim_settings, dip_settings = self.args
        results = run_sweep(sweep, comparison, frames, sim_settings, dip_settings,
                            progress=self.progressed.emit, should_stop=lambda: self._stop)
        self.finished_with.emit(results)


# --------------------------------------------------------------------------- main window

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AstraNav — Astro-Inertial Navigation System")
        self.resize(1600, 960)

        self.catalog = get_catalog()
        self.sim = SkySimulator(seed=7)
        self.solver = StarIdentifier()
        self.nav = InertialNavigator(seed=3)
        self.frame_no = 0
        self.rows = []            # per-frame metrics for CSV export
        self.last = None          # (frame, truth, dip, solution, overlay); truth is None for real images
        self.was_locked = None
        self.prev_sol = None      # last solution, used for tracking mode
        self.mode = "sim"         # "sim" or "image"
        self.image_paths, self.image_idx, self.image_color = [], 0, None

        self._build_ui()
        self._build_shortcuts()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.step)
        self._set_fps(self.fps.value())
        self.log(f"Star catalogue loaded: {len(self.catalog)} stars, {self.solver.index_size:,} index triangles")
        self.log("System initialised. Baghdad reference base "
                 f"{self.nav.lat0:.4f}N {self.nav.lon0:.4f}E, heading {self.nav.heading0:.1f}°")
        self.step()
        self.toggle_run()

    # ------------------------------------------------------------------ layout
    def _build_ui(self):
        pg.setConfigOptions(antialias=True, background="#0d1014", foreground="#9aa3ad")
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        root.addWidget(self._build_controls())

        centre = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        centre.addWidget(self._build_views())
        centre.addWidget(self._build_plots())
        centre.setStretchFactor(0, 3)
        centre.setStretchFactor(1, 2)
        root.addWidget(centre, 1)

        root.addWidget(self._build_telemetry())

        self.statusBar().showMessage("Space: play/pause   S: step   N: next target / next image   P: previous image")

    def _build_controls(self):
        # --- source
        self.source = QtWidgets.QComboBox()
        self.source.addItems(["Simulator (live)", "Image files"])
        self.source.currentIndexChanged.connect(self.set_source)
        open_files = QtWidgets.QPushButton("Open image(s)…")
        open_files.clicked.connect(self.open_images)
        open_folder = QtWidgets.QPushButton("Open folder…")
        open_folder.clicked.connect(self.open_folder)
        prev_btn = QtWidgets.QPushButton("◀  Prev")
        prev_btn.clicked.connect(lambda: self.show_image(self.image_idx - 1))
        next_btn = QtWidgets.QPushButton("Next  ▶")
        next_btn.clicked.connect(lambda: self.show_image(self.image_idx + 1))
        self.img_label = QtWidgets.QLabel("No image loaded")
        self.img_label.setWordWrap(True)
        self.img_label.setStyleSheet("color:#8a93a0;")
        self.fov_hint = QtWidgets.QDoubleSpinBox()
        self.fov_hint.setRange(0, 170)
        self.fov_hint.setDecimals(1)
        self.fov_hint.setSuffix("°")
        self.fov_hint.setSpecialValueText("Unknown")
        self.fov_hint.editingFinished.connect(self.reprocess)
        self.fov_axis = QtWidgets.QComboBox()
        self.fov_axis.addItems(["across height", "across width"])
        self.fov_axis.currentIndexChanged.connect(self.reprocess)
        self.allow_mirror = QtWidgets.QCheckBox("Also try mirrored image")
        self.allow_mirror.setChecked(True)
        self.lens_model = QtWidgets.QComboBox()
        self.lens_model.addItems(["Auto (try both)", "Stereographic (Stellarium default)", "Pinhole (real camera)"])
        self.lens_model.currentIndexChanged.connect(self.reprocess)
        files_row = QtWidgets.QHBoxLayout()
        files_row.addWidget(open_files)
        files_row.addWidget(open_folder)
        nav_row = QtWidgets.QHBoxLayout()
        nav_row.addWidget(prev_btn)
        nav_row.addWidget(next_btn)
        fov_row = QtWidgets.QHBoxLayout()
        fov_row.addWidget(QtWidgets.QLabel("FOV"))
        fov_row.addWidget(self.fov_hint, 1)
        fov_row.addWidget(self.fov_axis, 1)
        self.image_box = QtWidgets.QWidget()
        ib = QtWidgets.QVBoxLayout(self.image_box)
        ib.setContentsMargins(0, 0, 0, 0)
        ib.addLayout(files_row)
        ib.addLayout(nav_row)
        ib.addWidget(self.img_label)
        ib.addLayout(fov_row)
        ib.addLayout(labeled("Lens", self.lens_model))
        ib.addWidget(self.allow_mirror)
        self.image_box.setEnabled(False)

        # --- sky simulation
        self.const_combo = QtWidgets.QComboBox()
        self.const_combo.addItems(list(CONSTELLATIONS) + [RANDOM_SKY])
        self.const_combo.currentTextChanged.connect(self._target_changed)
        self.noise = LabeledSlider("Sensor noise σ", 0, 50, 15, suffix=" DN")
        self.hot = LabeledSlider("Hot pixels", 0, 200, 40)
        self.mag_limit = LabeledSlider("Faintest star drawn", 3.0, 6.5, 6.0, step=0.1, fmt="{:.1f}", suffix=" mag")
        self.psf = LabeledSlider("PSF blur σ", 0.8, 3.0, 1.4, step=0.1, fmt="{:.1f}", suffix=" px")
        self.jitter = LabeledSlider("Vibration jitter", 0, 1.5, 0.25, step=0.05, fmt="{:.2f}", suffix=" px")
        self.mount = LabeledSlider("Camera mount roll", -180, 180, 0, suffix="°")
        self.obscure = QtWidgets.QCheckBox("Clouds (block star view)")
        self.obscure.toggled.connect(lambda on: self.log("Sky obscured by cloud" if on else "Sky clear"))

        # --- harder conditions
        self.light_poll = LabeledSlider("Light pollution (horizon)", 0, 120, 0, suffix=" DN")
        self.moon = LabeledSlider("Moon glow", 0, 200, 0, suffix=" DN")
        self.blur_len = LabeledSlider("Motion streak length", 0, 25, 0, step=0.5, fmt="{:.1f}", suffix=" px")
        self.blur_ang = LabeledSlider("Motion streak angle", 0, 180, 30, suffix="°")

        # --- DIP
        self.bg_method = QtWidgets.QComboBox()
        self.bg_method.addItems(BACKGROUND_METHODS)
        self.bg_kernel = LabeledSlider("Background kernel", 7, 61, 21, suffix=" px")
        self.noise_filter = QtWidgets.QComboBox()
        self.noise_filter.addItems(NOISE_FILTERS)
        self.noise_filter.setCurrentText("Median")
        self.filter_size = QtWidgets.QComboBox()
        self.filter_size.addItems(["3 x 3", "5 x 5", "7 x 7"])
        self.thr_method = QtWidgets.QComboBox()
        self.thr_method.addItems(THRESHOLD_METHODS)
        self.k_sigma = LabeledSlider("Adaptive k", 2, 12, 5, step=0.5, fmt="{:.1f}", suffix=" σ")
        self.fixed_thr = LabeledSlider("Fixed threshold", 0, 255, 135)
        self.min_area = LabeledSlider("Min blob area", 1, 20, 3, suffix=" px")
        self.centroid = QtWidgets.QComboBox()
        self.centroid.addItems(CENTROID_METHODS)
        self.tracking = QtWidgets.QCheckBox("Tracking mode after first fix")
        self.tracking.setChecked(True)
        self.thr_method.currentIndexChanged.connect(self._update_thr_enabled)
        self._update_thr_enabled()
        # While paused, re-run the pipeline on the same frame so parameter effects are visible
        for w in (self.k_sigma, self.fixed_thr, self.min_area, self.bg_kernel):
            w.valueChanged.connect(self.reprocess)
        for w in (self.bg_method, self.noise_filter, self.filter_size, self.thr_method, self.centroid):
            w.currentIndexChanged.connect(self.reprocess)

        # --- navigation
        self.bias = LabeledSlider("Gyro bias", 0, 0.1, 0.02, step=0.002, fmt="{:.3f}", suffix=" °/s")
        self.aiding = QtWidgets.QCheckBox("Star-tracker aiding")
        self.aiding.setChecked(True)
        self.aiding.toggled.connect(lambda on: self.log("Star aiding " + ("ENABLED" if on else "DISABLED")))
        self.dt = LabeledSlider("Sim time per frame", 0.5, 5, 1, step=0.5, fmt="{:.1f}", suffix=" s")
        reset_nav = QtWidgets.QPushButton("Reset navigation")
        reset_nav.clicked.connect(self.reset_nav)

        # --- display
        self.show_truth = QtWidgets.QCheckBox("Show ground truth")
        self.show_labels = QtWidgets.QCheckBox("Show star labels")
        self.show_labels.setChecked(True)
        self.show_truth.toggled.connect(self.reprocess)
        self.show_labels.toggled.connect(self.reprocess)

        # --- run
        self.run_btn = QtWidgets.QPushButton("▶  Run")
        self.run_btn.clicked.connect(self.toggle_run)
        step_btn = QtWidgets.QPushButton("⏭  Step")
        step_btn.clicked.connect(self.step)
        self.fps = LabeledSlider("Frame rate", 1, 20, 6, suffix=" fps")
        self.fps.valueChanged.connect(self._set_fps)
        run_row = QtWidgets.QHBoxLayout()
        run_row.addWidget(self.run_btn)
        run_row.addWidget(step_btn)

        # --- export
        save_img = QtWidgets.QPushButton("Save stage images…")
        save_img.clicked.connect(self.save_images)
        save_csv = QtWidgets.QPushButton("Export metrics CSV…")
        save_csv.clicked.connect(self.export_csv)

        panel = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 6, 0)
        lay.addWidget(group("Source", self.source, self.image_box))
        self.sim_groups = [
            group("Run", run_row, self.fps),
            group("Sky simulation", labeled("Target", self.const_combo), self.noise, self.hot,
                  self.mag_limit, self.psf, self.jitter, self.mount, self.obscure),
            group("Harder conditions", self.light_poll, self.moon, self.blur_len, self.blur_ang),
        ]
        for g in self.sim_groups:
            lay.addWidget(g)
        lay.addWidget(group("Image processing", labeled("Background", self.bg_method), self.bg_kernel,
                            labeled("Noise filter", self.noise_filter), labeled("Kernel size", self.filter_size),
                            labeled("Threshold", self.thr_method), self.k_sigma, self.fixed_thr,
                            self.min_area, labeled("Centroid", self.centroid), self.tracking))
        self.nav_group = group("Navigation", self.bias, self.dt, self.aiding, reset_nav)
        self.sim_groups.append(self.nav_group)
        lay.addWidget(self.nav_group)
        lay.addWidget(group("Display", self.show_truth, self.show_labels))
        lay.addWidget(group("Export", save_img, save_csv))
        lay.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(315)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        return scroll

    def _build_views(self):
        titles = {"overlay": "Tracker overlay", "raw": "1 · Raw sensor", "corrected": "2 · Background removed",
                  "filtered": "3 · Noise filtered", "binary": "4 · Binary mask"}
        self.views = {k: ImageView() for k in titles}
        self.grid_views = {k: ImageView() for k in titles}

        tabs = QtWidgets.QTabWidget()
        self.tab_keys = []
        for k, title in titles.items():
            tabs.addTab(self.views[k], title)
            self.tab_keys.append(k)

        grid = QtWidgets.QWidget()
        g = QtWidgets.QGridLayout(grid)
        g.setContentsMargins(2, 2, 2, 2)
        for i, k in enumerate(("raw", "corrected", "filtered", "binary", "overlay")):
            cell = QtWidgets.QVBoxLayout()
            cap = QtWidgets.QLabel(titles[k])
            cap.setStyleSheet("color:#8a93a0;")
            cell.addWidget(cap)
            cell.addWidget(self.grid_views[k], 1)
            g.addLayout(cell, i // 3, i % 3)
        tabs.addTab(grid, "All stages")
        self.tab_keys.append("grid")

        tabs.addTab(self._build_comparison(), "Filter comparison")
        self.tab_keys.append("compare")
        tabs.addTab(self._build_benchmark(), "Benchmark")
        self.tab_keys.append("bench")

        tabs.currentChanged.connect(lambda _: self._update_views())
        self.view_tabs = tabs
        return tabs

    def _build_comparison(self):
        """Same frame through every noise filter, side by side, with live statistics."""
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(2, 2, 2, 2)
        note = QtWidgets.QLabel("Each panel runs the full pipeline on the current frame with a different noise filter "
                                "(all other settings from the sidebar). Green = real star, red = false detection.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#8a93a0;")
        lay.addWidget(note)

        grid = QtWidgets.QGridLayout()
        self.compare_views = {}
        for i, f in enumerate(NOISE_FILTERS):
            v = ImageView()
            self.compare_views[f] = v
            grid.addWidget(v, i // 2, i % 2)
        lay.addLayout(grid, 1)

        cols = ["Filter", "Catalog found", "False detections", "Centroid RMSE (px)", "Solved",
                "Roll error (arcmin)", "Time (ms)"]
        self.compare_table = QtWidgets.QTableWidget(len(NOISE_FILTERS), len(cols))
        self.compare_table.setHorizontalHeaderLabels(cols)
        self.compare_table.verticalHeader().setVisible(False)
        self.compare_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.compare_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.compare_table.setFixedHeight(150)
        lay.addWidget(self.compare_table)
        return w

    def _build_benchmark(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)

        self.bench_sweep = QtWidgets.QComboBox()
        self.bench_sweep.addItems(list(SWEEPS))
        self.bench_compare = QtWidgets.QComboBox()
        self.bench_compare.addItems(list(COMPARISONS))
        self.bench_compare.setCurrentText("Noise filters")
        self.bench_frames = QtWidgets.QSpinBox()
        self.bench_frames.setRange(5, 1000)
        self.bench_frames.setValue(50)
        self.bench_frames.setSuffix(" frames / point")
        self.bench_run = QtWidgets.QPushButton("▶  Run benchmark")
        self.bench_run.clicked.connect(self.start_benchmark)
        self.bench_stop = QtWidgets.QPushButton("■  Stop")
        self.bench_stop.setEnabled(False)
        self.bench_stop.clicked.connect(self.stop_benchmark)
        self.bench_progress = QtWidgets.QProgressBar()
        self.bench_export = QtWidgets.QPushButton("Export results CSV…")
        self.bench_export.setEnabled(False)
        self.bench_export.clicked.connect(self.export_benchmark)
        note = QtWidgets.QLabel("Renders many frames at random roll for each value of the swept parameter. "
                                "Everything else uses the current sidebar settings. Variants are compared on "
                                "exactly the same frames.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#8a93a0;")

        side = QtWidgets.QWidget()
        side.setFixedWidth(250)
        s = QtWidgets.QVBoxLayout(side)
        s.addWidget(QtWidgets.QLabel("Sweep parameter"))
        s.addWidget(self.bench_sweep)
        s.addWidget(QtWidgets.QLabel("Compare"))
        s.addWidget(self.bench_compare)
        s.addWidget(self.bench_frames)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.bench_run)
        row.addWidget(self.bench_stop)
        s.addLayout(row)
        s.addWidget(self.bench_progress)
        s.addWidget(self.bench_export)
        s.addWidget(note)
        s.addStretch(1)
        lay.addWidget(side)

        plots = QtWidgets.QGridLayout()
        specs = [("solve_rate", "Solve rate", "%", 100), ("false_per_frame", "False detections", "per frame", 1),
                 ("centroid_rmse_px", "Centroid RMSE", "px", 1), ("roll_rms_arcmin", "Roll error (RMS)", "arcmin", 1)]
        self.bench_plots = {}
        for i, (key, title, unit, scale) in enumerate(specs):
            p = pg.PlotWidget()
            p.setTitle(title, color="#c8ced8", size="10pt")
            p.setLabel("left", unit)
            p.getAxis("left").enableAutoSIPrefix(False)
            p.showGrid(x=True, y=True, alpha=0.2)
            p.addLegend(offset=(8, 8))
            self.bench_plots[key] = (p, scale)
            plots.addWidget(p, i // 2, i % 2)
        lay.addLayout(plots, 1)
        self.bench_results = []
        self.bench_worker = None
        return w

    def _make_plot(self, title, ylabel):
        p = pg.PlotWidget()
        p.setTitle(title, color="#c8ced8", size="10pt")
        p.setLabel("left", ylabel)
        p.setLabel("bottom", "time (s)")
        p.getAxis("left").enableAutoSIPrefix(False)
        p.showGrid(x=True, y=True, alpha=0.2)
        p.addLegend(offset=(8, 8))
        return p

    def _build_plots(self):
        # Histogram of the filtered image with the threshold marked
        self.hist_plot = pg.PlotWidget()
        self.hist_plot.setTitle("Filtered image histogram (log)", color="#c8ced8", size="10pt")
        self.hist_plot.setLogMode(y=True)
        self.hist_plot.setLabel("bottom", "grey level")
        self.hist_curve = self.hist_plot.plot(stepMode="center", fillLevel=0, brush=(0, 224, 176, 60),
                                              pen=pg.mkPen(ACCENT, width=1))
        self.thr_line = pg.InfiniteLine(angle=90, pen=pg.mkPen(BAD, width=2), label="T",
                                        labelOpts={"color": BAD, "position": 0.9})
        self.hist_plot.addItem(self.thr_line)
        self.hist_plot.setXRange(0, 255)

        self.head_plot = self._make_plot("Heading error", "deg")
        self.head_ins = self.head_plot.plot(name="INS only", pen=pg.mkPen(WARN, width=2))
        self.head_aid = self.head_plot.plot(name="INS + star tracker", pen=pg.mkPen(ACCENT, width=2))

        self.pos_plot = self._make_plot("Position error", "m")
        self.pos_ins = self.pos_plot.plot(name="INS only", pen=pg.mkPen(WARN, width=2))
        self.pos_aid = self.pos_plot.plot(name="INS + star tracker", pen=pg.mkPen(ACCENT, width=2))

        self.acc_plot = self._make_plot("Measurement accuracy", "")
        self.acc_rmse = self.acc_plot.plot(name="centroid RMSE (px)", pen=pg.mkPen("#7aa7ff", width=2))
        self.acc_roll = self.acc_plot.plot(name="|roll error| (arcmin)", pen=pg.mkPen("#ff7ad9", width=2))

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self.pos_plot, "Position drift")
        tabs.addTab(self.head_plot, "Heading drift")
        tabs.addTab(self.acc_plot, "Accuracy")

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self.hist_plot)
        split.addWidget(tabs)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        return split

    def _build_telemetry(self):
        self.status = QtWidgets.QLabel("INITIALISING")
        self.status.setAlignment(QtCore.Qt.AlignCenter)
        self.status.setMinimumHeight(44)
        self.clock = QtWidgets.QLabel()
        self.clock.setAlignment(QtCore.Qt.AlignCenter)
        self.clock.setStyleSheet("color:#8a93a0; font-family:Consolas;")

        self.fields = {}

        def section(title, keys):
            box = QtWidgets.QGroupBox(title)
            form = QtWidgets.QFormLayout(box)
            form.setLabelAlignment(QtCore.Qt.AlignLeft)
            form.setHorizontalSpacing(10)
            form.setVerticalSpacing(3)
            for k in keys:
                v = QtWidgets.QLabel("—")
                v.setStyleSheet("font-family:Consolas; color:#e6e9ee;")
                v.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                self.fields[k] = v
                form.addRow(k, v)
            return box

        attitude = section("Attitude solution", ["Solver mode", "Constellation", "Stars matched", "Fit residual",
                                                 "Field of view", "Roll measured", "Roll true", "Roll error",
                                                 "Boresight RA", "Boresight Dec", "Pointing error"])
        dip = section("Image processing", ["Detections", "Catalog found", "False detections",
                                           "Centroid RMSE", "Background", "Noise σ", "Threshold",
                                           "DIP time", "Solve time"])
        nav = section("Navigation", ["Sim time", "True heading", "Heading err INS", "Heading err aided",
                                     "Gyro bias est.", "Pos err INS", "Pos err aided", "Position (aided)"])

        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(500)
        self.log_box.setStyleSheet("font-family:Consolas; font-size:9pt;")

        panel = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(6, 0, 0, 0)
        lay.addWidget(self.status)
        lay.addWidget(self.clock)
        lay.addWidget(attitude)
        lay.addWidget(dip)
        lay.addWidget(nav)
        lay.addWidget(QtWidgets.QLabel("Event log"))
        lay.addWidget(self.log_box, 1)
        panel.setFixedWidth(330)
        return panel

    def _build_shortcuts(self):
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), self, self.toggle_run)
        QtGui.QShortcut(QtGui.QKeySequence("S"), self, self.step)
        QtGui.QShortcut(QtGui.QKeySequence("N"), self, self.next_constellation)
        QtGui.QShortcut(QtGui.QKeySequence("P"), self, lambda: self.mode == "image" and self.show_image(self.image_idx - 1))

    # ------------------------------------------------------------------ actions
    def _update_thr_enabled(self):
        m = self.thr_method.currentText()
        self.k_sigma.setEnabled(m.startswith("Adaptive"))
        self.fixed_thr.setEnabled(m == "Fixed")

    def _set_fps(self, fps):
        self.timer.setInterval(int(1000 / max(fps, 1)))

    def toggle_run(self):
        if self.timer.isActive():
            self.timer.stop()
            self.run_btn.setText("▶  Run")
        elif self.mode == "sim":
            self.timer.start()
            self.run_btn.setText("⏸  Pause")

    def next_constellation(self):
        if self.mode == "image":
            self.show_image(self.image_idx + 1)
        elif self.const_combo.currentText() == RANDOM_SKY:
            self._target_changed(RANDOM_SKY)   # jump to another random part of the sky
        else:
            self.const_combo.setCurrentIndex((self.const_combo.currentIndex() + 1) % self.const_combo.count())

    def _target_changed(self, name):
        if name == RANDOM_SKY:
            self.sim.new_random_pointing()
            ra, dec = self.sim.random_pointing
            self.log(f"Camera slewed to a random sky position (RA {format_ra(ra)}, Dec {format_dec(dec)})")
        else:
            self.log(f"Target field switched to {name}")

    # ------------------------------------------------------------------ image files
    def set_source(self, index):
        image = index == 1
        self.mode = "image" if image else "sim"
        if image and self.timer.isActive():
            self.toggle_run()
        self.image_box.setEnabled(image)
        for g in self.sim_groups:
            g.setEnabled(not image)
        self.prev_sol = None
        if image:
            if not self.image_paths:
                self.image_paths = list_images()
            if self.image_paths:
                self.show_image(0)
            else:
                self.log(f"No images found in {IMAGES_DIR}. Use 'Open image(s)…' or put files there.")
        else:
            self.step()

    def open_images(self):
        pattern = " ".join(f"*{e}" for e in IMAGE_EXTS)
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Open sky images", IMAGES_DIR, f"Images ({pattern})")
        if paths:
            self.image_paths = sorted(paths)
            self.show_image(0)

    def open_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Open folder of sky images", IMAGES_DIR)
        if folder:
            self.image_paths = list_images(folder)
            if self.image_paths:
                self.show_image(0)
            else:
                self.log(f"No images found in {folder}")

    def show_image(self, i):
        if self.mode != "image" or not self.image_paths:
            return
        self.image_idx = i % len(self.image_paths)
        path = self.image_paths[self.image_idx]
        try:
            self.image_color, scale = load_image(path)
        except ValueError as e:
            self.log(str(e))
            return
        h, w = self.image_color.shape[:2]
        note = f" (shrunk to {w}x{h})" if scale < 1 else f" ({w}x{h})"
        self.img_label.setText(f"{self.image_idx + 1}/{len(self.image_paths)}: {os.path.basename(path)}{note}")
        self.process_image()

    def image_focal_hint(self, w, h, projection="pinhole"):
        fov = self.fov_hint.value()
        if fov <= 0:
            return None
        return focal_from_fov(fov, h if self.fov_axis.currentIndex() == 0 else w, projection)

    def process_image(self):
        if self.image_color is None:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            gray = cv2.cvtColor(self.image_color, cv2.COLOR_BGR2GRAY)
            dip, sol = self._run_pipeline(gray)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        overlay = self.draw_overlay(gray, None, dip, sol, base=self.image_color)
        self.last = (gray, None, dip, sol, overlay)
        self._update_views()
        counts, edges = np.histogram(dip.filtered, bins=128, range=(0, 256))
        self.hist_curve.setData(edges, counts + 1)
        self.thr_line.setValue(dip.threshold)
        self._update_telemetry(None, dip, sol, self._image_metrics(dip))

        name = os.path.basename(self.image_paths[self.image_idx])
        if sol is None:
            self.log(f"{name}: NOT SOLVED ({len(dip.stars)} stars detected). Try a FOV hint, "
                     "or adjust the threshold / min blob area.")
            return
        names = [self.catalog.label(ci, greek=True) for ci, _ in
                 sorted(sol.matches, key=lambda m: self.catalog.mag[m[0]])[:12]]
        self.log(f"{name}: SOLVED in {sol.elapsed_ms:.0f} ms — {sol.constellation}, "
                 f"RA {format_ra(sol.boresight_radec[0])}, Dec {format_dec(sol.boresight_radec[1])}, "
                 f"roll {sol.roll_deg:+.1f}°, FOV {sol.fov_deg[0]:.1f}° x {sol.fov_deg[1]:.1f}°"
                 f"{', mirrored' if sol.mirrored else ''}, {sol.projection} lens, {len(sol.matches)} stars matched, "
                 f"residual {sol.residual_px:.2f} px")
        self.log("  Identified: " + ", ".join(names))

    @staticmethod
    def _image_metrics(dip):
        nan = float("nan")
        return {"detections": len(dip.stars), "catalog_visible": 0, "catalog_detected": 0, "false_detections": 0,
                "centroid_rmse_px": nan, "roll_error_deg": nan, "boresight_error_deg": nan,
                "solved": False, "correct_id": False}

    def reset_nav(self):
        self.nav.reset()
        self.rows.clear()
        self.log("Navigation reset to reference base")

    def log(self, msg):
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self.log_box.appendPlainText(f"[{ts}] {msg}")

    # ------------------------------------------------------------------ main loop
    def sim_settings(self):
        return dict(noise_sigma=self.noise.value(), hot_pixel_count=int(self.hot.value()),
                    mag_limit=self.mag_limit.value(), psf_sigma=self.psf.value(),
                    jitter=self.jitter.value(), obscured=self.obscure.isChecked(),
                    light_pollution=self.light_poll.value(), moon_glow=self.moon.value(),
                    blur_length=self.blur_len.value(), blur_angle=self.blur_ang.value())

    def dip_settings(self):
        return dict(background_method=self.bg_method.currentText(), background_kernel=int(self.bg_kernel.value()),
                    noise_filter=self.noise_filter.currentText(),
                    filter_ksize=[3, 5, 7][self.filter_size.currentIndex()],
                    threshold_method=self.thr_method.currentText(), k_sigma=self.k_sigma.value(),
                    fixed_threshold=self.fixed_thr.value(), min_area=int(self.min_area.value()),
                    centroid_method=self.centroid.currentText())

    def step(self):
        if self.mode != "sim":
            return
        sim, nav = self.sim, self.nav
        sim.configure(**self.sim_settings())
        nav.gyro_bias = self.bias.value()

        # 1. Vehicle moves; inertial navigators integrate the gyro
        nav.predict(self.dt.value())

        # 2. Camera sees the sky rotated by the vehicle's true heading (+ mount roll)
        mount = self.mount.value()
        roll_true = (nav.true_heading + mount + 180) % 360 - 180
        frame, truth = sim.render(self.const_combo.currentText(), roll_true)

        # 3. Classical DIP pipeline, 4. star identification (tracking from the last fix if possible)
        dip, sol = self._run_pipeline(frame, prior=self.prev_sol if self.tracking.isChecked() else None)
        self.prev_sol = sol

        # 5. Attitude -> heading fix for the aided navigator
        fix = sol is not None and self.aiding.isChecked()
        if fix:
            nav.star_fix(sol.roll_deg - mount)
        nav.record(fix)

        m = evaluate(truth, dip, sol)
        self.frame_no += 1
        self._log_transitions(sol, m)
        self._record_row(truth, dip, sol, m)

        overlay = self.draw_overlay(frame, truth, dip, sol)
        self.last = (frame, truth, dip, sol, overlay)
        self._update_views()
        self._update_plots(dip)
        self._update_telemetry(truth, dip, sol, m)

    def _run_pipeline(self, frame, prior=None, **overrides):
        dip = process_frame(frame, **{**self.dip_settings(), **overrides})
        h, w = frame.shape[:2]
        if self.mode == "image":
            # Unknown camera: optional FOV hint, focal length is refined by the solver
            choice = self.lens_model.currentIndex()
            models = ["stereographic", "pinhole"] if choice == 0 else [["", "stereographic", "pinhole"][choice]]
            sol = None
            for proj in models:
                hint = self.image_focal_hint(w, h, proj)
                # Solving an unknown camera can take a few seconds on a busy laptop; the limit only
                # matters when an image cannot be solved
                s = self.solver.solve(dip.xy, (w, h), focal_px=hint, focal_tol=0.25, projection=proj,
                                      allow_mirror=self.allow_mirror.isChecked(), time_budget=10.0)
                # Keep the lens model that explains more stars, then the tighter fit
                if s and (sol is None or (len(s.matches), -s.residual_px) > (len(sol.matches), -sol.residual_px)):
                    sol = s
                if sol is not None and sol.residual_px < 2.0:
                    break   # a tight fit: no need to try the other lens model
        else:
            # Simulated star tracker: the lens is known exactly
            sol = self.solver.solve(dip.xy, (w, h), focal_px=self.sim.focal, focal_tol=0.01,
                                    prior=prior, time_budget=1.0)
        return dip, sol

    def reprocess(self, *_):
        """Re-run DIP + solver on the current frame without advancing time (used while paused)."""
        if self.mode == "image":
            self.process_image()
            return
        if self.last is None or self.timer.isActive():
            return
        frame, truth = self.last[0], self.last[1]
        dip, sol = self._run_pipeline(frame)
        m = evaluate(truth, dip, sol)
        overlay = self.draw_overlay(frame, truth, dip, sol)
        self.last = (frame, truth, dip, sol, overlay)
        self._update_views()
        counts, edges = np.histogram(dip.filtered, bins=128, range=(0, 256))
        self.hist_curve.setData(edges, counts + 1)
        self.thr_line.setValue(dip.threshold)
        self._update_telemetry(truth, dip, sol, m)

    def _log_transitions(self, sol, m):
        state = sol.mode if sol is not None else None
        if state != self.was_locked:
            if state == "LOST-IN-SPACE":
                self.log(f"OPTICAL LOCK (lost-in-space): {sol.constellation}, {len(sol.matches)} stars, "
                         f"{sol.hypotheses} hypotheses in {sol.elapsed_ms:.0f} ms")
            elif state == "TRACKING":
                self.log("Switched to tracking mode")
            elif self.was_locked is not None:
                self.log("Lost star lock — coasting on inertial dead reckoning")
        if sol is not None and not m["correct_id"]:
            self.log(f"WARNING: wrong attitude solution ({sol.constellation})")
        self.was_locked = state

    def _record_row(self, truth, dip, sol, m):
        h = self.nav.history
        self.rows.append({
            "frame": self.frame_no,
            "sim_time_s": round(self.nav.t, 2),
            "target": truth["constellation"],
            "solved_as": sol.constellation if sol else "",
            "solver_mode": sol.mode if sol else "",
            "stars_matched": len(sol.matches) if sol else 0,
            "detections": m["detections"],
            "catalog_detected": m["catalog_detected"],
            "catalog_visible": m["catalog_visible"],
            "false_detections": m["false_detections"],
            "centroid_rmse_px": round(m["centroid_rmse_px"], 4),
            "roll_true_deg": round(truth["roll_deg"], 4),
            "roll_meas_deg": round(sol.roll_deg, 4) if sol else "",
            "roll_error_deg": round(m["roll_error_deg"], 5),
            "boresight_error_deg": round(m["boresight_error_deg"], 5),
            "threshold": round(dip.threshold, 2),
            "noise_sigma_setting": self.sim.noise_sigma,
            "light_pollution": self.sim.light_pollution,
            "moon_glow": self.sim.moon_glow,
            "blur_length_px": self.sim.blur_length,
            "background_method": self.bg_method.currentText(),
            "noise_filter": f"{self.noise_filter.currentText()} {self.filter_size.currentText()}",
            "threshold_method": self.thr_method.currentText(),
            "centroid_method": self.centroid.currentText(),
            "ins_heading_err_deg": round(h["ins_heading_err"][-1], 4),
            "aided_heading_err_deg": round(h["aided_heading_err"][-1], 4),
            "ins_pos_err_m": round(h["ins_pos_err"][-1], 2),
            "aided_pos_err_m": round(h["aided_pos_err"][-1], 2),
            "dip_ms": round(dip.elapsed_ms, 2),
            "solve_ms": round(sol.elapsed_ms, 2) if sol else "",
        })

    # ------------------------------------------------------------------ drawing
    def draw_overlay(self, frame, truth, dip, sol, base=None):
        """Annotated view. truth is None for real images; base is an optional colour image to draw on."""
        if base is not None:
            img = base.copy()
        else:
            # Gamma stretch so faint stars are visible to the eye (display only)
            lut = (255 * (np.arange(256) / 255.0) ** 0.55).astype(np.uint8)
            img = cv2.cvtColor(cv2.LUT(frame, lut), cv2.COLOR_GRAY2BGR)
        h, w = frame.shape[:2]
        aa = cv2.LINE_AA
        font = cv2.FONT_HERSHEY_SIMPLEX
        k = max(1.0, w / 950)             # scale marks and text with the image size
        th = max(1, round(k))
        ring = lambda r: int(round(r * k))

        if truth is not None and self.show_truth.isChecked():
            for x, y in truth["star_xy"][truth["star_visible"]]:
                cv2.drawMarker(img, (int(round(x)), int(round(y))), (60, 60, 255), cv2.MARKER_CROSS, 22, 1, aa)
            for x, y in truth["hot_xy"]:
                cv2.rectangle(img, (int(x) - 3, int(y) - 3), (int(x) + 3, int(y) + 3), (0, 0, 160), 1)

        # Every detection (yellow ring)
        for s in dip.stars:
            cv2.circle(img, (int(round(s.x)), int(round(s.y))), ring(8), (0, 200, 255), th, aa)

        if sol is not None:
            cat = self.catalog
            inside = lambda p: -w < p[0] < 2 * w and -h < p[1] < 2 * h
            # Constellation stick figures that are in view
            for const in CONSTELLATIONS.values():
                xy, front = sol.project(const.vec)
                if not front.all() or not any(0 <= x < w and 0 <= y < h for x, y in xy):
                    continue
                for a, b in const.lines:
                    if inside(xy[a]) and inside(xy[b]):
                        cv2.line(img, tuple(np.round(xy[a]).astype(int)), tuple(np.round(xy[b]).astype(int)),
                                 (200, 160, 40), th, aa)
            # Catalogue stars the solution predicts (blue) vs those actually matched (green)
            bright = np.flatnonzero(cat.mag <= 5.0)
            xy, front = sol.project(cat.vec[bright])
            for (x, y), ok in zip(xy, front):
                if ok and 0 <= x < w and 0 <= y < h:
                    cv2.circle(img, (int(round(x)), int(round(y))), ring(4), (255, 140, 60), th, aa)
            if len(sol.triangle_xy):
                tri = np.round(sol.triangle_xy).astype(np.int32)
                cv2.polylines(img, [tri.reshape(-1, 1, 2)], True, (255, 0, 255), th, aa)
            det = dip.xy
            labelled = sorted(sol.matches, key=lambda m: cat.mag[m[0]])
            for n, (ci, di) in enumerate(labelled):
                x, y = det[di]
                cv2.circle(img, (int(round(x)), int(round(y))), ring(13), (90, 255, 90), 2 * th, aa)
                if self.show_labels.isChecked() and n < 25:
                    cv2.putText(img, cat.label(ci), (int(x) + ring(16), int(y) - ring(6)), font, 0.45 * k,
                                (190, 255, 190), th, aa)

        # Boresight reticle + roll indicator
        cx, cy = w // 2, h // 2
        grey = (140, 140, 140)
        cv2.circle(img, (cx, cy), ring(18), grey, th, aa)
        cv2.line(img, (cx - ring(30), cy), (cx - ring(20), cy), grey, th, aa)
        cv2.line(img, (cx + ring(20), cy), (cx + ring(30), cy), grey, th, aa)
        if sol is not None:
            a = math.radians(sol.roll_deg - 90)
            tip = (int(cx + ring(60) * math.cos(a)), int(cy + ring(60) * math.sin(a)))
            cv2.arrowedLine(img, (cx, cy), tip, (90, 255, 90), 2 * th, aa, tipLength=0.2)
            cv2.putText(img, f"ROLL {sol.roll_deg:+.2f} deg", (cx + ring(24), cy + ring(36)), font, 0.45 * k,
                        (90, 255, 90), th, aa)
            banner, color = f"{sol.mode}  {sol.constellation.upper()}", (90, 255, 90)
        else:
            banner, color = ("NOT SOLVED" if truth is None else "NO FIX  -  INERTIAL COAST"), (60, 160, 255)
        cv2.putText(img, banner, (ring(14), ring(28)), font, 0.7 * k, color, 2 * th, aa)
        if truth is not None:
            cv2.putText(img, f"FRAME {self.frame_no}", (w - 120, 28), font, 0.5, (160, 160, 160), 1, aa)
        return img

    def _update_views(self):
        """Refresh only the image tab that is visible (keeps the frame rate up)."""
        if self.last is None:
            return
        frame, truth, dip, sol, overlay = self.last
        imgs = {"overlay": overlay, "raw": dip.gray, "corrected": dip.corrected,
                "filtered": dip.filtered, "binary": dip.binary}
        key = self.tab_keys[self.view_tabs.currentIndex()]
        if key in self.views:
            self.views[key].set_image(imgs[key])
        elif key == "grid":
            for k, v in self.grid_views.items():
                v.set_image(imgs[k])
        elif key == "compare":
            self._update_comparison(frame, truth)

    def _update_comparison(self, frame, truth):
        lut = (255 * (np.arange(256) / 255.0) ** 0.55).astype(np.uint8)
        font, aa = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA
        for row, f in enumerate(NOISE_FILTERS):
            dip, sol = self._run_pipeline(frame, noise_filter=f)
            real = truth is None     # real image: no ground truth to grade detections against
            m = self._image_metrics(dip) if real else evaluate(truth, dip, sol)
            good = np.ones(len(dip.stars), bool) if real else classify_detections(truth, dip.xy)[1]

            img = cv2.cvtColor(cv2.LUT(dip.filtered, lut), cv2.COLOR_GRAY2BGR)
            for s, ok in zip(dip.stars, good):
                cv2.circle(img, (int(round(s.x)), int(round(s.y))), 12 if ok else 9,
                           ((0, 200, 255) if real else (90, 255, 90)) if ok else (60, 60, 255), 2, aa)
            verdict = f"LOCK {sol.constellation}" if sol else "NO FIX"
            cv2.rectangle(img, (0, 0), (img.shape[1], 44), (20, 20, 20), -1)
            cv2.putText(img, f"{f.upper()}   {verdict}", (12, 30), font, 0.9,
                        (90, 255, 90) if (sol and (real or m["correct_id"])) else (60, 160, 255), 2, aa)
            if not real:
                cv2.putText(img, f"false: {m['false_detections']}", (img.shape[1] - 170, 30), font, 0.8,
                            (60, 60, 255) if m["false_detections"] else (160, 160, 160), 2, aa)
            self.compare_views[f].set_image(img)

            dash = "—"
            cells = [f, dash if real else f"{m['catalog_detected']} / {m['catalog_visible']}",
                     dash if real else str(m["false_detections"]),
                     f"{m['centroid_rmse_px']:.3f}" if not math.isnan(m["centroid_rmse_px"]) else dash,
                     ("yes" if (real or m["correct_id"]) else "WRONG") if sol else "no",
                     f"{abs(m['roll_error_deg']) * 60:.2f}" if not math.isnan(m["roll_error_deg"]) else dash,
                     f"{dip.elapsed_ms:.1f}"]
            for col, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.compare_table.setItem(row, col, item)

    def _update_plots(self, dip):
        counts, edges = np.histogram(dip.filtered, bins=128, range=(0, 256))
        self.hist_curve.setData(edges, counts + 1)
        self.thr_line.setValue(dip.threshold)

        h = self.nav.history
        t = np.array(h["t"][-PLOT_WINDOW:])
        self.head_ins.setData(t, h["ins_heading_err"][-PLOT_WINDOW:])
        self.head_aid.setData(t, h["aided_heading_err"][-PLOT_WINDOW:])
        self.pos_ins.setData(t, h["ins_pos_err"][-PLOT_WINDOW:])
        self.pos_aid.setData(t, h["aided_pos_err"][-PLOT_WINDOW:])

        rows = self.rows[-PLOT_WINDOW:]
        tr = np.array([r["sim_time_s"] for r in rows])
        rmse = np.array([r["centroid_rmse_px"] for r in rows], dtype=float)
        roll = np.abs(np.array([r["roll_error_deg"] for r in rows], dtype=float)) * 60
        self.acc_rmse.setData(tr, rmse, connect="finite")
        self.acc_roll.setData(tr, roll, connect="finite")

    def _update_telemetry(self, truth, dip, sol, m):
        f = self.fields
        nav = self.nav
        real = truth is None
        if sol is not None:
            word = "SOLVED" if real else "LOCKED"
            self.status.setText(f"●  {word} — {sol.constellation.upper()}")
            self.status.setStyleSheet(f"background:#0f3a2c; color:{ACCENT}; font-size:15pt; font-weight:bold;"
                                      "border-radius:6px;")
        else:
            self.status.setText("○  NOT SOLVED" if real else "○  SEARCHING — INS COAST")
            self.status.setStyleSheet(f"background:#3a2a10; color:{WARN}; font-size:15pt; font-weight:bold;"
                                      "border-radius:6px;")
        self.clock.setText(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))

        dash = "—"
        f["Solver mode"].setText(sol.mode + (" (mirrored)" if sol.mirrored else "")
                                 + ("" if sol.projection == "pinhole" else " · stereo") if sol else dash)
        f["Constellation"].setText(sol.constellation if sol else dash)
        f["Stars matched"].setText(str(len(sol.matches)) if sol else dash)
        f["Fit residual"].setText(f"{sol.residual_px:.3f} px" if sol else dash)
        f["Field of view"].setText(f"{sol.fov_deg[0]:.2f}° x {sol.fov_deg[1]:.2f}°" if sol else dash)
        f["Roll measured"].setText(f"{sol.roll_deg:+.3f}°" if sol else dash)
        f["Roll true"].setText(dash if real else f"{truth['roll_deg']:+.3f}°")
        f["Roll error"].setText(f"{m['roll_error_deg'] * 60:+.2f} arcmin" if not math.isnan(m["roll_error_deg"]) else dash)
        f["Boresight RA"].setText(format_ra(sol.boresight_radec[0]) if sol else dash)
        f["Boresight Dec"].setText(format_dec(sol.boresight_radec[1]) if sol else dash)
        f["Pointing error"].setText(f"{m['boresight_error_deg'] * 3600:.1f} arcsec"
                                    if not math.isnan(m["boresight_error_deg"]) else dash)

        f["Detections"].setText(str(m["detections"]))
        f["Catalog found"].setText(dash if real or not m["catalog_visible"]
                                   else f"{m['catalog_detected']} / {m['catalog_visible']}")
        f["False detections"].setText(dash if real else str(m["false_detections"]))
        f["Centroid RMSE"].setText(f"{m['centroid_rmse_px']:.3f} px" if not math.isnan(m["centroid_rmse_px"]) else dash)
        f["Background"].setText(f"{dip.background:.1f} DN")
        f["Noise σ"].setText(f"{dip.noise_sigma:.2f} DN")
        f["Threshold"].setText(f"{dip.threshold:.1f} DN")
        f["DIP time"].setText(f"{dip.elapsed_ms:.1f} ms")
        f["Solve time"].setText(f"{sol.elapsed_ms:.1f} ms" if sol else dash)

        h = nav.history
        lat, lon = nav.to_latlon(nav.aided_pos)
        f["Sim time"].setText(f"{nav.t:.1f} s")
        f["True heading"].setText(f"{nav.true_heading:+.2f}°")
        f["Heading err INS"].setText(f"{h['ins_heading_err'][-1]:+.3f}°")
        f["Heading err aided"].setText(f"{h['aided_heading_err'][-1]:+.3f}°")
        f["Gyro bias est."].setText(f"{nav.bias_estimate:.4f} °/s")
        f["Pos err INS"].setText(f"{h['ins_pos_err'][-1]:,.1f} m")
        f["Pos err aided"].setText(f"{h['aided_pos_err'][-1]:,.1f} m")
        f["Position (aided)"].setText(f"{lat:.5f}N {lon:.5f}E")

    # ------------------------------------------------------------------ export
    def save_images(self):
        if self.last is None:
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Save stage images to")
        if not folder:
            return
        frame, truth, dip, sol, overlay = self.last
        if truth is None:
            stem = os.path.splitext(os.path.basename(self.image_paths[self.image_idx]))[0]
            prefix = os.path.join(folder, stem)
        else:
            prefix = os.path.join(folder, f"frame{self.frame_no:05d}")
        for name, img in (("1_raw", dip.gray), ("2_background_removed", dip.corrected),
                          ("3_filtered", dip.filtered), ("4_binary", dip.binary), ("5_overlay", overlay)):
            cv2.imencode(".png", img)[1].tofile(f"{prefix}_{name}.png")   # works with non-ASCII paths
        self.log(f"Saved 5 stage images to {folder}")

    # ------------------------------------------------------------------ benchmark
    def start_benchmark(self):
        if self.timer.isActive():
            self.toggle_run()   # live simulation would compete for the CPU
        sim_settings = {**self.sim_settings(), "obscured": False}
        self.bench_worker = BenchmarkWorker(self.bench_sweep.currentText(), self.bench_compare.currentText(),
                                            self.bench_frames.value(), sim_settings, self.dip_settings())
        self.bench_worker.progressed.connect(lambda d, t: (self.bench_progress.setMaximum(t),
                                                           self.bench_progress.setValue(d)))
        self.bench_worker.finished_with.connect(self._benchmark_done)
        self.bench_run.setEnabled(False)
        self.bench_stop.setEnabled(True)
        self.bench_progress.setValue(0)
        self.log(f"Benchmark started: {self.bench_sweep.currentText()} sweep, "
                 f"{self.bench_compare.currentText()}, {self.bench_frames.value()} frames/point")
        self.bench_worker.start()

    def stop_benchmark(self):
        if self.bench_worker:
            self.bench_worker.stop()

    def _benchmark_done(self, results):
        self.bench_run.setEnabled(True)
        self.bench_stop.setEnabled(False)
        self.bench_results = results
        self.bench_export.setEnabled(bool(results))
        self.log(f"Benchmark finished: {len(results)} result points")
        self.plot_benchmark(results)

    def plot_benchmark(self, results):
        colors = [ACCENT, WARN, "#7aa7ff", "#ff7ad9", "#e6e16a"]
        symbols = ["o", "s", "t", "d", "+"]
        variants = list(dict.fromkeys(r["variant"] for r in results))
        xlabel = results[0]["parameter"] if results else ""
        for key, (plot, scale) in self.bench_plots.items():
            plot.clear()
            plot.setLabel("bottom", xlabel)
            for i, v in enumerate(variants):
                rows = [r for r in results if r["variant"] == v]
                x = np.array([r["value"] for r in rows], dtype=float)
                y = np.array([r[key] for r in rows], dtype=float) * scale
                plot.plot(x, y, name=v, pen=pg.mkPen(colors[i % len(colors)], width=2),
                          symbol=symbols[i % len(symbols)], symbolSize=7,
                          symbolBrush=colors[i % len(colors)], connect="finite")

    def export_benchmark(self):
        if not self.bench_results:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export benchmark", "astranav_benchmark.csv",
                                                        "CSV (*.csv)")
        if path:
            save_csv(self.bench_results, path)
            self.log(f"Exported benchmark results to {os.path.basename(path)}")

    def export_csv(self):
        if not self.rows:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export metrics", "astranav_metrics.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        self.log(f"Exported {len(self.rows)} frames of metrics to {os.path.basename(path)}")


# --------------------------------------------------------------------------- theme

def apply_dark_theme(app):
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    c = QtGui.QColor
    pal.setColor(QtGui.QPalette.Window, c("#14171c"))
    pal.setColor(QtGui.QPalette.WindowText, c("#d6dbe2"))
    pal.setColor(QtGui.QPalette.Base, c("#0d1014"))
    pal.setColor(QtGui.QPalette.AlternateBase, c("#1a1e24"))
    pal.setColor(QtGui.QPalette.Text, c("#d6dbe2"))
    pal.setColor(QtGui.QPalette.Button, c("#1e232a"))
    pal.setColor(QtGui.QPalette.ButtonText, c("#d6dbe2"))
    pal.setColor(QtGui.QPalette.Highlight, c("#00a582"))
    pal.setColor(QtGui.QPalette.HighlightedText, c("#000000"))
    pal.setColor(QtGui.QPalette.ToolTipBase, c("#1e232a"))
    pal.setColor(QtGui.QPalette.ToolTipText, c("#d6dbe2"))
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, c("#5a626d"))
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, c("#5a626d"))
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, c("#5a626d"))
    app.setPalette(pal)
    app.setStyleSheet(f"""
        QGroupBox {{ border:1px solid #2a3038; border-radius:6px; margin-top:14px; padding:8px 8px 6px 8px; }}
        QGroupBox::title {{ subcontrol-origin: margin; left:10px; padding:0 4px; color:{ACCENT}; font-weight:bold; }}
        QPushButton {{ padding:6px 10px; border:1px solid #2f3640; border-radius:4px; background:#1e232a; }}
        QPushButton:hover {{ border-color:{ACCENT}; }}
        QTabBar::tab {{ padding:6px 14px; background:#1a1e24; border:1px solid #2a3038; border-bottom:none; }}
        QTabBar::tab:selected {{ background:#0d1014; color:{ACCENT}; }}
        QTabWidget::pane {{ border:1px solid #2a3038; }}
        QSlider::groove:horizontal {{ height:4px; background:#2a3038; border-radius:2px; }}
        QSlider::handle:horizontal {{ background:{ACCENT}; width:12px; margin:-5px 0; border-radius:6px; }}
        QSlider::sub-page:horizontal {{ background:#00785f; border-radius:2px; }}
    """)


def main():
    app = QtWidgets.QApplication(sys.argv)
    apply_dark_theme(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
