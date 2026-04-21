"""
gui/dashboard.py
================
Main Tkinter GUI Dashboard for the Traffic Monitoring System.

Layout:
  ┌─────────────────────────────────────────────┐
  │  Header (title + webcam status)             │
  ├──────────────────────┬──────────────────────┤
  │  Video Canvas        │  Analytics Panel     │
  │                      │  - Counters          │
  │                      │  - Vehicle Types     │
  │                      │  - Plates list       │
  │                      │  - Accident Risk     │
  ├──────────────────────┴──────────────────────┤
  │  Controls (buttons)                         │
  └─────────────────────────────────────────────┘
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import cv2
import numpy as np
from PIL import Image, ImageTk

from detection.yolo_model import VehicleDetector
from detection.tracker import VehicleTracker
from analytics.counter import VehicleCounter
from analytics.accident import AccidentPredictor
from analytics.heatmap import HeatmapGenerator
from utils.helpers import (
    draw_boxes, resize_frame, frame_to_photoimage,
    get_timestamp, generate_sample_video
)
from utils.database import init_db, log_stats
from utils.pdf_generator import generate_pdf_report

# ── Colour palette (dark theme) ────────────────────────────────────────────
BG_DARK   = "#0D1117"
BG_PANEL  = "#161B22"
BG_CARD   = "#1C2128"
ACCENT    = "#1F6FEB"
ACCENT2   = "#3FB950"
TEXT_MAIN = "#E6EDF3"
TEXT_DIM  = "#8B949E"
RED       = "#F85149"
ORANGE    = "#D29922"
GREEN     = "#3FB950"
BORDER    = "#30363D"

FONT_TITLE = ("Segoe UI", 18, "bold")
FONT_HEAD  = ("Segoe UI", 11, "bold")
FONT_BODY  = ("Segoe UI", 10)
FONT_MONO  = ("Courier New", 9)
FONT_BIG   = ("Segoe UI", 22, "bold")


class TrafficDashboard:
    """Main application window and control loop."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Traffic Monitoring Dashboard")
        self.root.configure(bg=BG_DARK)
        self.root.geometry("1280x780")
        self.root.minsize(1100, 700)

        # ── State ──────────────────────────────────────────────────────────
        self.video_source   = None   # cv2.VideoCapture or None
        self.is_webcam      = False
        self.analysis_on    = False
        self.running        = False  # main loop flag
        self.video_thread   = None

        # ── Sub-systems ────────────────────────────────────────────────────
        self.detector   = VehicleDetector()
        self.tracker    = VehicleTracker()
        self.counter    = VehicleCounter()
        self.accident   = AccidentPredictor()
        self.heatmap_gen = HeatmapGenerator(width=640, height=360)

        # Initialize SQLite DB
        init_db()

        # ── Build UI ───────────────────────────────────────────────────────
        self._build_ui()
        self._update_status(connected=False)

        # Graceful close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ══════════════════════════════════════════════════════════════════════
    # UI CONSTRUCTION
    # ══════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        """Assemble all UI sections."""
        self._build_header()
        self._build_main_area()
        self._build_controls()

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=BG_PANEL, height=56)
        hdr.pack(fill=tk.X, side=tk.TOP)
        hdr.pack_propagate(False)

        # Title
        tk.Label(hdr, text="🚦  Traffic Monitoring Dashboard",
                 font=FONT_TITLE, bg=BG_PANEL, fg=TEXT_MAIN).pack(side=tk.LEFT, padx=20, pady=12)

        # Webcam status badge (right-aligned)
        self.status_frame = tk.Frame(hdr, bg=BG_PANEL)
        self.status_frame.pack(side=tk.RIGHT, padx=20)
        self.status_dot   = tk.Label(self.status_frame, text="●", font=("Segoe UI", 14), bg=BG_PANEL)
        self.status_label = tk.Label(self.status_frame, font=FONT_BODY, bg=BG_PANEL, fg=TEXT_DIM)
        self.status_dot.pack(side=tk.LEFT)
        self.status_label.pack(side=tk.LEFT, padx=(4, 0))

        # Timestamp
        self.clock_label = tk.Label(hdr, font=FONT_MONO, bg=BG_PANEL, fg=TEXT_DIM)
        self.clock_label.pack(side=tk.RIGHT, padx=16)
        self._tick_clock()

        # Separator
        sep = tk.Frame(self.root, bg=BORDER, height=1)
        sep.pack(fill=tk.X)

    def _build_main_area(self):
        """Two-column layout: video left, analytics right."""
        body = tk.Frame(self.root, bg=BG_DARK)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(8, 4))

        # ── Left: video canvas ─────────────────────────────────────────
        left = tk.Frame(body, bg=BG_DARK)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        canvas_frame = tk.Frame(left, bg=BORDER, bd=1, relief=tk.FLAT)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Label(canvas_frame, bg="#000000",
                               text="No feed — upload a video or start webcam",
                               fg=TEXT_DIM, font=FONT_BODY)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Heatmap toggle button below video
        self.show_heatmap_var = tk.BooleanVar(value=False)
        tk.Checkbutton(left, text="Show Heatmap Overlay",
                       variable=self.show_heatmap_var,
                       bg=BG_DARK, fg=TEXT_DIM, selectcolor=BG_CARD,
                       activebackground=BG_DARK, activeforeground=TEXT_MAIN,
                       font=FONT_BODY).pack(anchor=tk.W, pady=(4, 0))

        # ── Right: analytics panel ─────────────────────────────────────
        right = tk.Frame(body, bg=BG_DARK, width=320)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        self._build_analytics_panel(right)

    def _build_analytics_panel(self, parent):
        """Counters, type breakdown, plate list, accident risk."""

        # ── Counters card ──────────────────────────────────────────────
        card1 = self._card(parent, "📊  Live Analytics")

        row1 = tk.Frame(card1, bg=BG_CARD)
        row1.pack(fill=tk.X, pady=(0, 6))
        self.lbl_total = self._metric(row1, "Total\nVehicles", "0", ACCENT)
        self.lbl_vpm   = self._metric(row1, "Vehicles /\nMinute", "0", ACCENT2)

        # ── Vehicle types card ─────────────────────────────────────────
        card2 = self._card(parent, "🚗  Vehicle Types")

        types_frame = tk.Frame(card2, bg=BG_CARD)
        types_frame.pack(fill=tk.X)

        self.type_bars = {}
        self.type_labels = {}
        colours = {"Car": "#38BDF8", "Bike": "#A78BFA",
                   "Truck": "#FBBF24", "Bus": "#34D399"}
        for vtype, color in colours.items():
            row = tk.Frame(types_frame, bg=BG_CARD)
            row.pack(fill=tk.X, pady=2)

            tk.Label(row, text=vtype, width=5, anchor=tk.W,
                     font=FONT_BODY, bg=BG_CARD, fg=TEXT_DIM).pack(side=tk.LEFT)

            bar_bg = tk.Frame(row, bg=BORDER, height=8)
            bar_bg.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 6))
            bar_fg = tk.Frame(bar_bg, bg=color, height=8, width=0)
            bar_fg.place(x=0, y=0, relheight=1)
            self.type_bars[vtype] = (bar_bg, bar_fg, color)

            lbl = tk.Label(row, text="0", width=4, anchor=tk.E,
                           font=FONT_MONO, bg=BG_CARD, fg=color)
            lbl.pack(side=tk.RIGHT)
            self.type_labels[vtype] = lbl

        # ── Number plate list ──────────────────────────────────────────
        card3 = self._card(parent, "🔍  Detected Plates")

        plate_scroll = tk.Frame(card3, bg=BG_CARD)
        plate_scroll.pack(fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(plate_scroll)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.plate_list = tk.Listbox(
            plate_scroll, bg=BG_DARK, fg=ACCENT2,
            font=FONT_MONO, bd=0, highlightthickness=0,
            selectbackground=BORDER, activestyle="none",
            yscrollcommand=scrollbar.set, height=6
        )
        self.plate_list.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.plate_list.yview)

        # ── Accident risk card ─────────────────────────────────────────
        card4 = self._card(parent, "⚠️  Accident Risk Prediction")

        self.risk_pct_label = tk.Label(card4, text="0.0%", font=FONT_BIG,
                                       bg=BG_CARD, fg=GREEN)
        self.risk_pct_label.pack(pady=(0, 6))

        self.risk_bar = ttk.Progressbar(card4, orient=tk.HORIZONTAL,
                                        length=280, mode="determinate",
                                        maximum=100)
        self.risk_bar.pack(fill=tk.X)

        self.risk_status = tk.Label(card4, text="Status: LOW",
                                    font=FONT_BODY, bg=BG_CARD, fg=GREEN)
        self.risk_status.pack(pady=(4, 0))

        # Style the progressbar
        style = ttk.Style()
        style.theme_use("default")
        style.configure("green.Horizontal.TProgressbar",
                        troughcolor=BORDER, background=GREEN)
        self.risk_bar.config(style="green.Horizontal.TProgressbar")

    def _build_controls(self):
        """Bottom control bar with all buttons."""
        bar = tk.Frame(self.root, bg=BG_PANEL, height=52)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        bar.pack_propagate(False)

        sep = tk.Frame(self.root, bg=BORDER, height=1)
        sep.pack(fill=tk.X, side=tk.BOTTOM)

        def btn(parent, text, cmd, color=ACCENT):
            b = tk.Button(parent, text=text, command=cmd,
                          bg=color, fg="#FFFFFF",
                          font=("Segoe UI", 10, "bold"),
                          relief=tk.FLAT, bd=0,
                          padx=16, pady=6,
                          activebackground=BG_CARD,
                          activeforeground=TEXT_MAIN,
                          cursor="hand2")
            b.pack(side=tk.LEFT, padx=6, pady=10)
            return b

        # Source controls
        tk.Label(bar, text="SOURCE:", font=FONT_BODY,
                 bg=BG_PANEL, fg=TEXT_DIM).pack(side=tk.LEFT, padx=(14, 0), pady=10)
        btn(bar, "📂  Upload Video",  self._upload_video, "#1F6FEB")
        btn(bar, "🎥  Start Webcam",  self._start_webcam, "#388BFD")
        btn(bar, "⏹  Stop Webcam",   self._stop_webcam,  "#6E7681")

        # Separator
        tk.Frame(bar, bg=BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        # Analysis controls
        tk.Label(bar, text="ANALYSIS:", font=FONT_BODY,
                 bg=BG_PANEL, fg=TEXT_DIM).pack(side=tk.LEFT, padx=(0, 0), pady=10)
        self.btn_start = btn(bar, "▶  Start Analysis", self._start_analysis, GREEN)
        self.btn_stop  = btn(bar, "■  Stop Analysis",  self._stop_analysis,  RED)

        # PDF Export
        tk.Frame(bar, bg=BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
        btn(bar, "📥 Download PDF", self._download_pdf, "#A78BFA")

        # Reset
        tk.Frame(bar, bg=BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
        btn(bar, "↺  Reset",  self._reset,  "#6E7681")

        # FPS label (right-aligned)
        self.fps_label = tk.Label(bar, text="FPS: —", font=FONT_MONO,
                                  bg=BG_PANEL, fg=TEXT_DIM)
        self.fps_label.pack(side=tk.RIGHT, padx=14)

    # ══════════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _card(self, parent, title):
        """Labelled card widget; returns inner frame."""
        frame = tk.LabelFrame(parent, text=f"  {title}  ",
                              bg=BG_CARD, fg=TEXT_DIM,
                              font=("Segoe UI", 9, "bold"),
                              bd=1, relief=tk.GROOVE,
                              labelanchor=tk.NW)
        frame.pack(fill=tk.X, pady=(0, 8), padx=2)
        inner = tk.Frame(frame, bg=BG_CARD, pady=6, padx=8)
        inner.pack(fill=tk.BOTH, expand=True)
        return inner

    def _metric(self, parent, label, value, color):
        """Big number + small label metric widget."""
        frame = tk.Frame(parent, bg=BG_CARD)
        frame.pack(side=tk.LEFT, expand=True, fill=tk.X)
        val_lbl = tk.Label(frame, text=value, font=FONT_BIG,
                           bg=BG_CARD, fg=color)
        val_lbl.pack()
        tk.Label(frame, text=label, font=("Segoe UI", 8),
                 bg=BG_CARD, fg=TEXT_DIM).pack()
        return val_lbl

    def _update_status(self, connected: bool):
        if connected:
            self.status_dot.config(fg=GREEN)
            self.status_label.config(text="Webcam Connected", fg=GREEN)
        else:
            self.status_dot.config(fg=RED)
            self.status_label.config(text="Webcam Disconnected", fg=RED)

    def _tick_clock(self):
        self.clock_label.config(text=get_timestamp())
        self.root.after(1000, self._tick_clock)

    # ══════════════════════════════════════════════════════════════════════
    # CONTROL HANDLERS
    # ══════════════════════════════════════════════════════════════════════

    def _upload_video(self):
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All", "*.*")]
        )
        if not path:
            return
        self._stop_feed()
        self.video_source = cv2.VideoCapture(path)
        self.is_webcam = False
        self._update_status(False)
        self._log(f"Video loaded: {path.split('/')[-1]}")

    def _start_webcam(self):
        self._stop_feed()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Webcam Error",
                                 "Could not open webcam.\nPlease check your camera.")
            return
        self.video_source = cap
        self.is_webcam = True
        self._update_status(True)
        self._log("Webcam started.")

    def _stop_webcam(self):
        self._stop_feed()
        self._update_status(False)
        self._log("Webcam stopped.")

    def _start_analysis(self):
        if self.video_source is None:
            messagebox.showwarning("No Source", "Please upload a video or start the webcam first.")
            return
        if self.running:
            return
        self.analysis_on = True
        self.running = True
        self.counter.reset()
        self.tracker.reset()
        self.heatmap_gen.reset()
        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()
        self._log("Analysis started.")

    def _stop_analysis(self):
        self.analysis_on = False
        self.running = False
        self._log("Analysis stopped.")

    def _stop_feed(self):
        self.running = False
        self.analysis_on = False
        if self.video_source:
            self.video_source.release()
            self.video_source = None
        self.canvas.config(image="", text="No feed")
        self.canvas.image = None

    def _reset(self):
        self._stop_feed()
        self._update_status(False)
        self.counter.reset()
        self.tracker.reset()
        self.heatmap_gen.reset()
        self.plate_list.delete(0, tk.END)
        self._refresh_analytics(reset=True)
        self._log("Dashboard reset.")

    def _download_pdf(self):
        try:
            stats = self.counter.get_stats() if hasattr(self, 'counter') else None
            risk = float(self.risk_bar["value"]) if hasattr(self, "risk_bar") else 0.0
            plates = list(self.plate_list.get(0, tk.END)) if hasattr(self, "plate_list") else []
        except Exception:
            stats, risk, plates = None, 0.0, []
            
        path = filedialog.asksaveasfilename(
            title="Save Analytics Report",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All", "*.*")],
            initialfile="Traffic_Report.pdf"
        )
        if not path:
            return
        
        success = generate_pdf_report(path, current_stats=stats, current_risk=risk, plates_list=plates)
        if success:
            messagebox.showinfo("Success", f"PDF report successfully saved to:\n{path}")
            self._log("PDF Report generated.")
        else:
            messagebox.showerror("Error", "Failed to generate PDF. Check terminal for details.")

    # ══════════════════════════════════════════════════════════════════════
    # VIDEO LOOP  (runs in background thread)
    # ══════════════════════════════════════════════════════════════════════

    def _video_loop(self):
        prev_time = time.time()
        frame_idx  = 0

        while self.running and self.video_source and self.video_source.isOpened():
            ret, frame = self.video_source.read()
            if not ret:
                # Loop video file
                if not self.is_webcam:
                    self.video_source.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            frame_idx += 1
            display = resize_frame(frame, width=680)

            if self.analysis_on:
                # ── Detection ─────────────────────────────────────────
                detections = self.detector.detect(display)

                # ── Tracking ──────────────────────────────────────────
                tracks = self.tracker.update(detections, display)

                # ── Counting ──────────────────────────────────────────
                self.counter.update(tracks)

                # ── Heatmap ───────────────────────────────────────────
                for t in tracks:
                    cx = int((t["x1"] + t["x2"]) / 2)
                    cy = int((t["y1"] + t["y2"]) / 2)
                    self.heatmap_gen.add_point(cx, cy)

                # ── Accident prediction ───────────────────────────────
                risk = self.accident.predict(tracks)

                # ── Plate detection (every 10 frames) ─────────────────
                if frame_idx % 10 == 0 and tracks:
                    plate = self.detector.extract_plate(display, tracks)
                    if plate:
                        self.root.after(0, self._add_plate, plate)

                # ── DB Logging (every 30 frames) ──────────────────────
                if frame_idx % 30 == 0:
                    log_stats(self.counter.get_stats(), risk)

                # ── Draw boxes on frame ────────────────────────────────
                display = draw_boxes(display, tracks)

                # ── Heatmap overlay ────────────────────────────────────
                if self.show_heatmap_var.get():
                    hm = self.heatmap_gen.get_overlay(display.shape)
                    display = cv2.addWeighted(display, 0.7, hm, 0.5, 0)

                # ── Update analytics in main thread ───────────────────
                self.root.after(0, self._refresh_analytics,
                                False, risk, self.counter.get_stats())

            # ── FPS calculation ────────────────────────────────────────
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            self.root.after(0, self.fps_label.config,
                            {"text": f"FPS: {fps:.1f}"})

            # ── Push frame to canvas ───────────────────────────────────
            photo = frame_to_photoimage(display)
            self.root.after(0, self._show_frame, photo)

            # Throttle slightly to reduce CPU on non-GPU machines
            time.sleep(0.01)

        self._log("Feed ended.")

    def _show_frame(self, photo):
        self.canvas.config(image=photo, text="")
        self.canvas.image = photo  # prevent GC

    # ══════════════════════════════════════════════════════════════════════
    # ANALYTICS REFRESH
    # ══════════════════════════════════════════════════════════════════════

    def _refresh_analytics(self, reset=False, risk=0.0, stats=None):
        if reset or stats is None:
            stats = {"total": 0, "vpm": 0,
                     "types": {"Car": 0, "Bike": 0, "Truck": 0, "Bus": 0}}
            risk = 0.0

        # Counters
        self.lbl_total.config(text=str(stats["total"]))
        self.lbl_vpm.config(text=str(stats["vpm"]))

        # Type bars
        total = max(sum(stats["types"].values()), 1)
        for vtype, count in stats["types"].items():
            bar_bg, bar_fg, color = self.type_bars[vtype]
            self.type_labels[vtype].config(text=str(count))
            bar_bg.update_idletasks()
            bar_w = bar_bg.winfo_width()
            fill  = int((count / total) * bar_w)
            bar_fg.place(x=0, y=0, width=fill, relheight=1)

        # Risk
        risk = min(max(risk, 0), 100)
        self.risk_bar["value"] = risk
        self.risk_pct_label.config(text=f"{risk:.1f}%")

        if risk < 30:
            color, status = GREEN, "LOW"
        elif risk < 60:
            color, status = ACCENT2, "MODERATE"
        elif risk < 80:
            color, status = ORANGE, "HIGH"
        else:
            color, status = RED, "CRITICAL ⚠"

        self.risk_pct_label.config(fg=color)
        self.risk_status.config(text=f"Status: {status}", fg=color)

        # Update progressbar colour dynamically
        style = ttk.Style()
        style.configure("risk.Horizontal.TProgressbar",
                        troughcolor=BORDER, background=color)
        self.risk_bar.config(style="risk.Horizontal.TProgressbar")

    def _add_plate(self, plate_text: str):
        ts = get_timestamp()
        self.plate_list.insert(0, f"  {ts}  │  {plate_text}")
        # Keep list to 50 items
        if self.plate_list.size() > 50:
            self.plate_list.delete(50, tk.END)

    def _log(self, msg: str):
        print(f"[{get_timestamp()}] {msg}")

    # ══════════════════════════════════════════════════════════════════════
    # APP LIFECYCLE
    # ══════════════════════════════════════════════════════════════════════

    def run(self):
        self.root.mainloop()

    def _on_close(self):
        self.running = False
        if self.video_source:
            self.video_source.release()
        self.root.destroy()
