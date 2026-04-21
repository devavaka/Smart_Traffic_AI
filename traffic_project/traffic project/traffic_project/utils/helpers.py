"""
utils/helpers.py
================
General-purpose helper functions used across the project.

Contents
--------
- resize_frame          : keep aspect ratio resize
- frame_to_photoimage   : OpenCV BGR → Tkinter PhotoImage
- draw_boxes            : render track bounding boxes + labels on frame
- birds_eye_view        : perspective transform to top-down view
- get_timestamp         : formatted current time string
- generate_sample_video : create a synthetic traffic demo video
"""

import cv2
import numpy as np
import time
import os
import random
from PIL import Image, ImageTk


# ── Colour palette for vehicle-type labels ─────────────────────────────────
LABEL_COLOURS = {
    "Car":   (56, 189, 248),    # sky blue
    "Bike":  (167, 139, 250),   # violet
    "Truck": (251, 191, 36),    # amber
    "Bus":   (52,  211, 153),   # emerald
}
DEFAULT_COLOUR = (200, 200, 200)

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
THICKNESS  = 1


# ══════════════════════════════════════════════════════════════════════════════
# FRAME UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def resize_frame(frame: np.ndarray, width: int = 680) -> np.ndarray:
    """
    Resize frame to a target width while preserving aspect ratio.
    Avoids distortion for any input resolution.
    """
    h, w = frame.shape[:2]
    if w == width:
        return frame
    scale  = width / w
    new_h  = int(h * scale)
    return cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_LINEAR)


def frame_to_photoimage(frame: np.ndarray) -> ImageTk.PhotoImage:
    """Convert a BGR OpenCV frame to a Tkinter-compatible PhotoImage."""
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    return ImageTk.PhotoImage(image)


# ══════════════════════════════════════════════════════════════════════════════
# BOUNDING BOX DRAWING
# ══════════════════════════════════════════════════════════════════════════════

def draw_boxes(frame: np.ndarray, tracks: list[dict]) -> np.ndarray:
    """
    Draw bounding boxes with track ID, label and confidence on frame.

    Parameters
    ----------
    frame  : BGR ndarray
    tracks : list of track dicts (from VehicleTracker.update)

    Returns
    -------
    Annotated BGR ndarray.
    """
    out = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = t["x1"], t["y1"], t["x2"], t["y2"]
        label  = t.get("label", "Vehicle")
        conf   = t.get("confidence", 0.0)
        tid    = t.get("id", 0)
        speed  = t.get("speed", 0.0)

        colour = LABEL_COLOURS.get(label, DEFAULT_COLOUR)

        # Main bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

        # Corner accents
        corner_len = 10
        for cx, cy, dx, dy in [
            (x1, y1,  1,  1), (x2, y1, -1,  1),
            (x1, y2,  1, -1), (x2, y2, -1, -1)
        ]:
            cv2.line(out, (cx, cy), (cx + dx*corner_len, cy), colour, 2)
            cv2.line(out, (cx, cy), (cx, cy + dy*corner_len), colour, 2)

        # Label background + text
        text = f"#{tid} {label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
        lbl_y1 = max(y1 - th - 6, 0)
        cv2.rectangle(out, (x1, lbl_y1), (x1 + tw + 6, y1), colour, -1)
        cv2.putText(out, text, (x1 + 3, y1 - 3),
                    FONT, FONT_SCALE, (0, 0, 0), THICKNESS, cv2.LINE_AA)

        # Speed indicator (small, below box)
        spd_text = f"{speed:.1f}px/f"
        cv2.putText(out, spd_text, (x1, y2 + 14),
                    FONT, 0.40, colour, 1, cv2.LINE_AA)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# BIRD'S EYE VIEW  (perspective transform)
# ══════════════════════════════════════════════════════════════════════════════

def birds_eye_view(frame: np.ndarray,
                   src_points: np.ndarray | None = None) -> np.ndarray:
    """
    Apply a perspective transform to produce a top-down (bird's eye) view.

    Parameters
    ----------
    frame      : BGR input frame
    src_points : four (x, y) source points defining the road trapezoid.
                 If None, a sensible default is computed from frame size.

    Returns
    -------
    Warped BGR frame (same size as input).
    """
    h, w = frame.shape[:2]

    if src_points is None:
        # Default: trapezoidal ROI roughly covering a typical road perspective
        src_points = np.float32([
            [w * 0.10, h * 0.95],   # bottom-left
            [w * 0.90, h * 0.95],   # bottom-right
            [w * 0.60, h * 0.55],   # top-right
            [w * 0.40, h * 0.55],   # top-left
        ])

    dst_points = np.float32([
        [w * 0.15, h],        # bottom-left
        [w * 0.85, h],        # bottom-right
        [w * 0.85, 0],        # top-right
        [w * 0.15, 0],        # top-left
    ])

    M   = cv2.getPerspectiveTransform(src_points, dst_points)
    bev = cv2.warpPerspective(frame, M, (w, h))
    return bev


# ══════════════════════════════════════════════════════════════════════════════
# MISC
# ══════════════════════════════════════════════════════════════════════════════

def get_timestamp() -> str:
    """Return current local time as HH:MM:SS string."""
    return time.strftime("%H:%M:%S")


def generate_sample_video(output_path: str = "assets/sample_video.mp4",
                           duration_s: int = 30, fps: int = 20) -> str:
    """
    Create a synthetic traffic demo video with moving coloured rectangles.
    Used as the bundled sample video when no real footage is available.

    Parameters
    ----------
    output_path : where to write the .mp4 file
    duration_s  : video length in seconds
    fps         : frames per second

    Returns
    -------
    Path to the created file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    W, H   = 854, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    # Synthetic vehicles
    vehicles = []
    colours  = [(56,189,248),(167,139,250),(251,191,36),(52,211,153)]
    labels   = ["Car","Bike","Truck","Bus"]
    for i in range(8):
        bw = random.randint(60, 120)
        bh = random.randint(35, 60)
        vehicles.append({
            "x": random.randint(0, W-bw), "y": random.randint(H//4, H-bh),
            "w": bw, "h": bh,
            "dx": random.choice([-1,1]) * random.randint(2,5),
            "dy": random.choice([-1,1]) * random.randint(1,2),
            "colour": colours[i % len(colours)],
            "label": labels[i % len(labels)],
        })

    n_frames = duration_s * fps
    for f in range(n_frames):
        # Road background
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[:] = (30, 30, 30)  # dark grey road

        # Road markings
        for lx in range(0, W, 60):
            cv2.line(frame, (lx, H//2), (lx+35, H//2), (200,200,200), 2)

        for v in vehicles:
            v["x"] = max(0, min(W - v["w"], v["x"] + v["dx"]))
            v["y"] = max(H//4, min(H - v["h"], v["y"] + v["dy"]))
            if v["x"] <= 0 or v["x"] + v["w"] >= W:
                v["dx"] *= -1
            if v["y"] <= H//4 or v["y"] + v["h"] >= H:
                v["dy"] *= -1

            x, y, bw, bh = v["x"], v["y"], v["w"], v["h"]
            c = v["colour"]
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), c, -1)
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (255,255,255), 1)
            cv2.putText(frame, v["label"], (x+3, y+bh-5),
                        FONT, 0.4, (0,0,0), 1, cv2.LINE_AA)

        # Frame counter overlay
        cv2.putText(frame, f"DEMO  frame {f+1}/{n_frames}", (10, 24),
                    FONT, 0.55, (120,120,120), 1, cv2.LINE_AA)
        writer.write(frame)

    writer.release()
    print(f"[INFO] Sample video created: {output_path}")
    return output_path
