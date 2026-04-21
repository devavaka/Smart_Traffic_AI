"""
detection/tracker.py
====================
Lightweight centroid-based vehicle tracker.

Algorithm:
  1. For each new detection, compute IoU against all existing tracks.
  2. If best IoU > threshold → same vehicle (update track).
  3. Otherwise → new vehicle (assign new ID).
  4. Tracks not updated for MAX_MISSING frames are deleted.

This avoids requiring DeepSORT / SORT as external dependencies while
still producing stable track IDs suitable for per-vehicle counting.
"""

import numpy as np
from collections import defaultdict


IOU_THRESHOLD = 0.30   # minimum IoU to associate detection with track
MAX_MISSING   = 15     # frames before a track is dropped


class Track:
    """Represents one tracked vehicle across frames."""

    def __init__(self, track_id: int, detection: dict):
        self.track_id   = track_id
        self.label      = detection["label"]
        self.confidence = detection["confidence"]
        self.x1 = detection["x1"]
        self.y1 = detection["y1"]
        self.x2 = detection["x2"]
        self.y2 = detection["y2"]
        self.missing    = 0          # frames since last matched
        self.age        = 0          # total frames alive
        self.history    = []         # (cx, cy) centroid history for speed

    def update(self, detection: dict):
        """Refresh position from a matched detection."""
        cx_prev = (self.x1 + self.x2) // 2
        cy_prev = (self.y1 + self.y2) // 2

        self.x1 = detection["x1"]
        self.y1 = detection["y1"]
        self.x2 = detection["x2"]
        self.y2 = detection["y2"]
        self.confidence = detection["confidence"]
        self.missing = 0
        self.age += 1

        cx = (self.x1 + self.x2) // 2
        cy = (self.y1 + self.y2) // 2
        self.history.append((cx, cy))
        if len(self.history) > 30:
            self.history.pop(0)

    @property
    def speed(self) -> float:
        """Pixel displacement per frame averaged over recent history."""
        if len(self.history) < 2:
            return 0.0
        deltas = [
            np.hypot(self.history[i][0] - self.history[i-1][0],
                     self.history[i][1] - self.history[i-1][1])
            for i in range(1, len(self.history))
        ]
        return float(np.mean(deltas))

    def to_dict(self) -> dict:
        return {
            "id": self.track_id,
            "x1": self.x1, "y1": self.y1,
            "x2": self.x2, "y2": self.y2,
            "label": self.label,
            "confidence": self.confidence,
            "speed": round(self.speed, 2),
            "age": self.age,
        }


class VehicleTracker:
    """Manages a pool of Track objects."""

    def __init__(self):
        self.tracks: list[Track] = []
        self._next_id = 1

    # ──────────────────────────────────────────────────────────────────────

    def update(self, detections: list[dict],
               frame: np.ndarray) -> list[dict]:
        """
        Match detections to existing tracks and return updated track list.

        Parameters
        ----------
        detections : list of detection dicts from VehicleDetector.detect()
        frame      : current BGR frame (unused here, reserved for deep features)

        Returns
        -------
        List of track dicts (same schema as Track.to_dict()).
        """
        if not detections:
            # Age all tracks; remove stale ones
            self._age_tracks()
            return [t.to_dict() for t in self.tracks]

        # Build IoU cost matrix: rows=tracks, cols=detections
        matched_tracks = set()
        matched_dets   = set()

        if self.tracks:
            iou_matrix = self._build_iou_matrix(detections)
            # Greedy matching (largest IoU first)
            pairs = np.dstack(np.unravel_index(
                np.argsort(-iou_matrix, axis=None), iou_matrix.shape
            ))[0]
            for ti, di in pairs:
                if ti in matched_tracks or di in matched_dets:
                    continue
                if iou_matrix[ti, di] < IOU_THRESHOLD:
                    break
                self.tracks[ti].update(detections[di])
                matched_tracks.add(ti)
                matched_dets.add(di)

        # Unmatched tracks → age up
        for ti, track in enumerate(self.tracks):
            if ti not in matched_tracks:
                track.missing += 1

        # Unmatched detections → new tracks
        for di, det in enumerate(detections):
            if di not in matched_dets:
                new_track = Track(self._next_id, det)
                new_track.age = 1
                new_track.history = [((det["x1"]+det["x2"])//2,
                                      (det["y1"]+det["y2"])//2)]
                self.tracks.append(new_track)
                self._next_id += 1

        # Drop stale tracks
        self.tracks = [t for t in self.tracks if t.missing < MAX_MISSING]

        return [t.to_dict() for t in self.tracks]

    def reset(self):
        self.tracks  = []
        self._next_id = 1

    # ──────────────────────────────────────────────────────────────────────

    def _build_iou_matrix(self, detections: list[dict]) -> np.ndarray:
        matrix = np.zeros((len(self.tracks), len(detections)), dtype=float)
        for ti, track in enumerate(self.tracks):
            for di, det in enumerate(detections):
                matrix[ti, di] = _iou(
                    (track.x1, track.y1, track.x2, track.y2),
                    (det["x1"], det["y1"], det["x2"], det["y2"])
                )
        return matrix

    def _age_tracks(self):
        for t in self.tracks:
            t.missing += 1
        self.tracks = [t for t in self.tracks if t.missing < MAX_MISSING]


# ── Utility ──────────────────────────────────────────────────────────────────

def _iou(boxA: tuple, boxB: tuple) -> float:
    """Compute Intersection-over-Union for two (x1,y1,x2,y2) boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / float(areaA + areaB - inter + 1e-6)
