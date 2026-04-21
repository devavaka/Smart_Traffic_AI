"""
analytics/counter.py
====================
Counts vehicles and computes Vehicles Per Minute (VPM).

Strategy:
- Maintain a set of seen track IDs.  Each new ID = +1 total.
- A sliding window of timestamp→count pairs drives the VPM calculation.
"""

import time
from collections import deque


class VehicleCounter:
    """
    Tracks cumulative vehicle count and real-time VPM.

    Usage::

        counter = VehicleCounter()
        counter.update(tracks)   # call every frame
        stats = counter.get_stats()
        # → {"total": 42, "vpm": 12, "types": {"Car": 30, ...}}
    """

    VEHICLE_TYPES = ("Car", "Bike", "Truck", "Bus")

    def __init__(self):
        self._seen_ids: set[int]       = set()
        self._type_counts: dict        = {t: 0 for t in self.VEHICLE_TYPES}
        self._window: deque            = deque()  # (timestamp, count_added)
        self._window_seconds: int      = 60       # VPM window

    # ──────────────────────────────────────────────────────────────────────

    def update(self, tracks: list[dict]) -> None:
        """Process one frame's worth of tracks."""
        added = 0
        for t in tracks:
            tid = t["id"]
            if tid not in self._seen_ids:
                self._seen_ids.add(tid)
                label = t.get("label", "Car")
                if label in self._type_counts:
                    self._type_counts[label] += 1
                added += 1

        if added:
            now = time.time()
            self._window.append((now, added))

        # Purge old entries outside the VPM window
        cutoff = time.time() - self._window_seconds
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()

    def get_stats(self) -> dict:
        """Return current analytics snapshot."""
        vpm = sum(c for _, c in self._window)
        return {
            "total": len(self._seen_ids),
            "vpm"  : vpm,
            "types": dict(self._type_counts),
        }

    def reset(self) -> None:
        self._seen_ids.clear()
        self._type_counts = {t: 0 for t in self.VEHICLE_TYPES}
        self._window.clear()
