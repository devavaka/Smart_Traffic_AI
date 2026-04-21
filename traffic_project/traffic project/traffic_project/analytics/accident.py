"""
analytics/accident.py
=====================
Rule-based accident risk predictor.

Risk factors considered
-----------------------
1. **Speed risk** – average pixel displacement per frame.
   Fast-moving vehicles → higher risk.

2. **Proximity risk** – minimum inter-vehicle distance.
   Vehicles close together → higher risk.

3. **Density risk** – total number of vehicles in frame.
   Dense traffic → higher baseline risk.

4. **Convergence risk** – two vehicles whose centroids are
   approaching each other rapidly.

The four factors are weighted and combined into a 0–100% score.
"""

import numpy as np
import math


# ── Tunable weights (sum need not be 1; normalised at end) ────────────────
W_SPEED       = 0.30
W_PROXIMITY   = 0.35
W_DENSITY     = 0.20
W_CONVERGENCE = 0.15

# ── Threshold constants ───────────────────────────────────────────────────
HIGH_SPEED_PX    = 25.0   # px/frame considered "fast"
MIN_SAFE_DIST_PX = 80.0   # pixels — below this is risky
HIGH_DENSITY     = 8      # vehicles in frame considered congested


class AccidentPredictor:
    """
    Stateless predictor: call predict(tracks) each frame.
    Keeps a short history of centroids to compute convergence.
    """

    def __init__(self):
        self._prev_centroids: dict[int, tuple] = {}  # track_id → (cx, cy)

    # ──────────────────────────────────────────────────────────────────────

    def predict(self, tracks: list[dict]) -> float:
        """
        Compute accident risk for the current frame.

        Parameters
        ----------
        tracks : list of track dicts from VehicleTracker.update()

        Returns
        -------
        float in [0, 100] representing risk percentage.
        """
        if len(tracks) == 0:
            return 0.0

        centroids = {
            t["id"]: (
                (t["x1"] + t["x2"]) // 2,
                (t["y1"] + t["y2"]) // 2,
            )
            for t in tracks
        }

        # ── Factor 1: Speed ────────────────────────────────────────────
        speeds = [t.get("speed", 0.0) for t in tracks]
        avg_speed  = np.mean(speeds) if speeds else 0.0
        speed_risk = min(avg_speed / HIGH_SPEED_PX, 1.0)

        # ── Factor 2: Proximity ────────────────────────────────────────
        proximity_risk = 0.0
        ids = list(centroids.keys())
        if len(ids) >= 2:
            min_dist = float("inf")
            for i in range(len(ids)):
                for j in range(i+1, len(ids)):
                    d = math.dist(centroids[ids[i]], centroids[ids[j]])
                    if d < min_dist:
                        min_dist = d
            proximity_risk = max(0.0, 1.0 - min_dist / MIN_SAFE_DIST_PX)

        # ── Factor 3: Density ──────────────────────────────────────────
        density_risk = min(len(tracks) / HIGH_DENSITY, 1.0)

        # ── Factor 4: Convergence ──────────────────────────────────────
        convergence_risk = 0.0
        conv_count = 0
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                id_a, id_b = ids[i], ids[j]
                if id_a in self._prev_centroids and id_b in self._prev_centroids:
                    prev_d = math.dist(self._prev_centroids[id_a],
                                       self._prev_centroids[id_b])
                    curr_d = math.dist(centroids[id_a], centroids[id_b])
                    if prev_d > curr_d and curr_d < MIN_SAFE_DIST_PX * 1.5:
                        # Vehicles approaching each other in danger zone
                        approach_rate = (prev_d - curr_d) / max(prev_d, 1)
                        convergence_risk += approach_rate
                        conv_count += 1

        if conv_count:
            convergence_risk = min(convergence_risk / conv_count, 1.0)

        # ── Combine ────────────────────────────────────────────────────
        risk = (
            W_SPEED       * speed_risk +
            W_PROXIMITY   * proximity_risk +
            W_DENSITY     * density_risk +
            W_CONVERGENCE * convergence_risk
        ) * 100.0  # → percentage

        # Clamp
        risk = min(max(risk, 0.0), 100.0)

        # Save centroids for next frame
        self._prev_centroids = centroids

        return round(risk, 2)

    def reset(self):
        self._prev_centroids.clear()
