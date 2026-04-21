"""
analytics/heatmap.py
====================
Accumulates vehicle centroid positions and renders a colour heatmap
overlay that can be blended onto the video feed.

Uses a Gaussian-blurred accumulator → normalised → mapped to a
Jet colourmap → BGRA overlay for cv2.addWeighted().
"""

import cv2
import numpy as np


class HeatmapGenerator:
    """
    Incrementally builds a heatmap from (x, y) vehicle positions.

    Usage::

        hm = HeatmapGenerator(width=640, height=360)
        hm.add_point(cx, cy)             # call per tracked vehicle per frame
        overlay = hm.get_overlay(shape)  # returns BGR image same size as frame
    """

    def __init__(self, width: int = 640, height: int = 360,
                 decay: float = 0.997):
        """
        Parameters
        ----------
        width, height : canvas dimensions (should match video frame)
        decay         : per-frame multiplier to fade old activations (0–1)
        """
        self.width  = width
        self.height = height
        self.decay  = decay
        self._accumulator = np.zeros((height, width), dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────

    def add_point(self, cx: int, cy: int, radius: int = 20,
                  weight: float = 1.0) -> None:
        """
        Add a Gaussian blob centred at (cx, cy) to the accumulator.

        Parameters
        ----------
        cx, cy  : centroid coordinates in the frame
        radius  : spread of the Gaussian blob in pixels
        weight  : intensity contribution
        """
        # Apply decay to the whole accumulator each time a point is added
        self._accumulator *= self.decay

        # Clamp to canvas
        cx = int(np.clip(cx, 0, self.width  - 1))
        cy = int(np.clip(cy, 0, self.height - 1))

        # Draw a filled circle and blur it for smooth heatmap
        tmp = np.zeros_like(self._accumulator)
        cv2.circle(tmp, (cx, cy), radius, weight, -1)
        tmp = cv2.GaussianBlur(tmp, (radius*2+1 | 1, radius*2+1 | 1), 0)
        self._accumulator += tmp

    def get_overlay(self, frame_shape: tuple) -> np.ndarray:
        """
        Render the heatmap as a BGR image the same size as the video frame.

        Parameters
        ----------
        frame_shape : (H, W, C) tuple of the target frame

        Returns
        -------
        BGR ndarray, same (H, W) as frame_shape — ready for addWeighted.
        """
        fh, fw = frame_shape[:2]

        if self._accumulator.max() < 1e-6:
            return np.zeros((fh, fw, 3), dtype=np.uint8)

        # Normalise to 0-255
        norm = cv2.normalize(self._accumulator, None, 0, 255,
                             cv2.NORM_MINMAX).astype(np.uint8)

        # Apply Jet colourmap
        coloured = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

        # Resize to match frame if needed
        if (norm.shape[1], norm.shape[0]) != (fw, fh):
            coloured = cv2.resize(coloured, (fw, fh))

        # Mask near-zero regions → transparent (no overlay where no traffic)
        norm_rs = cv2.resize(norm, (fw, fh))
        coloured[norm_rs < 10] = 0

        return coloured

    def reset(self) -> None:
        """Clear all accumulated heatmap data."""
        self._accumulator[:] = 0.0

    def save_heatmap(self, path: str, frame_shape: tuple) -> None:
        """Save current heatmap as a PNG image file."""
        overlay = self.get_overlay(frame_shape)
        cv2.imwrite(path, overlay)
        print(f"[INFO] Heatmap saved to {path}")
