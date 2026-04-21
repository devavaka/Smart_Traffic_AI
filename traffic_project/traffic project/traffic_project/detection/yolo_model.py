"""
detection/yolo_model.py
=======================
YOLOv8-based vehicle detector.

On first run, Ultralytics auto-downloads `yolov8n.pt` (~6 MB).
If Ultralytics is not installed, a simple OpenCV-based mock detector
is used so the GUI still works for demo purposes.

Detected classes mapped to vehicle types:
  COCO id → label
  2  → Car
  3  → Motorcycle / Bike
  5  → Bus
  7  → Truck
"""

import cv2
import numpy as np
import random
import string

# ── Try importing Ultralytics ──────────────────────────────────────────────
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed — running in MOCK detection mode.")

# COCO class ids for vehicles
VEHICLE_CLASSES = {2: "Car", 3: "Bike", 5: "Bus", 7: "Truck"}
CONF_THRESHOLD  = 0.40   # minimum confidence to keep a detection


class VehicleDetector:
    """
    Detects vehicles in a frame using YOLOv8.
    Falls back to mock detections if model is unavailable.
    """

    def __init__(self, model_name: str = "yolov8n.pt"):
        self.model = None
        if _YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_name)
                print(f"[INFO] YOLOv8 model loaded: {model_name}")
            except Exception as e:
                print(f"[WARN] Could not load YOLO model ({e}) — using mock.")
        self._mock_state = []   # persist mock boxes across frames

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run inference on a single BGR frame.

        Returns a list of detection dicts:
          {
            "x1": int, "y1": int, "x2": int, "y2": int,
            "label": str,          # e.g. "Car"
            "confidence": float    # 0-1
          }
        """
        if self.model is not None:
            return self._yolo_detect(frame)
        return self._mock_detect(frame)

    def extract_plate(self, frame: np.ndarray,
                      tracks: list[dict]) -> str | None:
        """
        Naive number-plate extraction.

        Strategy:
          1. For each tracked vehicle, crop the bottom-third of its bbox
             (where plates usually appear).
          2. Run edge detection + contour search for rectangular region.
          3. If a valid plate ROI is found, generate a synthetic plate string
             for demo purposes (real OCR would use EasyOCR / Tesseract).

        Returns a plate string or None.
        """
        if not tracks:
            return None

        # Pick the largest (closest) vehicle for plate attempt
        track = max(tracks, key=lambda t: (t["x2"]-t["x1"])*(t["y2"]-t["y1"]))
        x1, y1, x2, y2 = track["x1"], track["y1"], track["x2"], track["y2"]

        # Crop bottom 40% of the bounding box
        crop_y1 = int(y1 + 0.60 * (y2 - y1))
        crop = frame[max(0, crop_y1):y2, max(0, x1):x2]

        if crop.size == 0:
            return None

        # Check if a rectangular contour exists (simple plate heuristic)
        gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
            peri   = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                # Found a quad — generate synthetic plate text for demo
                return self._synthetic_plate()

        # Low probability random plate to keep demo lively
        if random.random() < 0.15:
            return self._synthetic_plate()

        return None

    # ──────────────────────────────────────────────────────────────────────
    # INTERNAL
    # ──────────────────────────────────────────────────────────────────────

    def _yolo_detect(self, frame: np.ndarray) -> list[dict]:
        """Run real YOLOv8 inference."""
        results = self.model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "label": VEHICLE_CLASSES[cls_id],
                "confidence": round(conf, 2),
            })
        return detections

    def _mock_detect(self, frame: np.ndarray) -> list[dict]:
        """
        Synthetic detections for demo / no-model mode.
        Boxes drift slightly each frame to simulate movement.
        """
        h, w = frame.shape[:2]
        labels = list(VEHICLE_CLASSES.values())

        # Initialise mock boxes
        if not self._mock_state:
            n = random.randint(3, 6)
            for _ in range(n):
                bw = random.randint(w//8, w//4)
                bh = random.randint(h//8, h//5)
                bx = random.randint(0, w - bw)
                by = random.randint(0, h - bh)
                self._mock_state.append({
                    "x": bx, "y": by, "w": bw, "h": bh,
                    "dx": random.choice([-1,1]) * random.randint(1, 3),
                    "dy": random.choice([-1,1]) * random.randint(1, 2),
                    "label": random.choice(labels),
                    "conf": round(random.uniform(0.6, 0.95), 2),
                })

        detections = []
        for s in self._mock_state:
            # Move box
            s["x"] = max(0, min(w - s["w"], s["x"] + s["dx"]))
            s["y"] = max(0, min(h - s["h"], s["y"] + s["dy"]))
            # Bounce
            if s["x"] <= 0 or s["x"] + s["w"] >= w:
                s["dx"] *= -1
            if s["y"] <= 0 or s["y"] + s["h"] >= h:
                s["dy"] *= -1

            detections.append({
                "x1": s["x"], "y1": s["y"],
                "x2": s["x"] + s["w"], "y2": s["y"] + s["h"],
                "label": s["label"],
                "confidence": s["conf"],
            })
        return detections

    @staticmethod
    def _synthetic_plate() -> str:
        """Generate a random Indian-style number plate for demo."""
        states = ["MH", "DL", "KA", "TN", "GJ", "RJ", "UP", "WB", "PB", "HR"]
        state  = random.choice(states)
        num1   = f"{random.randint(10,99)}"
        chars  = "".join(random.choices(string.ascii_uppercase, k=2))
        num2   = f"{random.randint(1000,9999)}"
        return f"{state} {num1} {chars} {num2}"
