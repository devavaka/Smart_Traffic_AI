"""
generate_sample.py
==================
Run this once to create assets/sample_video.mp4 before the first demo.

Usage:
    python generate_sample.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from utils.helpers import generate_sample_video

if __name__ == "__main__":
    path = generate_sample_video("assets/sample_video.mp4", duration_s=30, fps=20)
    print(f"Done → {path}")
