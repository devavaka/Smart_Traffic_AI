"""
Traffic Monitoring Dashboard — Entry Point
==========================================
Run this file to launch the full application:
    python main.py
"""

import sys
import os

# Ensure sub-packages are importable regardless of working directory
sys.path.insert(0, os.path.dirname(__file__))

from gui.dashboard import TrafficDashboard


def main():
    app = TrafficDashboard()
    app.run()


if __name__ == "__main__":
    main()
