"""
Traffic Monitoring Dashboard — Entry Point
==========================================
Run this file to launch the full application:
    python main.py
"""

import sys
import os

# Ensure traffic_project sub-packages are importable from root
traffic_project_path = os.path.join(os.path.dirname(__file__), 'traffic_project')
sys.path.insert(0, traffic_project_path)

from traffic_project.gui.dashboard import TrafficDashboard


def main():
    app = TrafficDashboard()
    app.run()


if __name__ == "__main__":
    main()
