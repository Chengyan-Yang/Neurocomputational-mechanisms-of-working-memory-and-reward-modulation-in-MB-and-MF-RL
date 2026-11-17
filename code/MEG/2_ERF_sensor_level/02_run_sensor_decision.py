# -*- coding: utf-8 -*-
"""Sensor-level ERF analysis for the 0-1.5 s decision window."""

from pathlib import Path
import sys


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from sensor_cluster_common import SensorAnalysisConfig, run_sensor_analysis


CONFIG = SensorAnalysisConfig(
    phase="decision",
    data_phase="decision",
    phase_label="Decision period (0-1.5 s)",
    conditions=("11", "21", "31", "41"),
    tmin=0.0,
    tmax=1.5,
)


if __name__ == "__main__":
    run_sensor_analysis(CONFIG)
