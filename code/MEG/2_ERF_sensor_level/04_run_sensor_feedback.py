# -*- coding: utf-8 -*-
"""Sensor-level ERF analysis for the full feedback episode."""

from pathlib import Path
import sys


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from sensor_cluster_common import SensorAnalysisConfig, run_sensor_analysis


CONFIG = SensorAnalysisConfig(
    phase="feedback",
    data_phase="feedback",
    phase_label="Feedback",
    conditions=("51", "52", "53", "54"),
    tmin=0.0,
    tmax=1.5,
)


if __name__ == "__main__":
    run_sensor_analysis(CONFIG)
