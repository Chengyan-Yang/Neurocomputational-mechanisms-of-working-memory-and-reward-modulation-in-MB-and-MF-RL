# -*- coding: utf-8 -*-
"""Sensor-level analysis for magnitude cue onset (Incentive)."""

from pathlib import Path
import sys

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from sensor_cluster_common import SensorAnalysisConfig, run_sensor_analysis


CONFIG = SensorAnalysisConfig(
    phase="incentive",
    phase_label="Incentive cue onset",
    conditions=("1", "2", "3", "4"),
    tmin=0.0,
    tmax=1.0,
)


if __name__ == "__main__":
    run_sensor_analysis(CONFIG)
