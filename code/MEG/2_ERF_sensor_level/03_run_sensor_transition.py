# -*- coding: utf-8 -*-
"""Sensor-level analysis time-locked to Transition scene onset."""

from pathlib import Path
import sys

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from sensor_cluster_common import SensorAnalysisConfig, run_sensor_analysis


# The 0-0.5 s window matches the source analysis and avoids the original 1.2 s
# window, which included substantial activity after the second response.
CONFIG = SensorAnalysisConfig(
    phase="transition",
    phase_label="Transition scene onset",
    conditions=("13", "23", "33", "43"),
    tmin=0.0,
    tmax=0.5,
)


if __name__ == "__main__":
    run_sensor_analysis(CONFIG)
