"""Draw the four phases from saved cluster tests, without running permutations."""
import argparse
from pathlib import Path
import runpy
import sys

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import tfr_analysis_config as config


def plot_combination(phase, band, effect):
    print(f"Plotting {phase} | {band} | {effect}", flush=True)
    runpy.run_path(str(MODULE_DIR / "02_group_cluster_test.py"))["plot_saved_clusters"](phase, band, effect)
    runpy.run_path(str(MODULE_DIR / "plotting" / "condition_heatmaps.py"))["main"](phase, band, effect)
    if effect == "interaction":
        runpy.run_path(str(MODULE_DIR / "plotting" / "interaction_effects.py"))["main"](phase, band, effect)
    runpy.run_path(str(MODULE_DIR / "plotting" / "band_timecourses.py"))["main"](phase, band, effect)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=config.ANALYZED_PHASES, help="Plot only this phase.")
    args = parser.parse_args()
    phases = [args.phase] if args.phase else config.ANALYZED_PHASES
    for phase in phases:
        for band in config.BANDS:
            for effect in config.EFFECT_WEIGHTS:
                path = config.result_dir(phase, band, effect) / "cluster_results.pkl"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing {path}. Run script 02 for phase {phase}.")
    for phase in phases:
        for band in config.BANDS:
            for effect in config.EFFECT_WEIGHTS:
                plot_combination(phase, band, effect)


if __name__ == "__main__":
    main()
