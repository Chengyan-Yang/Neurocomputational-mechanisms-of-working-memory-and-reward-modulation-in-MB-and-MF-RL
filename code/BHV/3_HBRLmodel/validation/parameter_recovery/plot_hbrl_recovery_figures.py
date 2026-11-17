"""Render four load-overlaid beta recovery panels from hierarchical refits."""
import argparse
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE.parent / "_mplconfig"))
import matplotlib as mpl
mpl.use("Agg")
import pandas as pd

sys.path.insert(0, str(BASE.parent))
from validation_panel_style import recovery_panel, save_panel

COLORS = {"low": "#eacdd2", "high": "#b03949"}


def make_load_overlay_panels(comp, directory):
    for system in ("mf", "mb"):
        for reward in ("small", "large"):
            pairs = {}
            for load, suffix in (("low", "lload"), ("high", "hload")):
                param = f"beta_{system}_{reward}_{suffix}"
                pairs[load] = comp.loc[comp.parameter == param, ["generating", "recovered"]]
            title = rf"$\beta_{{\mathrm{{{system.upper()},\ {reward}}}}}$"
            fig = recovery_panel(title, pairs, COLORS, hierarchical=True)
            save_panel(fig, directory, f"parameter_recovery_beta_{system}_{reward}")


def write_condition_key(root):
    key = pd.DataFrame({
        "condition": ["LS", "LL", "HS", "HL"],
        "WM_load": ["low", "low", "high", "high"],
        "reward": ["small", "large", "small", "large"],
        "note": [
            "low WM load, small reward",
            "low WM load, large reward",
            "high WM load, small reward",
            "high WM load, large reward",
        ],
    })
    key.to_csv(root / "condition_code_key.csv", index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default=os.environ.get("HBRL_RECOVERY_LABEL", "MEG_HBRL_5sets"))
    root = BASE / f"outputs_{parser.parse_args().label}"
    comp = pd.read_csv(root / "parameter_recovery_comparison_long.csv")
    directory = root / "Figures" / "single_panels"
    make_load_overlay_panels(comp, directory)
    write_condition_key(root)
    print("Saved HBRL recovery figures to", directory)


if __name__ == "__main__":
    main()




