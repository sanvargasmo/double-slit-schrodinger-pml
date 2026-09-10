"""Regenerate the strong-barrier README comparison using the experiment viewer.

Run from the repository root after installing the package. These illustrations
are separate from the original V0=1 quantitative PML validation benchmarks.
"""

import os
from pathlib import Path

from run_experiment import main as run_experiment


def main() -> None:
    os.chdir(Path(__file__).resolve().parents[1])
    figures = Path("figures")
    # E(k0)=k0**2/2=450, so V0=1500 blocks the opaque part of the screen.
    # Use identical geometry, basis, incident state, times, and display window.
    common = [
        "--lx", "3.5", "--ly", "6", "--nx", "120", "--ny", "100",
        "--barrier-height", "1500", "--barrier-thickness", "0.20",
        "--slit-width", "0.30", "--slit-separation", "0.40",
        "--k0", "30", "--packet-left", "-1", "--packet-right", "-0.525",
        "--pml-start", "1.5", "--pml-thickness", "2",
        "--pml-order", "4", "--target-reflection", "1e-3",
        "--t-final", "0.20", "--snapshots", "41",
        "--plot-snapshots", "9", "--plot-columns", "3",
        "--view-x-min", "-1.4", "--view-x-max", "1.4",
        "--view-y-min", "-1", "--view-y-max", "1",
        "--density-normalization", "physical", "--colormap", "turbo",
        "--color-gamma", "0.5",
    ]
    for mode, name in (("--no-pml", "hreal_evolution"), ("--pml", "pml_evolution")):
        run_experiment(common + [mode, "--output", str(figures / f"{name}.jpg")])


if __name__ == "__main__":
    main()
