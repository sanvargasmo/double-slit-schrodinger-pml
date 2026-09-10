"""Spot-check the README's transmitted density at t=0.05 with a larger basis."""

import json
from pathlib import Path

import numpy as np

from double_slit_pml.model import (
    Geometry, PMLSettings, PlaneWaveModel, compact_packet,
    project_separable_state, evolve, reconstruct,
)


def main() -> None:
    x = np.linspace(0.12, 1.4, 180)
    y = np.linspace(-1, 1, 200)
    densities = []
    for nx, ny in ((120, 100), (160, 140)):
        model = PlaneWaveModel(
            Lx=3.5, Ly=6, nx=nx, ny=ny,
            geometry=Geometry(barrier_height=1500), pml=PMLSettings(),
        )
        initial = project_separable_state(model, lambda x: compact_packet(x, k0=30))
        state = evolve(model, initial, [0, 0.05])[-1]
        densities.append(np.abs(reconstruct(model, state, x, y)) ** 2)
    coarse, fine = densities
    result = {
        "time": 0.05,
        "window": {"x": [0.12, 1.4], "y": [-1, 1]},
        "coarse_cutoffs": [120, 100],
        "fine_cutoffs": [160, 140],
        "relative_density_l2_error": float(np.linalg.norm(coarse - fine) / np.linalg.norm(fine)),
        "normalized_density_l2_error": float(
            np.linalg.norm(coarse / coarse.max() - fine / fine.max())
            / np.linalg.norm(fine / fine.max())
        ),
    }
    output = Path(__file__).resolve().parents[1] / "figures/evolution_resolution.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
