"""Run one configurable double-slit experiment from the command line.

Examples are documented in the repository README.  The script intentionally
keeps every physical and numerical parameter explicit so parameter studies do
not require editing the core Hamiltonian implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from double_slit_pml.model import (
    Geometry,
    PMLSettings,
    PlaneWaveModel,
    compact_packet,
    evolve,
    project_separable_state,
    reconstruct,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Run a configurable plane-wave double-slit simulation with or without an x-only PML."
    )
    result.add_argument("--pml", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--lx", type=float, default=None, help="x half-width; with PML it must equal start + thickness")
    result.add_argument("--ly", type=float, default=6.0, help="y half-width")
    result.add_argument("--nx", type=int, default=80, help="maximum x plane-wave index")
    result.add_argument("--ny", type=int, default=30, help="maximum y plane-wave index")

    result.add_argument("--slit-width", type=float, default=0.30)
    result.add_argument("--slit-separation", type=float, default=0.40)
    result.add_argument("--barrier-thickness", type=float, default=0.20)
    result.add_argument("--barrier-height", type=float, default=1.0)

    result.add_argument("--pml-start", type=float, default=1.50)
    result.add_argument("--pml-thickness", type=float, default=2.00)
    result.add_argument("--pml-order", type=int, default=4)
    result.add_argument("--target-reflection", type=float, default=1.0e-3)

    result.add_argument("--packet-left", type=float, default=-1.0)
    result.add_argument("--packet-right", type=float, default=-0.525)
    result.add_argument("--k0", type=float, default=30.0, help="incident x wave number")
    result.add_argument("--t-final", type=float, default=0.20)
    result.add_argument("--snapshots", type=int, default=21, help="number of evenly spaced saved states")

    result.add_argument("--view-x-min", type=float, default=-1.2)
    result.add_argument("--view-x-max", type=float, default=2.4)
    result.add_argument("--view-y-min", type=float, default=-1.0)
    result.add_argument("--view-y-max", type=float, default=1.0)
    result.add_argument("--output", type=Path, default=Path("results/experiment.png"))
    result.add_argument("--metrics", type=Path, default=None, help="JSON path; defaults to the output name with .json")
    return result


def validate(args: argparse.Namespace) -> None:
    if args.ly <= 0 or args.nx < 0 or args.ny < 0:
        raise ValueError("Box half-widths must be positive and spectral cutoffs must be non-negative")
    if args.snapshots < 2 or args.t_final <= 0:
        raise ValueError("Use at least two snapshots and a positive final time")
    if args.packet_left >= args.packet_right:
        raise ValueError("packet-left must be smaller than packet-right")
    if not 0 < args.target_reflection < 1:
        raise ValueError("target-reflection must lie strictly between zero and one")


def main() -> None:
    args = parser().parse_args()
    validate(args)

    geometry = Geometry(
        slit_width=args.slit_width,
        slit_separation=args.slit_separation,
        barrier_thickness=args.barrier_thickness,
        barrier_height=args.barrier_height,
    )
    pml_settings = PMLSettings(
        start=args.pml_start,
        thickness=args.pml_thickness,
        order=args.pml_order,
        target_reflection=args.target_reflection,
    )
    default_lx = pml_settings.outer_half_length
    lx = default_lx if args.lx is None else args.lx
    if args.pml and not np.isclose(lx, default_lx):
        raise ValueError("With PML enabled, lx must equal pml-start + pml-thickness")

    model = PlaneWaveModel(
        Lx=lx,
        Ly=args.ly,
        nx=args.nx,
        ny=args.ny,
        geometry=geometry,
        pml=pml_settings if args.pml else None,
    )
    packet = lambda x: compact_packet(
        x,
        left=args.packet_left,
        right=args.packet_right,
        k0=args.k0,
    )
    state0 = project_separable_state(model, packet)
    times = np.linspace(0.0, args.t_final, args.snapshots)
    states = evolve(model, state0, times)
    probability = np.sum(np.abs(states) ** 2, axis=1)

    x = np.linspace(args.view_x_min, args.view_x_max, 300)
    y = np.linspace(args.view_y_min, args.view_y_max, 190)
    panel_indices = np.unique(np.linspace(0, args.snapshots - 1, min(4, args.snapshots), dtype=int))
    densities = [np.abs(reconstruct(model, states[index], x, y)) ** 2 for index in panel_indices]
    vmax = max(np.quantile(density, 0.995) for density in densities)

    fig = plt.figure(figsize=(3.1 * len(panel_indices), 5.3))
    grid = fig.add_gridspec(2, len(panel_indices), height_ratios=(2.5, 1.0))
    image = None
    for column, (index, density) in enumerate(zip(panel_indices, densities)):
        ax = fig.add_subplot(grid[0, column])
        image = ax.imshow(
            density.T,
            origin="lower",
            extent=(args.view_x_min, args.view_x_max, args.view_y_min, args.view_y_max),
            aspect="auto",
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(f"t = {times[index]:.3f}")
        ax.set_xlabel("x")
        if column == 0:
            ax.set_ylabel("y")
    probability_ax = fig.add_subplot(grid[1, :])
    probability_ax.plot(times, probability, color="#e76f51", lw=2)
    probability_ax.set(xlabel="time", ylabel="total probability")
    mode = "PML" if args.pml else "no PML"
    fig.suptitle(f"Double slit parameter experiment ({mode})")
    fig.colorbar(image, ax=fig.axes[:-1], label=r"$|\psi|^2$", shrink=0.75, pad=0.015)
    fig.subplots_adjust(top=0.88, bottom=0.10, hspace=0.40, wspace=0.30)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metrics_path = args.metrics or args.output.with_suffix(".json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    configuration = vars(args).copy()
    configuration["output"] = str(args.output)
    configuration["metrics"] = str(metrics_path)
    result = {
        "configuration": configuration,
        "derived": {"lx": lx, "basis_shape": model.shape, "basis_size": model.size},
        "times": times.tolist(),
        "total_probability": probability.tolist(),
        "final_probability_ratio": float(probability[-1] / probability[0]),
    }
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Figure:  {args.output}")
    print(f"Metrics: {metrics_path}")
    print(f"Final probability ratio: {result['final_probability_ratio']:.8f}")


if __name__ == "__main__":
    main()

