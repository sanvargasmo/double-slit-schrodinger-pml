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
from matplotlib.colors import PowerNorm

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
    result.add_argument("--snapshots", type=int, default=21, help="number of evenly spaced evolved states")
    result.add_argument(
        "--plot-snapshots",
        type=int,
        default=8,
        help="number of evolved states displayed in the output figure",
    )
    result.add_argument(
        "--plot-columns",
        type=int,
        default=4,
        help="maximum number of density panels per row",
    )
    result.add_argument(
        "--density-normalization",
        choices=("physical", "integral", "absolute"),
        default="physical",
        help=(
            "normalize each panel by the maximum density in the displayed non-PML "
            "region; use integral for conditional-density normalization, or absolute "
            "to retain the unscaled density"
        ),
    )
    result.add_argument(
        "--colormap",
        type=str,
        default="turbo",
        help="Matplotlib colormap used for density panels",
    )
    result.add_argument(
        "--color-gamma",
        type=float,
        default=0.5,
        help="color power-law exponent; values below one emphasize low densities",
    )

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
    if args.plot_snapshots < 2 or args.plot_snapshots > args.snapshots:
        raise ValueError("plot-snapshots must be between 2 and snapshots")
    if args.plot_columns < 1:
        raise ValueError("plot-columns must be at least one")
    if args.color_gamma <= 0.0:
        raise ValueError("color-gamma must be positive")
    if args.colormap not in matplotlib.colormaps:
        raise ValueError(f"Unknown Matplotlib colormap: {args.colormap}")
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
    panel_indices = np.unique(
        np.rint(np.linspace(0, args.snapshots - 1, args.plot_snapshots)).astype(int)
    )
    raw_densities = [np.abs(reconstruct(model, states[index], x, y)) ** 2 for index in panel_indices]

    # The physical normalization region is the visible part of the domain
    # before the x-PML interface. There is no y-PML, so the displayed y-range
    # is used in full.
    physical_x = np.ones_like(x, dtype=bool)
    if args.pml:
        physical_x = np.abs(x) <= args.pml_start
    if np.count_nonzero(physical_x) < 2:
        raise ValueError("The plot window does not contain a resolvable non-PML x interval")

    physical_probability = np.asarray(
        [
            np.trapezoid(
                np.trapezoid(density[physical_x, :], y, axis=1),
                x[physical_x],
            )
            for density in raw_densities
        ]
    )
    if physical_probability[0] <= np.finfo(float).eps:
        raise ValueError("The initial state has no probability in the displayed physical region")

    physical_peak_density = np.asarray(
        [np.max(density[physical_x, :]) for density in raw_densities]
    )
    peak_floor = max(np.finfo(float).eps, 1.0e-12 * physical_peak_density[0])
    if args.density_normalization == "physical":
        densities = [
            density / physical_peak_density[position]
            if physical_peak_density[position] > peak_floor
            else np.zeros_like(density)
            for position, density in enumerate(raw_densities)
        ]
        colorbar_label = r"$|\psi|^2/\rho_{\max,\mathrm{phys}}(t)$"
        vmax = 1.0
    elif args.density_normalization == "integral":
        probability_floor = max(np.finfo(float).eps, 1.0e-12 * physical_probability[0])
        densities = [
            density / physical_probability[position]
            if physical_probability[position] > probability_floor
            else np.zeros_like(density)
            for position, density in enumerate(raw_densities)
        ]
        colorbar_label = r"$|\psi|^2/P_{\mathrm{phys}}(t)$"
        vmax = max(np.quantile(density, 0.995) for density in densities)
    else:
        densities = raw_densities
        colorbar_label = r"$|\psi|^2$"
        vmax = max(np.quantile(density, 0.995) for density in densities)

    physical_probability_ratio = physical_probability / physical_probability[0]
    color_norm = PowerNorm(gamma=args.color_gamma, vmin=0.0, vmax=vmax)

    panel_count = len(panel_indices)
    panel_columns = min(args.plot_columns, panel_count)
    panel_rows = int(np.ceil(panel_count / panel_columns))
    fig = plt.figure(
        figsize=(3.1 * panel_columns + 0.8, 3.0 * panel_rows + 2.4),
        layout="constrained",
    )
    grid = fig.add_gridspec(
        panel_rows + 1,
        panel_columns + 1,
        width_ratios=(*([1.0] * panel_columns), 0.055),
        height_ratios=(*([1.0] * panel_rows), 0.55),
    )
    image = None
    for column, (index, density) in enumerate(zip(panel_indices, densities)):
        row, panel_column = divmod(column, panel_columns)
        ax = fig.add_subplot(grid[row, panel_column])
        image = ax.imshow(
            density.T,
            origin="lower",
            extent=(args.view_x_min, args.view_x_max, args.view_y_min, args.view_y_max),
            aspect="auto",
            cmap=args.colormap,
            norm=color_norm,
        )
        ax.set_title(
            rf"$t={times[index]:.3f}$   "
            rf"$\rho_{{\max,\rm phys}}={physical_peak_density[column]:.3g}$",
            fontsize=10,
        )
        ax.set_xlabel("x")
        if panel_column == 0:
            ax.set_ylabel("y")

    for empty_index in range(panel_count, panel_rows * panel_columns):
        row, panel_column = divmod(empty_index, panel_columns)
        empty_ax = fig.add_subplot(grid[row, panel_column])
        empty_ax.axis("off")

    colorbar_ax = fig.add_subplot(grid[:panel_rows, -1])
    fig.colorbar(image, cax=colorbar_ax, label=colorbar_label)

    probability_ax = fig.add_subplot(grid[panel_rows, :panel_columns])
    probability_ax.plot(
        times,
        probability / probability[0],
        color="#e76f51",
        lw=2,
        label=r"$P_{\mathrm{total}}(t)/P_{\mathrm{total}}(0)$",
    )
    probability_ax.plot(
        times[panel_indices],
        physical_probability_ratio,
        color="#277da1",
        marker="o",
        lw=1.5,
        label=r"$P_{\mathrm{phys}}(t)/P_{\mathrm{phys}}(0)$",
    )
    probability_ax.set(xlabel="time", ylabel="retained probability")
    probability_ax.legend()
    mode = "PML" if args.pml else "no PML"
    fig.suptitle(f"Double slit parameter experiment ({mode})")

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
        "derived": {
            "lx": lx,
            "basis_shape": model.shape,
            "basis_size": model.size,
            "plotted_snapshot_indices": panel_indices.tolist(),
            "plotted_times": times[panel_indices].tolist(),
            "physical_normalization_x_interval": [
                float(x[physical_x][0]),
                float(x[physical_x][-1]),
            ],
            "physical_normalization_y_interval": [float(y[0]), float(y[-1])],
            "plotted_physical_probability": physical_probability.tolist(),
            "plotted_physical_probability_ratio": physical_probability_ratio.tolist(),
            "plotted_physical_peak_density": physical_peak_density.tolist(),
            "color_scale": {
                "colormap": args.colormap,
                "gamma": args.color_gamma,
                "vmin": 0.0,
                "vmax": vmax,
            },
        },
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
