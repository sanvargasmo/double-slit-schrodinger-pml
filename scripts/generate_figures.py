"""Reproduce every numerical figure and validation number used in the README."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from double_slit_pml.diagnostics import integrate_rectangle, relative_density_error
from double_slit_pml.model import (
    Geometry,
    PMLSettings,
    PlaneWaveModel,
    compact_packet,
    evolve,
    project_separable_state,
    reconstruct,
    sigma_and_derivative,
)


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

BLUE = "#1769aa"
ORANGE = "#e76f51"
GREEN = "#2a9d8f"
DARK = "#243447"

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 180,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def density_panel(
    model: PlaneWaveModel,
    states: np.ndarray,
    times: np.ndarray,
    name: str,
    x_limits: tuple[float, float],
    title: str,
) -> None:
    x = np.linspace(*x_limits, 280)
    y = np.linspace(-1.0, 1.0, 170)
    densities = [np.abs(reconstruct(model, state, x, y)) ** 2 for state in states]
    vmax = max(np.quantile(rho, 0.995) for rho in densities)
    fig, axes = plt.subplots(1, len(times), figsize=(3.0 * len(times), 2.75), sharex=True, sharey=True)
    for ax, rho, time in zip(np.atleast_1d(axes), densities, times):
        image = ax.imshow(
            rho.T,
            origin="lower",
            extent=(*x_limits, -1.0, 1.0),
            aspect="auto",
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
        )
        g = model.geometry
        slit_edges = (
            -1.0,
            -g.slit_separation / 2 - g.slit_width / 2,
            -g.slit_separation / 2 + g.slit_width / 2,
            g.slit_separation / 2 - g.slit_width / 2,
            g.slit_separation / 2 + g.slit_width / 2,
            1.0,
        )
        for bottom, top in ((slit_edges[0], slit_edges[1]), (slit_edges[2], slit_edges[3]), (slit_edges[4], slit_edges[5])):
            ax.add_patch(
                Rectangle(
                    (-g.barrier_thickness / 2, bottom),
                    g.barrier_thickness,
                    top - bottom,
                    facecolor="none",
                    edgecolor="white",
                    linewidth=0.7,
                    alpha=0.8,
                )
            )
        ax.set_title(f"t = {time:.2f}")
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.suptitle(title, y=1.02)
    fig.colorbar(image, ax=axes, label=r"$|\psi|^2$", shrink=0.78, pad=0.02)
    save(fig, name)


def x_probability(model: PlaneWaveModel, states: np.ndarray, x: np.ndarray) -> np.ndarray:
    """x marginal for a ny=0 state (integrated analytically over y)."""

    values = []
    for state in states:
        psi = reconstruct(model, state, x, np.array([0.0]))[:, 0]
        values.append(2.0 * model.Ly * np.abs(psi) ** 2)
    return np.asarray(values)


def main() -> None:
    settings = PMLSettings(start=1.5, thickness=2.0, order=4, target_reflection=1e-3)

    # Exact geometry and domain separation used throughout the project.
    geometry = Geometry()
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.4, 5.2), gridspec_kw={"height_ratios": [1.05, 1.0]})
    ax0.set_xlim(-0.5, 0.5)
    ax0.set_ylim(-1.0, 1.0)
    slit_intervals = [
        (-geometry.slit_separation / 2 - geometry.slit_width / 2, -geometry.slit_separation / 2 + geometry.slit_width / 2),
        (geometry.slit_separation / 2 - geometry.slit_width / 2, geometry.slit_separation / 2 + geometry.slit_width / 2),
    ]
    solid_segments = [(-1.0, slit_intervals[0][0]), (slit_intervals[0][1], slit_intervals[1][0]), (slit_intervals[1][1], 1.0)]
    for bottom, top in solid_segments:
        ax0.add_patch(
            Rectangle(
                (-geometry.barrier_thickness / 2, bottom),
                geometry.barrier_thickness,
                top - bottom,
                facecolor=DARK,
                edgecolor=DARK,
            )
        )
    ax0.annotate("two apertures", xy=(0.12, 0.20), xytext=(0.28, 0.55), arrowprops={"arrowstyle": "->", "color": BLUE}, color=BLUE)
    ax0.set(xlabel="x", ylabel="y", title="Double-slit potential mask in the physical region")
    ax0.set_aspect("equal")

    profile_x = np.linspace(-settings.outer_half_length, settings.outer_half_length, 1200)
    profile_sigma, _ = sigma_and_derivative(profile_x, settings)
    ax1.plot(profile_x, profile_sigma, color=ORANGE, lw=2)
    ax1.axvspan(-1.0, 1.0, color=BLUE, alpha=0.10, label="region of interest")
    ax1.axvspan(-settings.outer_half_length, -settings.start, color=ORANGE, alpha=0.10, label="PML")
    ax1.axvspan(settings.start, settings.outer_half_length, color=ORANGE, alpha=0.10)
    ax1.axvline(-settings.start, color=ORANGE, ls="--", lw=1)
    ax1.axvline(settings.start, color=ORANGE, ls="--", lw=1)
    ax1.set(xlabel="x", ylabel=r"$\sigma(x)$", title="Physical window, buffer, and quartic PML")
    ax1.legend(ncol=2, loc="upper center")
    fig.tight_layout()
    save(fig, "geometry_and_pml.png")

    # Hreal: the original periodic plane-wave box, intentionally without PML.
    hreal = PlaneWaveModel(Lx=1.0, Ly=1.0, nx=22, ny=18)
    hreal_state = project_separable_state(hreal, lambda x: compact_packet(x, k0=12.0))
    hreal_times = np.array([0.0, 0.05, 0.10, 0.15])
    hreal_states = evolve(hreal, hreal_state, hreal_times)
    density_panel(
        hreal,
        hreal_states,
        hreal_times,
        "hreal_evolution.png",
        (-1.0, 1.0),
        "Hreal: evolution in the periodic plane-wave box (no PML)",
    )

    # The actual double-slit calculation with the validated x-only PML.
    pml_model = PlaneWaveModel(Lx=3.5, Ly=6.0, nx=80, ny=30, pml=settings)
    packet = lambda x: compact_packet(x, k0=30.0)
    pml_state = project_separable_state(pml_model, packet)
    pml_times = np.linspace(0.0, 0.2, 21)
    pml_states = evolve(pml_model, pml_state, pml_times)
    density_panel(
        pml_model,
        pml_states[[0, 5, 10, 20]],
        pml_times[[0, 5, 10, 20]],
        "pml_evolution.png",
        (-1.2, 2.4),
        "Double slit with an x-only PML (physical window shown)",
    )

    # Independent free-packet benchmark: PML against a large conservative box.
    free = Geometry(barrier_height=0.0)
    benchmark_pml = PlaneWaveModel(Lx=3.5, Ly=1.0, nx=80, ny=0, geometry=free, pml=settings)
    benchmark_ref = PlaneWaveModel(Lx=8.0, Ly=1.0, nx=180, ny=0, geometry=free)
    state_pml = project_separable_state(benchmark_pml, packet)
    state_ref = project_separable_state(benchmark_ref, packet)
    bench_times = np.linspace(0.0, 0.2, 41)
    states_pml = evolve(benchmark_pml, state_pml, bench_times)
    states_ref = evolve(benchmark_ref, state_ref, bench_times)

    x_profile = np.linspace(-1.5, 6.2, 900)
    profile_pml = x_probability(benchmark_pml, states_pml[[-1]], x_profile)[0]
    profile_ref = x_probability(benchmark_ref, states_ref[[-1]], x_profile)[0]
    fig, ax = plt.subplots(figsize=(8.4, 3.7))
    ax.plot(x_profile, profile_ref, color=DARK, lw=1.6, label="large box, no PML")
    ax.plot(x_profile, profile_pml, color=ORANGE, lw=1.8, label="PML")
    ax.axvspan(settings.start, settings.outer_half_length, color=ORANGE, alpha=0.12, label="PML layer")
    ax.axvline(settings.start, color=ORANGE, ls="--", lw=1)
    ax.set(xlabel="x", ylabel="x-marginal probability", title="Outgoing packet at t = 0.20")
    ax.set_xlim(-1.5, 6.2)
    ax.legend(loc="upper right")
    save(fig, "pml_vs_large_box.png")

    norm_pml = np.sum(np.abs(states_pml) ** 2, axis=1)
    norm_ref = np.sum(np.abs(states_ref) ** 2, axis=1)
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.plot(bench_times, norm_ref, color=DARK, lw=1.8, label="large box")
    ax.plot(bench_times, norm_pml, color=ORANGE, lw=2.0, label="PML")
    ax.set(xlabel="time", ylabel="total probability", ylim=(-0.03, 1.04), title="Probability leaves through the absorbing layer")
    ax.legend()
    save(fig, "probability_absorption.png")

    # Upstream monitor for the full double-slit problem.
    reference_model = PlaneWaveModel(Lx=8.0, Ly=6.0, nx=183, ny=30)
    reference_state = project_separable_state(reference_model, packet)
    reference_states = evolve(reference_model, reference_state, pml_times)
    monitor_pml = np.array(
        [integrate_rectangle(pml_model, state, (-1.4, -0.2), (-1.0, 1.0), (140, 90)) for state in pml_states]
    )
    monitor_ref = np.array(
        [integrate_rectangle(reference_model, state, (-1.4, -0.2), (-1.0, 1.0), (140, 90)) for state in reference_states]
    )
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.semilogy(pml_times, monitor_ref, color=DARK, lw=1.7, label="large-box reference")
    ax.semilogy(pml_times, monitor_pml, color=ORANGE, lw=1.7, ls="--", label="PML")
    ax.set(xlabel="time", ylabel="monitor probability", title="Upstream reflection monitor")
    ax.legend()
    save(fig, "reflection_monitor.png")

    excess = np.maximum(monitor_pml - monitor_ref, 0.0)
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.plot(pml_times, 100.0 * excess / monitor_pml[0], color=GREEN, lw=1.9)
    ax.axvspan(0.08, 0.2, color=GREEN, alpha=0.08, label="post-passage window")
    ax.set(xlabel="time", ylabel="positive excess (% of initial monitor)", title="Conservative upper bound on reflected probability")
    ax.legend()
    save(fig, "reflection_excess.png")

    # y-box convergence.  sqrt(Ly) removes the arbitrary unit-normalization of
    # a y-uniform plane wave, so all boxes represent the same incident density.
    ly_models: dict[int, PlaneWaveModel] = {}
    ly_states: dict[int, np.ndarray] = {}
    for ly, ny in ((5, 25), (6, 30), (7, 35)):
        model = PlaneWaveModel(Lx=3.5, Ly=float(ly), nx=80, ny=ny, pml=settings)
        initial = project_separable_state(model, packet)
        final = evolve(model, initial, [0.0, 0.2])[-1]
        ly_models[ly] = model
        ly_states[ly] = np.sqrt(ly) * final

    error_5_6 = relative_density_error(ly_models[5], ly_states[5], ly_models[6], ly_states[6])
    error_6_7 = relative_density_error(ly_models[6], ly_states[6], ly_models[7], ly_states[7])
    fig, ax = plt.subplots(figsize=(5.9, 3.7))
    ax.bar(["Ly 5 vs 6", "Ly 6 vs 7"], 100 * np.array([error_5_6, error_6_7]), color=[BLUE, GREEN], width=0.58)
    ax.set(ylabel="relative density difference (%)", title=r"Convergence inside $|x|,|y|\leq 1$ at t = 0.20")
    for index, value in enumerate((error_5_6, error_6_7)):
        ax.text(index, 100 * value, f"{100 * value:.3f}%", ha="center", va="bottom")
    save(fig, "ly_convergence.png")

    post_passage = bench_times >= 0.08
    free_monitor_x = np.linspace(-1.4, -0.2, 600)
    free_pml_density = x_probability(benchmark_pml, states_pml, free_monitor_x)
    free_ref_density = x_probability(benchmark_ref, states_ref, free_monitor_x)
    free_monitor_pml = np.trapezoid(free_pml_density, free_monitor_x, axis=1)
    free_monitor_ref = np.trapezoid(free_ref_density, free_monitor_x, axis=1)
    free_reflection_bound = np.maximum(free_monitor_pml - free_monitor_ref, 0.0)[post_passage].max()

    metrics = {
        "pml": {
            "start": settings.start,
            "thickness": settings.thickness,
            "outer_half_length": settings.outer_half_length,
            "order": settings.order,
            "target_reflection": settings.target_reflection,
            "probability_ratio_t_0_2": float(norm_pml[-1] / norm_pml[0]),
        },
        "large_box": {"probability_ratio_t_0_2": float(norm_ref[-1] / norm_ref[0])},
        "free_packet_reflection_bound_fraction": float(free_reflection_bound / free_monitor_pml[0]),
        "double_slit_reflection_bound_fraction": float(excess[pml_times >= 0.08].max() / monitor_pml[0]),
        "ly_density_error": {"5_vs_6": float(error_5_6), "6_vs_7": float(error_6_7)},
    }
    (FIGURES / "validation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
