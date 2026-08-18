"""Create clean, output-free notebooks from version-controlled cell sources."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
NOTEBOOKS.mkdir(exist_ok=True)


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


hreal = notebook(
    [
        markdown(
            """# Hreal: double slit without PML

This notebook preserves the normalized plane-wave Hamiltonian of the original
`Hreal` calculation.  Its x and y boundaries are periodic because the basis is
global.  The helper module keeps the original equations independently so the
clean implementation can be regression-tested against them."""
        ),
        code(
            """import numpy as np
import matplotlib.pyplot as plt

from double_slit_pml.model import (
    PlaneWaveModel, build_hamiltonian, compact_packet,
    evolve, project_separable_state, reconstruct,
)
from double_slit_pml.legacy_hreal import legacy_hreal_matrix"""
        ),
        code(
            """# Original Hreal parameters
model = PlaneWaveModel(Lx=1.0, Ly=1.0, nx=6, ny=6)
H = build_hamiltonian(model)

# Exact algebraic regression against the preserved notebook expression
np.max(np.abs(H.dense() - legacy_hreal_matrix(nx=6, ny=6)))"""
        ),
        code(
            """eigenvalues = np.linalg.eigvalsh(H.dense())[:10]
eigenvalues"""
        ),
        code(
            """psi0 = project_separable_state(model, lambda x: compact_packet(x, k0=12.0))
times = np.linspace(0.0, 0.15, 4)
states = evolve(model, psi0, times, H)

x = np.linspace(-1.0, 1.0, 240)
y = np.linspace(-1.0, 1.0, 180)
fig, axes = plt.subplots(1, len(times), figsize=(12, 3), sharex=True, sharey=True)
for ax, state, time in zip(axes, states, times):
    density = np.abs(reconstruct(model, state, x, y))**2
    ax.imshow(density.T, origin='lower', extent=(-1, 1, -1, 1), aspect='auto', cmap='magma')
    ax.set_title(f't = {time:.2f}')
    ax.set_xlabel('x')
axes[0].set_ylabel('y')
fig.suptitle('Hreal: periodic plane-wave box, no PML')
plt.tight_layout()"""
        ),
    ]
)


pml = notebook(
    [
        markdown(
            """# Untitled28 PML: validated x-only absorbing layer

The global plane-wave basis is retained.  Complex coordinate stretching is
applied only in x, beginning at `|x| = 1.5`; the physical region
`|x|, |y| <= 1` is unchanged.  A compact incident packet ends at `x = -0.525`,
leaving a 0.5-unit buffer before the left PML.  The y direction uses a larger
periodic box and is validated separately by convergence."""
        ),
        code(
            """import numpy as np
import matplotlib.pyplot as plt

from double_slit_pml.model import (
    PMLSettings, PlaneWaveModel, compact_packet, evolve,
    project_separable_state, reconstruct, sigma_and_derivative,
)
from double_slit_pml.diagnostics import integrate_rectangle, relative_density_error"""
        ),
        code(
            """pml = PMLSettings(start=1.5, thickness=2.0, order=4, target_reflection=1e-3)
model = PlaneWaveModel(Lx=pml.outer_half_length, Ly=6.0, nx=80, ny=30, pml=pml)

x_plot = np.linspace(-model.Lx, model.Lx, 1200)
sigma, _ = sigma_and_derivative(x_plot, pml)
plt.plot(x_plot, sigma)
plt.axvspan(-1, 1, alpha=.12, label='region of interest')
plt.xlabel('x'); plt.ylabel(r'$\\sigma(x)$'); plt.legend();"""
        ),
        code(
            """psi0 = project_separable_state(model, lambda x: compact_packet(x, k0=30.0))
times = np.linspace(0.0, 0.2, 21)
states = evolve(model, psi0, times)
probability = np.sum(np.abs(states)**2, axis=1)

plt.plot(times, probability)
plt.xlabel('time'); plt.ylabel('total probability');
probability[-1] / probability[0]"""
        ),
        code(
            """# Density in the physical window.  The PML itself is outside this plot.
x = np.linspace(-1.0, 1.0, 240)
y = np.linspace(-1.0, 1.0, 180)
density = np.abs(reconstruct(model, states[-1], x, y))**2
plt.imshow(density.T, origin='lower', extent=(-1, 1, -1, 1), aspect='auto', cmap='magma')
plt.xlabel('x'); plt.ylabel('y'); plt.colorbar(label=r'$|\\psi|^2$');"""
        ),
        markdown(
            """## Reproducible validation

Run `python scripts/generate_figures.py` from the repository root.  It compares
this calculation with a conservative `Lx = 8` box, evaluates the upstream
reflection monitor, and repeats the physical-window density calculation for
`Ly = 5, 6, 7`.  The machine-readable results are written to
`figures/validation_metrics.json`."""
        ),
    ]
)


parameter_explorer = notebook(
    [
        markdown(
            """# Parameter Explorer

Edit the next cell to test a new double-slit or PML configuration. The core
Hamiltonian code does not need to be modified. Set `USE_PML = False` to run a
periodic box without an absorbing layer."""
        ),
        code(
            """import numpy as np
import matplotlib.pyplot as plt

from double_slit_pml.model import (
    Geometry, PMLSettings, PlaneWaveModel, compact_packet,
    evolve, project_separable_state, reconstruct,
)"""
        ),
        markdown("## Parameters — edit this cell"),
        code(
            """# Double-slit geometry
SLIT_WIDTH = 0.30
SLIT_SEPARATION = 0.40
BARRIER_THICKNESS = 0.20
BARRIER_HEIGHT = 1.0

# PML and box
USE_PML = True
PML_START = 1.50
PML_THICKNESS = 2.00
PML_ORDER = 4
TARGET_REFLECTION = 1e-3
LY = 6.0
NX = 80
NY = 30

# Incident packet and time interval
PACKET_LEFT = -1.0
PACKET_RIGHT = -0.525
K0 = 30.0
T_FINAL = 0.20
NUMBER_OF_SNAPSHOTS = 21

# Visualization
NUMBER_OF_PLOTTED_SNAPSHOTS = 9
PLOT_COLUMNS = 3
DENSITY_NORMALIZATION = 'physical'  # 'physical', 'integral', or 'absolute'
VIEW_X_MIN, VIEW_X_MAX = -1.0, 1.45
VIEW_Y_MIN, VIEW_Y_MAX = -1.5, 1.5"""
        ),
        code(
            """geometry = Geometry(
    slit_width=SLIT_WIDTH,
    slit_separation=SLIT_SEPARATION,
    barrier_thickness=BARRIER_THICKNESS,
    barrier_height=BARRIER_HEIGHT,
)
pml = PMLSettings(
    start=PML_START,
    thickness=PML_THICKNESS,
    order=PML_ORDER,
    target_reflection=TARGET_REFLECTION,
)
model = PlaneWaveModel(
    Lx=pml.outer_half_length,
    Ly=LY,
    nx=NX,
    ny=NY,
    geometry=geometry,
    pml=pml if USE_PML else None,
)
psi0 = project_separable_state(
    model,
    lambda x: compact_packet(x, left=PACKET_LEFT, right=PACKET_RIGHT, k0=K0),
)
times = np.linspace(0.0, T_FINAL, NUMBER_OF_SNAPSHOTS)
states = evolve(model, psi0, times)
probability = np.sum(np.abs(states)**2, axis=1)
probability[-1] / probability[0]"""
        ),
        code(
            """x = np.linspace(VIEW_X_MIN, VIEW_X_MAX, 300)
y = np.linspace(VIEW_Y_MIN, VIEW_Y_MAX, 190)
indices = np.unique(np.rint(np.linspace(
    0, len(times) - 1, NUMBER_OF_PLOTTED_SNAPSHOTS
)).astype(int))

raw_densities = [
    np.abs(reconstruct(model, states[index], x, y))**2
    for index in indices
]
physical_x = np.abs(x) <= PML_START if USE_PML else np.ones_like(x, dtype=bool)
physical_probability = np.asarray([
    np.trapezoid(np.trapezoid(density[physical_x, :], y, axis=1), x[physical_x])
    for density in raw_densities
])
if physical_probability[0] <= np.finfo(float).eps:
    raise ValueError('The initial state has no probability in the displayed physical region')
physical_probability_ratio = physical_probability / physical_probability[0]
physical_peak_density = np.asarray([
    np.max(density[physical_x, :]) for density in raw_densities
])

if DENSITY_NORMALIZATION == 'physical':
    floor = max(np.finfo(float).eps, 1e-12 * physical_peak_density[0])
    densities = [
        density / peak if peak > floor else np.zeros_like(density)
        for density, peak in zip(raw_densities, physical_peak_density)
    ]
    colorbar_label = r'$|\\psi|^2/\\rho_{\\max,\\rm phys}(t)$'
    vmax = 1.0
elif DENSITY_NORMALIZATION == 'integral':
    floor = max(np.finfo(float).eps, 1e-12 * physical_probability[0])
    densities = [
        density / p_phys if p_phys > floor else np.zeros_like(density)
        for density, p_phys in zip(raw_densities, physical_probability)
    ]
    colorbar_label = r'$|\\psi|^2/P_{\\rm phys}(t)$'
    vmax = max(np.quantile(density, 0.995) for density in densities)
elif DENSITY_NORMALIZATION == 'absolute':
    densities = raw_densities
    colorbar_label = r'$|\\psi|^2$'
    vmax = max(np.quantile(density, 0.995) for density in densities)
else:
    raise ValueError("DENSITY_NORMALIZATION must be 'physical', 'integral', or 'absolute'")

columns = min(PLOT_COLUMNS, len(indices))
rows = int(np.ceil(len(indices) / columns))
fig, axes = plt.subplots(
    rows, columns, figsize=(3.8 * columns, 3.2 * rows),
    sharex=True, sharey=True, squeeze=False, constrained_layout=True,
)
image = None
for position, (ax, index, density) in enumerate(zip(axes.flat, indices, densities)):
    image = ax.imshow(
        density.T, origin='lower', extent=(x.min(), x.max(), y.min(), y.max()),
        aspect='auto', cmap='magma', vmin=0.0, vmax=vmax,
    )
    ax.set_title(
        rf'$t={times[index]:.3f}$   '
        rf'$\\rho_{{\\max,\\rm phys}}={physical_peak_density[position]:.3g}$',
        fontsize=10,
    )
    ax.set_xlabel('x')
    if position % columns == 0:
        ax.set_ylabel('y')
for ax in axes.flat[len(indices):]:
    ax.axis('off')
fig.colorbar(image, ax=axes.ravel().tolist(), label=colorbar_label, shrink=0.85)

plt.figure(figsize=(7, 3))
plt.plot(times, probability / probability[0], label=r'$P_{\\rm total}(t)/P_{\\rm total}(0)$')
plt.plot(
    times[indices], physical_probability_ratio, marker='o',
    label=r'$P_{\\rm phys}(t)/P_{\\rm phys}(0)$',
)
plt.xlabel('time')
plt.ylabel('retained probability')
plt.legend()
plt.tight_layout();"""
        ),
        markdown(
            """For automated parameter studies, use `scripts/run_experiment.py`.
Run `python scripts/run_experiment.py --help` to list all command-line options."""
        ),
    ]
)


for name, content in (
    ("Hreal.ipynb", hreal),
    ("Untitled28_PML.ipynb", pml),
    ("Parameter_Explorer.ipynb", parameter_explorer),
):
    (NOTEBOOKS / name).write_text(json.dumps(content, indent=1) + "\n", encoding="utf-8")
