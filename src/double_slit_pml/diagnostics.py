"""Diagnostics used by tests, notebooks, and README figures."""

from __future__ import annotations

import numpy as np

from .model import PlaneWaveModel, reconstruct


def density_on_grid(
    model: PlaneWaveModel,
    state: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    return np.abs(reconstruct(model, state, x, y)) ** 2


def integrate_rectangle(
    model: PlaneWaveModel,
    state: np.ndarray,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    points: tuple[int, int] = (220, 160),
) -> float:
    x = np.linspace(*x_bounds, points[0])
    y = np.linspace(*y_bounds, points[1])
    rho = density_on_grid(model, state, x, y)
    return float(np.trapezoid(np.trapezoid(rho, y, axis=1), x))


def relative_density_error(
    model_a: PlaneWaveModel,
    state_a: np.ndarray,
    model_b: PlaneWaveModel,
    state_b: np.ndarray,
    x_bounds: tuple[float, float] = (-1.0, 1.0),
    y_bounds: tuple[float, float] = (-1.0, 1.0),
    points: tuple[int, int] = (180, 140),
) -> float:
    x = np.linspace(*x_bounds, points[0])
    y = np.linspace(*y_bounds, points[1])
    rho_a = density_on_grid(model_a, state_a, x, y)
    rho_b = density_on_grid(model_b, state_b, x, y)
    denominator = np.linalg.norm(rho_b.ravel())
    return float(np.linalg.norm((rho_a - rho_b).ravel()) / denominator)

