"""Spectral Galerkin model for the double-slit/PML comparison.

The wave function is expanded in a global, orthonormal plane-wave basis.  The
PML is introduced by complex coordinate stretching only in x,

    d/dx -> (1/s) d/dx,  s(x) = 1 + i sigma(x),

which gives ``(1/s**2)d2/dx2 - (s'/s**3)d/dx``.  The y direction remains a
larger periodic plane-wave box and is checked independently for convergence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from scipy.sparse.linalg import LinearOperator, expm_multiply


@dataclass(frozen=True)
class Geometry:
    """Double-slit geometry in the dimensionless units of ``Hreal``."""

    slit_width: float = 0.30
    slit_separation: float = 0.40
    barrier_thickness: float = 0.20
    barrier_height: float = 1.0


@dataclass(frozen=True)
class PMLSettings:
    """One-directional polynomial PML."""

    start: float = 1.50
    thickness: float = 2.00
    order: int = 4
    target_reflection: float = 1.0e-3

    @property
    def outer_half_length(self) -> float:
        return self.start + self.thickness

    @property
    def sigma_max(self) -> float:
        return -(self.order + 1) * np.log(self.target_reflection) / (2.0 * self.thickness)


@dataclass(frozen=True)
class PlaneWaveModel:
    """Numerical and physical parameters of one spectral calculation."""

    Lx: float
    Ly: float
    nx: int
    ny: int
    geometry: Geometry = Geometry()
    hbar: float = 1.0
    mass: float = 1.0
    pml: PMLSettings | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return 2 * self.nx + 1, 2 * self.ny + 1

    @property
    def size(self) -> int:
        sx, sy = self.shape
        return sx * sy


def wave_numbers(half_length: float, modes: int) -> np.ndarray:
    return np.pi * np.arange(-modes, modes + 1) / half_length


def interval_matrix(half_length: float, modes: int, left: float, right: float) -> np.ndarray:
    """Plane-wave matrix of the indicator of ``[left,right]``."""

    k = wave_numbers(half_length, modes)
    q = k[None, :] - k[:, None]
    out = np.empty_like(q, dtype=complex)
    zero = np.isclose(q, 0.0)
    out[zero] = (right - left) / (2.0 * half_length)
    out[~zero] = (
        np.exp(1j * q[~zero] * right) - np.exp(1j * q[~zero] * left)
    ) / (2j * half_length * q[~zero])
    return out


def multiplication_matrix(
    func: Callable[[np.ndarray], np.ndarray],
    half_length: float,
    modes: int,
    quadrature_points: int | None = None,
) -> np.ndarray:
    """Plane-wave matrix of a bounded periodic multiplication function."""

    qn = quadrature_points or max(2048, 32 * (2 * modes + 1))
    x = np.linspace(-half_length, half_length, qn, endpoint=False)
    values = np.asarray(func(x), dtype=complex)
    # Only 4*modes+1 Fourier coefficients are distinct.  Computing those first
    # avoids allocating a large (basis x basis x quadrature) temporary array.
    deltas = np.arange(-2 * modes, 2 * modes + 1)
    q = np.pi * deltas / half_length
    coefficients = np.mean(values[None, :] * np.exp(1j * q[:, None] * x), axis=1)
    indices = np.arange(-modes, modes + 1)
    difference = indices[None, :] - indices[:, None]
    return coefficients[difference + 2 * modes]


def sigma_and_derivative(x: np.ndarray, settings: PMLSettings) -> tuple[np.ndarray, np.ndarray]:
    """Return the even PML profile and its analytic derivative."""

    distance = np.maximum(np.abs(x) - settings.start, 0.0)
    scaled = np.minimum(distance / settings.thickness, 1.0)
    sigma = settings.sigma_max * scaled**settings.order
    derivative = np.zeros_like(x, dtype=float)
    active = (distance > 0.0) & (distance < settings.thickness)
    derivative[active] = (
        settings.sigma_max
        * settings.order
        * scaled[active] ** (settings.order - 1)
        * np.sign(x[active])
        / settings.thickness
    )
    return sigma, derivative


def double_slit_factors(model: PlaneWaveModel) -> tuple[np.ndarray, np.ndarray]:
    """Return the x barrier and y opaque-screen plane-wave matrices."""

    g = model.geometry
    x_barrier = interval_matrix(model.Lx, model.nx, -g.barrier_thickness / 2, g.barrier_thickness / 2)
    identity_y = np.eye(2 * model.ny + 1, dtype=complex)
    apertures = np.zeros_like(identity_y)
    for centre in (-g.slit_separation / 2, g.slit_separation / 2):
        apertures += interval_matrix(
            model.Ly,
            model.ny,
            centre - g.slit_width / 2,
            centre + g.slit_width / 2,
        )
    return x_barrier, identity_y - apertures


class SeparableHamiltonian(LinearOperator):
    """Matrix-free Kronecker representation of the 2D Hamiltonian."""

    def __init__(
        self,
        tx: np.ndarray,
        ty: np.ndarray,
        x_barrier: np.ndarray,
        y_screen: np.ndarray,
        barrier_height: float,
    ) -> None:
        self.tx = np.asarray(tx, dtype=complex)
        self.ty = np.asarray(ty, dtype=complex)
        self.x_barrier = np.asarray(x_barrier, dtype=complex)
        self.y_screen = np.asarray(y_screen, dtype=complex)
        self.barrier_height = float(barrier_height)
        self.grid_shape = (self.tx.shape[0], self.ty.shape[0])
        size = self.grid_shape[0] * self.grid_shape[1]
        super().__init__(dtype=np.dtype(complex), shape=(size, size))

    def _apply(self, vector: np.ndarray, adjoint: bool = False) -> np.ndarray:
        psi = np.asarray(vector).reshape(self.grid_shape)
        if adjoint:
            out = self.tx.conj().T @ psi + psi @ self.ty.conj()
            out += self.barrier_height * self.x_barrier.conj().T @ psi @ self.y_screen.conj()
        else:
            out = self.tx @ psi + psi @ self.ty.T
            out += self.barrier_height * self.x_barrier @ psi @ self.y_screen.T
        return np.asarray(out).reshape(-1)

    def _matvec(self, vector: np.ndarray) -> np.ndarray:
        return self._apply(vector, adjoint=False)

    def _rmatvec(self, vector: np.ndarray) -> np.ndarray:
        return self._apply(vector, adjoint=True)

    def _matmat(self, matrix: np.ndarray) -> np.ndarray:
        return np.column_stack([self._matvec(matrix[:, j]) for j in range(matrix.shape[1])])

    def _rmatmat(self, matrix: np.ndarray) -> np.ndarray:
        return np.column_stack([self._rmatvec(matrix[:, j]) for j in range(matrix.shape[1])])

    @property
    def trace_value(self) -> complex:
        sx, sy = self.grid_shape
        return (
            sy * np.trace(self.tx)
            + sx * np.trace(self.ty)
            + self.barrier_height * np.trace(self.x_barrier) * np.trace(self.y_screen)
        )

    def dense(self) -> np.ndarray:
        """Materialize small models (intended for tests and legacy checks)."""

        ix = np.eye(self.grid_shape[0], dtype=complex)
        iy = np.eye(self.grid_shape[1], dtype=complex)
        return (
            np.kron(self.tx, iy)
            + np.kron(ix, self.ty)
            + self.barrier_height * np.kron(self.x_barrier, self.y_screen)
        )


def build_hamiltonian(model: PlaneWaveModel) -> SeparableHamiltonian:
    """Build the real or PML Hamiltonian without changing the plane-wave basis."""

    kx = wave_numbers(model.Lx, model.nx)
    ky = wave_numbers(model.Ly, model.ny)
    ty = np.diag(model.hbar**2 * ky**2 / (2.0 * model.mass)).astype(complex)

    if model.pml is None:
        tx = np.diag(model.hbar**2 * kx**2 / (2.0 * model.mass)).astype(complex)
    else:
        if not np.isclose(model.Lx, model.pml.outer_half_length):
            raise ValueError("For the validated PML, Lx must equal start + thickness")

        def a_function(x: np.ndarray) -> np.ndarray:
            sigma, _ = sigma_and_derivative(x, model.pml)
            return 1.0 / (1.0 + 1j * sigma) ** 2

        def b_function(x: np.ndarray) -> np.ndarray:
            sigma, sigma_prime = sigma_and_derivative(x, model.pml)
            return (1j * sigma_prime) / (1.0 + 1j * sigma) ** 3

        a_matrix = multiplication_matrix(a_function, model.Lx, model.nx)
        b_matrix = multiplication_matrix(b_function, model.Lx, model.nx)
        # -hbar^2/(2m) [a d_xx - b d_x]
        tx = model.hbar**2 / (2.0 * model.mass) * (
            a_matrix * (kx[None, :] ** 2) + 1j * b_matrix * kx[None, :]
        )

    x_barrier, y_screen = double_slit_factors(model)
    return SeparableHamiltonian(tx, ty, x_barrier, y_screen, model.geometry.barrier_height)


def compact_packet(x: np.ndarray, left: float = -1.0, right: float = -0.525, k0: float = 18.0) -> np.ndarray:
    """Smooth compact packet with a buffer before the PML interface."""

    out = np.zeros_like(x, dtype=complex)
    active = (x > left) & (x < right)
    phase = (x[active] - left) / (right - left)
    out[active] = np.sin(np.pi * phase) ** 2 * np.exp(1j * k0 * x[active])
    return out


def project_1d(
    func: Callable[[np.ndarray], np.ndarray],
    half_length: float,
    modes: int,
    quadrature_points: int = 8192,
) -> np.ndarray:
    x = np.linspace(-half_length, half_length, quadrature_points, endpoint=False)
    values = np.asarray(func(x), dtype=complex)
    k = wave_numbers(half_length, modes)
    basis_conjugate = np.exp(-1j * k[:, None] * x) / np.sqrt(2.0 * half_length)
    return (2.0 * half_length / quadrature_points) * (basis_conjugate @ values)


def project_separable_state(
    model: PlaneWaveModel,
    x_function: Callable[[np.ndarray], np.ndarray],
    y_function: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Project a separable state using the same domain and measure as H."""

    if y_function is None:
        y_function = lambda y: np.ones_like(y, dtype=complex) / np.sqrt(2.0 * model.Ly)
    cx = project_1d(x_function, model.Lx, model.nx)
    cy = project_1d(y_function, model.Ly, model.ny)
    state = np.outer(cx, cy).reshape(-1)
    norm = np.linalg.norm(state)
    if norm == 0.0:
        raise ValueError("The projected state is zero; increase the spectral cutoff")
    return state / norm


def evolve(
    model: PlaneWaveModel,
    state0: np.ndarray,
    times: Iterable[float],
    hamiltonian: SeparableHamiltonian | None = None,
) -> np.ndarray:
    """Propagate with the matrix exponential at an evenly spaced set of times."""

    times_array = np.asarray(list(times), dtype=float)
    if times_array.ndim != 1 or times_array.size == 0:
        raise ValueError("times must be a non-empty one-dimensional sequence")
    if times_array.size == 1:
        return np.asarray(state0, dtype=complex)[None, :]
    spacing = np.diff(times_array)
    if not np.allclose(spacing, spacing[0]):
        raise ValueError("expm_multiply requires evenly spaced output times")

    h = hamiltonian or build_hamiltonian(model)
    generator = (-1j / model.hbar) * h
    return expm_multiply(
        generator,
        np.asarray(state0, dtype=complex),
        start=float(times_array[0]),
        stop=float(times_array[-1]),
        num=times_array.size,
        endpoint=True,
        traceA=(-1j / model.hbar) * h.trace_value,
    )


def basis_values(points: np.ndarray, half_length: float, modes: int) -> np.ndarray:
    k = wave_numbers(half_length, modes)
    return np.exp(1j * points[:, None] * k[None, :]) / np.sqrt(2.0 * half_length)


def reconstruct(model: PlaneWaveModel, state: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Reconstruct ``psi(x,y)`` from plane-wave coefficients."""

    bx = basis_values(np.asarray(x), model.Lx, model.nx)
    by = basis_values(np.asarray(y), model.Ly, model.ny)
    coefficients = np.asarray(state).reshape(model.shape)
    return bx @ coefficients @ by.T
