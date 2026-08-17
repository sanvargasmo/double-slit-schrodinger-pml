"""Reference implementation of the Hamiltonian used in ``Hreal.ipynb``.

The original notebook uses normalized plane waves on ``[-Lx,Lx] x
[-Ly,Ly]``.  The barrier is a rectangle of thickness ``b`` with two apertures
of width ``a`` whose centres are separated by ``c``.  This intentionally
direct implementation is kept separate so tests can guard the cleaned model
against accidental changes to the legacy equations.
"""

from __future__ import annotations

import numpy as np


def _legacy_interval_element(km: float, kn: float, left: float, right: float, half_length: float) -> complex:
    q = kn - km
    if np.isclose(q, 0.0):
        return (right - left) / (2.0 * half_length)
    return (np.exp(1j * q * right) - np.exp(1j * q * left)) / (2j * half_length * q)


def legacy_hreal_matrix(
    *,
    Lx: float = 1.0,
    Ly: float = 1.0,
    a: float = 0.30,
    c: float = 0.40,
    b: float = 0.20,
    hbar: float = 1.0,
    mass: float = 1.0,
    u0: float = 1.0,
    nx: int = 6,
    ny: int = 6,
) -> np.ndarray:
    """Return the legacy dense Hamiltonian, using explicit index loops."""

    ix = np.arange(-nx, nx + 1)
    iy = np.arange(-ny, ny + 1)
    kx = np.pi * ix / Lx
    ky = np.pi * iy / Ly
    dim_y = iy.size
    size = ix.size * iy.size
    out = np.zeros((size, size), dtype=complex)

    slit_centres = (-c / 2.0, c / 2.0)
    for mx, kmx in enumerate(kx):
        for my, kmy in enumerate(ky):
            row = mx * dim_y + my
            for nx_i, knx in enumerate(kx):
                xel = _legacy_interval_element(kmx, knx, -b / 2.0, b / 2.0, Lx)
                for ny_i, kny in enumerate(ky):
                    col = nx_i * dim_y + ny_i
                    if row == col:
                        out[row, col] += hbar**2 * (knx**2 + kny**2) / (2.0 * mass)

                    full_y = 1.0 if my == ny_i else 0.0
                    apertures = 0.0j
                    for centre in slit_centres:
                        apertures += _legacy_interval_element(
                            kmy,
                            kny,
                            centre - a / 2.0,
                            centre + a / 2.0,
                            Ly,
                        )
                    out[row, col] += u0 * xel * (full_y - apertures)
    return out

