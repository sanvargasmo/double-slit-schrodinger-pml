import numpy as np

from double_slit_pml.legacy_hreal import legacy_hreal_matrix
from double_slit_pml.model import (
    Geometry,
    PMLSettings,
    PlaneWaveModel,
    build_hamiltonian,
    compact_packet,
    evolve,
    interval_matrix,
    project_separable_state,
    reconstruct,
    sigma_and_derivative,
)


def test_full_interval_is_identity():
    matrix = interval_matrix(1.7, 5, -1.7, 1.7)
    np.testing.assert_allclose(matrix, np.eye(11), atol=2e-14)


def test_barrier_matrix_is_hermitian_without_pml():
    model = PlaneWaveModel(Lx=1.0, Ly=1.0, nx=3, ny=3)
    h = build_hamiltonian(model).dense()
    np.testing.assert_allclose(h, h.conj().T, atol=2e-13)


def test_clean_hamiltonian_matches_hreal_legacy_equations():
    model = PlaneWaveModel(Lx=1.0, Ly=1.0, nx=2, ny=2)
    clean = build_hamiltonian(model).dense()
    legacy = legacy_hreal_matrix(nx=2, ny=2)
    np.testing.assert_allclose(clean, legacy, atol=3e-13)


def test_pml_profile_is_exactly_zero_in_physical_region():
    settings = PMLSettings(start=1.5, thickness=2.0)
    x = np.linspace(-1.5, 1.5, 101)
    sigma, derivative = sigma_and_derivative(x, settings)
    np.testing.assert_array_equal(sigma, 0.0)
    np.testing.assert_array_equal(derivative, 0.0)


def test_pml_profile_reaches_design_strength():
    settings = PMLSettings(start=1.5, thickness=2.0, order=4, target_reflection=1e-3)
    sigma, _ = sigma_and_derivative(np.array([-3.5, 3.5]), settings)
    np.testing.assert_allclose(sigma, settings.sigma_max)
    assert settings.sigma_max > 0.0


def test_pml_hamiltonian_is_non_hermitian():
    settings = PMLSettings()
    model = PlaneWaveModel(Lx=settings.outer_half_length, Ly=3.0, nx=8, ny=2, pml=settings)
    h = build_hamiltonian(model).dense()
    assert np.linalg.norm(h - h.conj().T) > 1e-3


def test_initial_packet_has_half_unit_buffer_before_pml():
    settings = PMLSettings(start=1.5, thickness=2.0)
    x = np.linspace(-settings.start, settings.start, 2001)
    packet = compact_packet(x)
    support = x[np.abs(packet) > 0.0]
    assert support.min() > -1.0
    assert support.max() < -0.525
    assert support.min() - (-settings.start) > 0.49


def test_free_packet_reflection_is_below_point_zero_three_percent():
    settings = PMLSettings()
    free = Geometry(barrier_height=0.0)
    pml_model = PlaneWaveModel(Lx=3.5, Ly=1.0, nx=60, ny=0, geometry=free, pml=settings)
    ref_model = PlaneWaveModel(Lx=8.0, Ly=1.0, nx=137, ny=0, geometry=free)
    packet = lambda x: compact_packet(x, k0=22.0)
    pml_initial = project_separable_state(pml_model, packet)
    ref_initial = project_separable_state(ref_model, packet)
    times = np.linspace(0.0, 0.2, 6)
    pml_states = evolve(pml_model, pml_initial, times)
    ref_states = evolve(ref_model, ref_initial, times)

    x = np.linspace(-1.4, -0.2, 400)
    def monitor(model, states):
        values = []
        for state in states:
            psi = reconstruct(model, state, x, np.array([0.0]))[:, 0]
            values.append(2.0 * model.Ly * np.trapezoid(np.abs(psi) ** 2, x))
        return np.asarray(values)

    pml_monitor = monitor(pml_model, pml_states)
    ref_monitor = monitor(ref_model, ref_states)
    positive_excess = np.maximum(pml_monitor - ref_monitor, 0.0)
    reflected_fraction = positive_excess[times >= 0.08].max() / pml_monitor[0]
    assert reflected_fraction < 3.0e-4
