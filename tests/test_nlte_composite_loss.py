import numpy as np
import torch

from loss.nlte_composite_loss import compute_line_source_function_batched, kB


def test_source_exponent_reconstructs_departure_coefficients_from_populations():
    temperature = torch.tensor([[5000.0]], dtype=torch.float32)
    log10_lte = torch.tensor([[[10.0], [8.0]]], dtype=torch.float32)
    log10_departure = torch.tensor([[[0.2], [-0.1]]], dtype=torch.float32)
    log10_nlte = log10_lte + log10_departure

    chi = torch.tensor([2.0e-19], dtype=torch.float32)
    lines = torch.tensor([[0, 1]], dtype=torch.long)

    _, source_exponent = compute_line_source_function_batched(
        T=temperature,
        log10_nlte=log10_nlte,
        log10_lte=log10_lte,
        chi=chi,
        lines=lines,
        line_frequency=torch.ones(1),
        source_function_prefactor=torch.ones(1),
        return_source_exponent=True,
    )

    expected = (
        np.log(10.0) * (0.2 - (-0.1))
        + 2.0e-19 / (float(kB) * 5000.0)
    )
    torch.testing.assert_close(
        source_exponent, torch.tensor([[[expected]]], dtype=torch.float32)
    )


def test_chi_lookup_includes_zero_energy_ground_state():
    temperature = torch.tensor([[6000.0]], dtype=torch.float32)
    log10_lte = torch.zeros((1, 3, 1), dtype=torch.float32)
    log10_nlte = log10_lte.clone()
    chi_level_1 = 1.6e-18

    _, source_exponent = compute_line_source_function_batched(
        T=temperature,
        log10_nlte=log10_nlte,
        log10_lte=log10_lte,
        chi=torch.tensor([chi_level_1, 1.9e-18], dtype=torch.float32),
        lines=torch.tensor([[0, 1]], dtype=torch.long),
        line_frequency=torch.ones(1),
        source_function_prefactor=torch.ones(1),
        return_source_exponent=True,
    )

    expected = chi_level_1 / (float(kB) * 6000.0)
    torch.testing.assert_close(
        source_exponent, torch.tensor([[[expected]]], dtype=torch.float32)
    )
