"""The species-total term of NLTECompositeLoss: |log10 sum of levels pred - true| per species in units of the species'
mean std_Y. It is zero when only the split among levels is wrong, measures a common shift of all levels, counts only
when lam_total > 0, and reaches the predicted populations' gradients."""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loss.nlte_composite_loss import NLTECompositeLoss
from loss.weighted_mse_loss import WeightedMSE_L1

LEVELS = [3, 2]
MEAN_Y = np.array([12.0, 10.0, 9.0, 4.0, 3.0], np.float32)
STD_Y = np.array([1.0, 2.0, 3.0, 0.5, 1.5], np.float32)
MEAN_X = np.array([3.8, 0.0, 0.0, 0.0, 16.0, -7.0], np.float32)
STD_X = np.array([0.4, 5.0, 5.0, 5.0, 2.0, 2.5], np.float32)


def loss(lam_total):
    return NLTECompositeLoss(
        chi=[[1.6e-18, 1.9e-18], [2.5e-19]], lines=[[[0, 1], [1, 2]], [[0, 1]]], wave=[[1216.0, 6563.0], [3934.0]],
        levels=LEVELS, data_loss_func=WeightedMSE_L1(), lam_total=lam_total,
        mean_X=MEAN_X, std_X=STD_X, mean_Y=MEAN_Y, std_Y=STD_Y)


def batch(seed=0):
    torch.manual_seed(seed)
    X = torch.randn(4, 6, 20)
    log10_true = torch.as_tensor(MEAN_Y)[None, :, None] + torch.randn(4, 5, 20)
    return X, log10_true


def normalised(log10_n):
    return (log10_n - torch.as_tensor(MEAN_Y)[None, :, None]) / torch.as_tensor(STD_Y)[None, :, None]


def log10_sum(log10_n):
    return torch.log10(torch.sum(10.0 ** log10_n.double(), 1, keepdim=True)).float()


class SpeciesTotalLoss(unittest.TestCase):
    def test_composition_error_alone_costs_nothing(self):
        X, true = batch()
        pred = true.clone()
        for a, b in ((0, 3), (3, 5)):                       # each species: new random split, same total
            split = torch.randn(4, b - a, 20)
            split = split - log10_sum(split)
            pred[:, a:b] = log10_sum(true[:, a:b]) + split
        _, comp = loss(1.0)(X, normalised(pred), normalised(true), log_lte=torch.zeros(4, 5, 20))
        self.assertLess(comp["total"].item(), 1e-5)
        self.assertGreater(comp["data"].item(), 0.1)

    def test_common_shift_is_measured_in_mean_std(self):
        X, true = batch()
        pred = true.clone()
        pred[:, 0:3] += 0.3                                  # species 0 total off by 0.3 dex, species 1 exact
        _, comp = loss(1.0)(X, normalised(pred), normalised(true), log_lte=torch.zeros(4, 5, 20))
        expected = 0.5 * (0.3 / STD_Y[0:3].mean() + 0.0)
        self.assertAlmostEqual(comp["total"].item(), expected, places=5)

    def test_weight_and_default(self):
        X, true = batch()
        pred = true + 0.2 * torch.randn_like(true)
        args = (X, normalised(pred), normalised(true))
        L0, comp = loss(0.0)(*args, log_lte=torch.zeros(4, 5, 20))
        L2, _ = loss(2.0)(*args, log_lte=torch.zeros(4, 5, 20))
        self.assertAlmostEqual(L0.item(), comp["data"].item(), places=6)
        self.assertAlmostEqual(L2.item() - L0.item(), 2.0 * comp["total"].item(), places=5)

    def test_gradient_reaches_predictions(self):
        X, true = batch()
        pred = normalised(true + 0.1).requires_grad_(True)
        L, _ = loss(1.0)(X, pred, normalised(true), log_lte=torch.zeros(4, 5, 20))
        L_data, _ = loss(0.0)(X, pred, normalised(true), log_lte=torch.zeros(4, 5, 20))
        (g,) = torch.autograd.grad(L - L_data, pred)
        self.assertTrue(torch.isfinite(g).all())
        self.assertGreater(g.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
