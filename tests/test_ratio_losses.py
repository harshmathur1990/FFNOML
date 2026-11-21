"""The line-ratio and ionisation-stage terms of NLTECompositeLoss: |log10 n_u/n_l pred - true| over every line and
|log10 N_k/N_k+1 pred - true| over adjacent ionisation stages, per species in units of the species' mean std_Y. A common
shift of all levels costs neither; a shift of one level is measured on the lines that use it; a shift of one stage is
measured by the stage term alone when the levels of each stage keep their ratios; a one-stage species gives zero; both
count only with their weight and reach the predicted populations' gradients."""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loss.nlte_composite_loss import NLTECompositeLoss
from loss.weighted_mse_loss import WeightedMSE_L1

LEVELS = [3, 2]
LINES = [[[0, 1], [0, 2], [1, 2]], [[0, 1]]]
IONS = [[1, 1, 2], [2, 2]]                                    # species 0: two stages; species 1: one
MEAN_Y = np.array([12.0, 10.0, 9.0, 4.0, 3.0], np.float32)
STD_Y = np.array([1.0, 2.0, 3.0, 0.5, 1.5], np.float32)
MEAN_X = np.array([3.8, 0.0, 0.0, 0.0, 16.0, -7.0], np.float32)
STD_X = np.array([0.4, 5.0, 5.0, 5.0, 2.0, 2.5], np.float32)


def loss(lam_ratio=0.0, lam_ion=0.0, ions=IONS):
    return NLTECompositeLoss(
        chi=[[1.6e-18, 1.9e-18], [2.5e-19]], lines=LINES, wave=[[1216.0, 6563.0, 4861.0], [3934.0]],
        levels=LEVELS, data_loss_func=WeightedMSE_L1(), lam_ratio=lam_ratio, lam_ion=lam_ion, ions=ions,
        mean_X=MEAN_X, std_X=STD_X, mean_Y=MEAN_Y, std_Y=STD_Y)


def batch(seed=0):
    torch.manual_seed(seed)
    X = torch.randn(4, 6, 20)
    log10_true = torch.as_tensor(MEAN_Y)[None, :, None] + torch.randn(4, 5, 20)
    return X, log10_true


def normalised(log10_n):
    return (log10_n - torch.as_tensor(MEAN_Y)[None, :, None]) / torch.as_tensor(STD_Y)[None, :, None]


def run(pred, true, X, **kw):
    return loss(**kw)(X, normalised(pred), normalised(true), log_lte=torch.zeros(4, 5, 20))


class RatioLosses(unittest.TestCase):
    def test_common_shift_costs_nothing(self):
        X, true = batch()
        pred = true.clone()
        pred[:, 0:3] += 0.4
        pred[:, 3:5] -= 0.2
        _, comp = run(pred, true, X)
        self.assertLess(comp["ratio"].item(), 1e-5)
        self.assertLess(comp["ion"].item(), 1e-5)

    def test_one_level_shift_is_measured_on_its_lines(self):
        X, true = batch()
        pred = true.clone()
        pred[:, 1] += 0.3                                    # species 0 level 1: in lines (0,1) and (1,2), not (0,2)
        _, comp = run(pred, true, X)
        expected = 0.5 * ((2 / 3) * 0.3 / STD_Y[0:3].mean() + 0.0)
        self.assertAlmostEqual(comp["ratio"].item(), expected, places=5)

    def test_stage_shift_is_measured_by_the_stage_term(self):
        X, true = batch()
        pred = true.clone()
        pred[:, 0:2] += 0.5                                  # species 0, stage 1 up 0.5 dex, its level ratio kept
        _, comp = run(pred, true, X)
        expected = 0.5 * (0.5 / STD_Y[0:3].mean() + 0.0)     # species 1 has one stage: zero
        self.assertAlmostEqual(comp["ion"].item(), expected, places=5)

    def test_one_stage_species_and_default_ions(self):
        X, true = batch()
        pred = true + 0.2 * torch.randn_like(true)
        _, comp = run(pred, true, X, ions=None)              # every level one stage
        self.assertEqual(comp["ion"].item(), 0.0)
        with self.assertRaises(ValueError):
            loss(ions=[[1, 1], [2, 2]])

    def test_weights_and_default(self):
        X, true = batch()
        pred = true + 0.2 * torch.randn_like(true)
        L0, comp = run(pred, true, X)
        L1, _ = run(pred, true, X, lam_ratio=0.7, lam_ion=1.3)
        self.assertAlmostEqual(L0.item(), comp["data"].item(), places=6)
        self.assertAlmostEqual(L1.item() - L0.item(), 0.7 * comp["ratio"].item() + 1.3 * comp["ion"].item(), places=5)

    def test_gradient_reaches_predictions(self):
        X, true = batch()
        pred = normalised(true + 0.1 * torch.randn_like(true)).requires_grad_(True)
        L, _ = loss(1.0, 1.0)(X, pred, normalised(true), log_lte=torch.zeros(4, 5, 20))
        L0, _ = loss()(X, pred, normalised(true), log_lte=torch.zeros(4, 5, 20))
        (g,) = torch.autograd.grad(L - L0, pred)
        self.assertTrue(torch.isfinite(g).all())
        self.assertGreater(g.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
