import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WeightedMSE(nn.Module):
    """
    Weighted MSE loss where weights depend only on the target magnitude.
    Designed for NLTE departure coefficients (log b).

    Loss = mean( (1 + |y_true|) * (y_pred - y_true)^2 )
    """
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        w = 1.0 + torch.abs(y_true)
        return (w * (y_pred - y_true) ** 2).mean()


class RelativeMSE(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        return (((y_pred - y_true) / (torch.abs(y_true) + self.eps))**2).mean()

