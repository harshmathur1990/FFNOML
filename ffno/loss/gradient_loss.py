import torch
import torch.nn as nn


class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        dx_p = y_pred[..., :, 1:] - y_pred[..., :, :-1]
        dy_p = y_pred[..., 1:, :] - y_pred[..., :-1, :]

        dx_t = y_true[..., :, 1:] - y_true[..., :, :-1]
        dy_t = y_true[..., 1:, :] - y_true[..., :-1, :]

        return ((dx_p - dx_t) ** 2).mean() + ((dy_p - dy_t) ** 2).mean()
