

def compute_channel_stats(X):
    """
    X: [C, D, nx, ny]
    returns mean, std of shape [C]
    """
    C = X.shape[0]

    X_flat = X.reshape(C, -1)

    mean = X_flat.mean(axis=1)
    std = X_flat.std(axis=1)

    std = np.maximum(std, 1e-6)

    return mean.astype(np.float32), std.astype(np.float32)


def normalize_channels(X, mean, std):
    """
    X: [C, D, nx, ny]
    mean/std: [C]
    """
    return (X - mean[:, None, None, None]) / std[:, None, None, None]


def denormalize_channels(X, mean, std):
    return X * std[:, None, None, None] + mean[:, None, None, None]


def read_normalization(h5_path):
    with h5py.File(h5_path, "r") as f:
        if "mean_X" not in f:
            raise RuntimeError("Normalization stats not found in training file")

        mean_X = f["mean_X"][...]
        std_X  = f["std_X"][...]
        mean_Y = f["mean_Y"][...]
        std_Y  = f["std_Y"][...]

    return mean_X, std_X, mean_Y, std_Y
