"""Factory adapter for the currently configured FFNO3D architecture."""

from models.ffno_model import FFNO3D


def create_ffno3d(Cin, Cout, device="cpu", **overrides):
    from config import MODEL_CONFIG

    configuration = dict(MODEL_CONFIG)
    configuration.update(overrides)
    configuration["in_channels"] = int(Cin)
    configuration["out_channels"] = int(Cout)
    # Device placement is owned by PersistentFFNOBackend after construction.
    return FFNO3D(**configuration)
