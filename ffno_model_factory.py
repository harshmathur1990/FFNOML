"""Factory adapter: FFNO3D from config.MODEL_CONFIG, the final model."""

from models.ffno_model import FFNO3D


def create_ffno3d(Cin, Cout, device="cpu", atoms=None, **overrides):
    """atoms (e.g. ["CA"]): the species of the level-fraction head, in place of FFNOML_ACTIVE_ATOMS, so that one
    service process can serve an H and a Ca II model."""
    from config import ATOM_CONFIG, MODEL_CONFIG

    configuration = dict(MODEL_CONFIG)
    configuration.update(overrides)
    if atoms is not None:
        configuration["species"] = [
            {k: ATOM_CONFIG[a][k] for k in ("levels", "log10_particle_mass", "multi3d_lte")} for a in atoms]
    configuration["in_channels"] = int(Cin)
    configuration["out_channels"] = int(Cout)
    # Device placement is owned by PersistentFFNOBackend after construction.
    return FFNO3D(**configuration)
