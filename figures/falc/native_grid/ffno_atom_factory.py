"""create_ffno3d for one atom's level-fraction checkpoint in a service that holds several (FALC H + Ca II run): the species
tables of `atom` from config.ATOM_CONFIG in place of those of FFNOML_ACTIVE_ATOMS. On PYTHONPATH via run_ca.sh."""
import config
from ffno_model_factory import create_ffno3d as _create


def create_ffno3d(Cin, Cout, device="cpu", atom="H", **overrides):
    species = [{k: config.ATOM_CONFIG[atom][k] for k in ("levels", "log10_particle_mass", "multi3d_lte")}]
    return _create(Cin, Cout, device=device, species=species, **overrides)
