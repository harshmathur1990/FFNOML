import numpy as np
import os


def _build_split_tag(split_dict):
    parts = []

    for sim in sorted(split_dict):
        snaps = "-".join(sorted(split_dict[sim], key=str))
        parts.append(f"{sim}_{snaps}")

    return "__".join(parts)


def _active_atoms_tag(active_atoms):
    return "-".join(active_atoms)


def _dataset_filename(split_name, split_dict, patch, stride, active_atoms):
    split_tag = _build_split_tag(split_dict)
    atoms_tag = _active_atoms_tag(active_atoms)
    return f"3D_sim_{split_name}_{split_tag}_atoms{atoms_tag}_patch{patch}_stride{stride}.hdf5"


def build_multi3d_entries(split_dict):

    data = []

    for sim, snaps in split_dict.items():

        base = SIMULATIONS[sim]["base_path"]

        for snap in snaps:

            if snap not in SIMULATIONS[sim]["snaps"]:
                raise ValueError(
                    f"{snap} not listed in SIMULATIONS[{sim}]"
                )

            entry = {
                "MULTI3D_PATHS": [],
                "MULTI3D_ATMOS": f"{base}/{snap}/atm3d",
                "MESH": f"{base}/{snap}/mesh",
            }

            for atom in ACTIVE_ATOMS:
                atom_dir = ATOM_CONFIG[atom]["subdir"]

                entry["MULTI3D_PATHS"].append(
                    f"{base}/{snap}/{atom_dir}"
                )

            data.append(entry)

    return data

SIMULATIONS = {
    "en024048_hion": {
        "base_path": "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion",
        "snaps": ["385", "386", "465", "700"],
    },
    "nw012023": {
        "base_path": "/mn/stornext/d9/data/harshm/bifrost_data/nw012023",
        "snaps": ["1050", "1120", "915", "940"],
    },
    "ch012012_hion": {
        "base_path": "/mn/stornext/d9/data/harshm/bifrost_data/ch012012_hion",
        "snaps": ["759", "834", "910", "984"]
    },
    "ch012006": {
        "base_path": "/mn/stornext/d9/data/harshm/bifrost_data/ch012006",
        "snaps": ["795", "820", "836", "849"]
    },
    "qs006003_sap": {
        "base_path": "/mn/stornext/d9/data/harshm/bifrost_data/qs006003_sap",
        "snaps": ["1100", "1297", "689", "900"]
    }
}

TRAIN_SPLIT = {
    "en024048_hion": ["465"],
    "nw012023": ["915"],
    "ch012012_hion": ["910"],
    "ch012006": ["836"],
    "qs006003_sap": ["689"]
}

VAL_SPLIT = {
    "en024048_hion": ["700"],
    "nw012023": ["940"],
    "ch012012_hion": ["984"],
    "ch012006": ["849"],
    "qs006003_sap": ["900"]
}


ATOM_CONFIG = {
    "H": {
        "subdir": "H",

        "lines": np.array(
            [(0,1),(0,2),(0,3),(0,4),
             (1,2),(1,3),(1,4),
             (2,3),(2,4),(3,4)],
            dtype=np.float32
        ),

        "wave": np.array(
            [1215.6701,1025.7220,972.53650,949.74287,
             6562.79,4861.35,4340.472,
             18750,12820,40510],
            dtype=np.float32
        ),

        "chi": np.array([
            1.6339941854018686e-18,
            1.936585907218822e-18,
            2.0424878450955273e-18,
            2.091506177644877e-18,
            2.1802152677122893e-18
        ], dtype=np.float32),

        "levels": 6
    },

    "CA": {
        "subdir": "CA",

        "lines": np.array(
            [(0,3),(0,4),(1,3),(1,4),(2,4)],
            dtype=np.float32
        ),

        "wave": np.array(
            [3968.47,3933.66,8662.14,8498.02,8542.09],
            dtype=np.float32
        ),

        "chi": np.array([
            2.7115478588655445e-19,
            2.7235960502783243e-19,
            5.004162033597224e-19,
            5.048445871090645e-19,
            1.902059054757628e-18
        ], dtype=np.float32),

        "levels": 6
    }
}

ACTIVE_ATOMS = ["H"]
MULTI3D_TRAIN_DATA = build_multi3d_entries(TRAIN_SPLIT)
MULTI3D_VAL_DATA   = build_multi3d_entries(VAL_SPLIT)

atom_names = ACTIVE_ATOMS
lines = [ATOM_CONFIG[a]["lines"] for a in ACTIVE_ATOMS]
wave  = [ATOM_CONFIG[a]["wave"]  for a in ACTIVE_ATOMS]
chi   = [ATOM_CONFIG[a]["chi"]   for a in ACTIVE_ATOMS]
levels = [ATOM_CONFIG[a]["levels"]   for a in ACTIVE_ATOMS]


PRED_OLIVIA_DIR = "/cluster/work/projects/nn2834k/harshm/"
PRED_EAGLE7_DIR = "/mn/stornext/d9/data/harshm/"

# PRED_DIR = PRED_OLIVIA_DIR
PRED_DIR = PRED_EAGLE7_DIR

MULTI3D_PRED_DATA = [
    {
        "MULTI3D_ATMOS": PRED_DIR + "bifrost_data/en024048_hion/385/atm3d",
        "MESH":  PRED_DIR + "bifrost_data/en024048_hion/385/mesh",
        "NAME": "en024048_hion_385"
    },
    {
        "MULTI3D_ATMOS": PRED_DIR + "bifrost_data/nw012023/1050/atm3d",
        "MESH":  PRED_DIR + "bifrost_data/nw012023/1050/mesh",
        "NAME": "nw012023_1050"
    },
    {
        "MULTI3D_ATMOS": PRED_DIR + "bifrost_data/ch024031_by200bz005/450/atm3d",
        "MESH":  PRED_DIR + "bifrost_data/ch024031_by200bz005/450/mesh",
        "NAME": "ch024031_by200bz005_450"
    },
    {
        "MULTI3D_ATMOS": PRED_DIR + "bifrost_data/en024031_by100_helium/109/atm3d",
        "MESH":  PRED_DIR + "bifrost_data/en024031_by100_helium/109/mesh",
        "NAME": "en024031_by100_helium_109"
    },
    {
        "MULTI3D_ATMOS": PRED_DIR + "bifrost_data/en024048_hion_504/385/atm3d",
        "MESH":  PRED_DIR + "bifrost_data/en024048_hion_504/385/mesh",
        "NAME": "en024048_hion_504_385"
    },
    {
        "MULTI3D_ATMOS": PRED_DIR + "bifrost_data/nw012023_512/1050/atm3d",
        "MESH":  PRED_DIR + "bifrost_data/nw012023_512/1050/mesh",
        "NAME": "nw012023_512_1050"
    },
    {
        "MULTI3D_ATMOS": PRED_DIR + "bifrost_data/ch024031_by200bz005_768/450/atm3d",
        "MESH":  PRED_DIR + "bifrost_data/ch024031_by200bz005_768/450/mesh",
        "NAME": "ch024031_by200bz005_768_450"
    },
    {
        "MULTI3D_ATMOS": PRED_DIR + "bifrost_data/en024031_by100_helium_768/109/atm3d",
        "MESH":  PRED_DIR + "bifrost_data/en024031_by100_helium_768/109/mesh",
        "NAME": "en024031_by100_helium_768_109"
    }
]

MULTI_GPU = True

MODEL = "FFNO3D"

MODEL_CONFIG = dict(
    in_channels=6,
    out_channels=6,
    width=48,
    n_layers=4,
    dropout=0.0,
    spec_dropout=0.03,
    vertical_dropout=0.08,
    spec_dropout_layers=[0, 1, 2, 3],
    vertical_dropout_layers=[0, 1, 2, 3],
    checkpoint_blocks=True,
)


IODIR = "IO/"

MODEL_DIR = f"training_{MODEL}_zscale_expand/"
MODEL_FILE = MODEL_DIR + "3D_sim_train_s123.pt"

for pred_atmos in MULTI3D_PRED_DATA:
    name_parts = pred_atmos["NAME"].split("_")
    pred_atmos["SIM_NAME"] = "_".join(name_parts[:-1])
    pred_atmos["SNAP"] = name_parts[-1]
    pred_atmos["TRAIN_DIR"] = MODEL_DIR.rstrip("/")

EXPAND_FROM_CHECKPOINT = "training_FFNO3D_zscale/3D_sim_train_s123.pt"
ZERO_INIT_NEW_BLOCKS = True
FORCE_EXPAND_VALIDATION_BASELINE = True


DEBUG_LOSS = False
DEBUG_DIST_INFERENCE = False

# Dataset Params
PATCH=40
STRIDE=20
SCALES=(1,2,3,4,5,6,7,8)

TRAIN_FILE = os.path.join(
    IODIR,
    _dataset_filename("train", TRAIN_SPLIT, PATCH, STRIDE, ACTIVE_ATOMS),
)
TEST_FILE = os.path.join(
    IODIR,
    _dataset_filename("test", VAL_SPLIT, PATCH, STRIDE, ACTIVE_ATOMS),
)


# Training Params
NUM_EPOCHS = 400
BATCH_SIZE = 1
LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-8
RESUME_LAST_EPOCH = 0
RESUME_LAST_LEARNING_RATE = 5e-5
WEIGHT_DECAY = 3e-4
NUM_WORKERS = 8
PIN_MEMORY = True
GRAD_CLIP = 1.0
DEVICE = "cuda"
PATIENCE = 50
MIN_DELTA = 5e-5
DATASET_TYPE = "patch"
TRAINSELECT = 1
TRAINSELECT_SEED = 12345
USE_COSINE = True
LOAD_EARLIER_VAL = True

# Prediction params
CUDA = DEVICE == "cuda"
TILED = True

TENSOR_SCALE = 1.0
