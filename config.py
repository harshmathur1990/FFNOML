import numpy as np
import os


SIMULATIONS = {
    "en024048_hion": {
        "base_path": "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion",
        "snaps": ["385", "386"],
    }
}

TRAIN_SPLIT = {
    "en024048_hion": ["385"],
}

VAL_SPLIT = {
    "en024048_hion": ["386"],
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

ACTIVE_SIMS  = ["en024048_hion"]
ACTIVE_ATOMS = ["H"]


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


MULTI3D_TRAIN_DATA = build_multi3d_entries(TRAIN_SPLIT)
MULTI3D_VAL_DATA   = build_multi3d_entries(VAL_SPLIT)

atom_names = ACTIVE_ATOMS
lines = [ATOM_CONFIG[a]["lines"] for a in ACTIVE_ATOMS]
wave  = [ATOM_CONFIG[a]["wave"]  for a in ACTIVE_ATOMS]
chi   = [ATOM_CONFIG[a]["chi"]   for a in ACTIVE_ATOMS]
levels = [ATOM_CONFIG[a]["levels"]   for a in ACTIVE_ATOMS]


MULTI3D_PRED_DATA = [
    {
        "MULTI3D_ATMOS": "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/385/atm3d",
        "MESH":  "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/385/mesh",
        "NAME": "en024048_hion_385"
    },
    {
        "MULTI3D_ATMOS": "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/386/atm3d",
        "MESH":  "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/386/mesh",
        "NAME": "en024048_hion_386"
    },
    {
        "MULTI3D_ATMOS": "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/465/atm3d",
        "MESH":  "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/465/mesh",
        "NAME": "en024048_hion_465"
    },
    {
        "MULTI3D_ATMOS": "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion_504/385/atm3d",
        "MESH":  "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion_504/385/mesh",
        "NAME": "en024048_hion_504_385"
    },
    {
        "MULTI3D_ATMOS": "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion_504/465/atm3d",
        "MESH":  "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion_504/465/mesh",
        "NAME": "en024048_hion_504_465"
    },
    {
        "MULTI3D_ATMOS": "/mn/stornext/d9/data/harshm/bifrost_data/nw012023/1050/atm3d",
        "MESH":  "/mn/stornext/d9/data/harshm/bifrost_data/nw012023/1050/mesh",
        "NAME": "nw012023_1050"
    }
]


NDEP = 400
MULTI_GPU = True

MODEL = "FFNO3D"

MODEL_CONFIG = dict(
    width=64,
    n_layers=6,
    dropout=0.1,
    mlp_expansion=2,
    vertical_hidden=64,
    padding=0,
    checkpoint_blocks=True,
    use_gating=True,
    spectral_hidden=64,
    spectral_rank=16,
    dx_cutoff=96000,
    dy_cutoff=96000,
    k_scale=1e5,
    spectral_use_bias=True,
    spectral_apply_mask=True
)


IODIR = "IO/"
TRAIN_FILE   = IODIR + f"3D_sim_train.hdf5"
TEST_FILE   = IODIR + f"3D_sim_test.hdf5"
MODEL_DIR  = f"training/"
MODEL_FILE = MODEL_DIR + f"3D_sim_train_s123.pt"


DEBUG_LOSS = False

# Dataset Params
NDEP=400
PATCH=32
STRIDE=16
SCALES=(1,2,3,4,5,6,7,8)


# Training Params
NUM_EPOCHS = 400
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 8
PIN_MEMORY = True
GRAD_CLIP = 1.0
DEVICE = "cuda"
PATIENCE = 400
MIN_DELTA = 1e-5
DATASET_TYPE = "patch"
USE_COSINE = True

# Prediction params
CUDA = DEVICE == "cuda"
TILED = True

TENSOR_SCALE = 1.0