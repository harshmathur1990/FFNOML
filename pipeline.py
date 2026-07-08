import warnings
import torch

warnings.filterwarnings("ignore", message=".*_get_pg_default_device.*")
torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

import os
import numpy as np
import h5py
import json
from helita.sim.multi3d import Multi3dAtmos, Multi3dOut
import matplotlib.pyplot as plt
from config import *
from FFNONet import (
    build_dataset_ffno,
    build_solving_set_ffno,
    ffno_train_model,
    ffno_test_model,
    ffno_predict_populations,
    ffno_predict_populations_distributed_full,
    ffno_inspect_freq_gate,
)
import torch.distributed as dist
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--fsdppredict", action="store_true")
    parser.add_argument("--buildforpredict", action="store_true")
    parser.add_argument(
        "--predname",
        default=None,
        help=(
            "Limit prediction/buildforpredict to one NAME. For --predict and "
            "--fsdppredict, this can also name an already-built solving HDF5."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bestpath", action="store_true")
    parser.add_argument("--expand", action="store_true")
    parser.add_argument("--inspectfreqgate", action="store_true")
    parser.add_argument("--inspect-h", type=int, default=None)
    parser.add_argument("--inspect-w", type=int, default=None)
    parser.add_argument("--inspect-dx", type=float, default=None)
    parser.add_argument("--inspect-dy", type=float, default=None)
    parser.add_argument("--inspect-out", default=None)
    return parser.parse_args()


def validate_runtime_device():
    wants_cuda = str(DEVICE).startswith("cuda") or CUDA

    if not wants_cuda:
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "config.py requests CUDA, but PyTorch could not find a CUDA-capable GPU. "
            "Set DEVICE='cpu' if you want CPU execution."
        )

    gpu_count = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]

    if int(os.environ.get("RANK", "0")) != 0:
        return

    print("\n=== CUDA DEVICES ===")
    for idx, name in enumerate(gpu_names):
        print(f"GPU {idx}: {name}")


def ensure_built_datasets_exist():
    missing = []

    if not os.path.exists(TRAIN_FILE):
        missing.append(("train", TRAIN_FILE))

    if not os.path.exists(TEST_FILE):
        missing.append(("test", TEST_FILE))

    if missing:
        missing_lines = "\n".join(
            f"  - {label}: {path}" for label, path in missing
        )
        raise FileNotFoundError(
            "With the current config, the required dataset files do not exist.\n"
            f"{missing_lines}\n"
            "Run pipeline.py --build first."
        )


def ensure_validation_dataset_exists():
    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(
            "With the current config, the required validation dataset file does not exist.\n"
            f"  - test: {TEST_FILE}\n"
            "Run pipeline.py --build first."
        )


def ensure_checkpoint_exists():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            "With the current config, the required model checkpoint does not exist.\n"
            f"  - checkpoint: {MODEL_FILE}\n"
            "Run pipeline.py --train first."
        )


def build_datasets():

    if not os.path.exists(TRAIN_FILE):

        multi3d_atmos = load_multi3d_blocks(MULTI3D_TRAIN_DATA)

        build_dataset_ffno(
            multi3d_atmos['temp_list'],
            multi3d_atmos['vx_list'],
            multi3d_atmos['vy_list'],
            multi3d_atmos['vz_list'],
            multi3d_atmos['ne_list'],
            multi3d_atmos['lte_list'],
            multi3d_atmos['nlte_list'],
            multi3d_atmos['rho_list'],
            multi3d_atmos['z_list'],
            multi3d_atmos['dx_list'],
            multi3d_atmos['dy_list'],
            save_path=TRAIN_FILE,
            chi=chi,
            lines=lines,
            wave=wave,
            levels=levels,
            patch=PATCH,
            stride=STRIDE,
            scales=SCALES,
            stat_file=None
        )

    if not os.path.exists(TEST_FILE):

        multi3d_atmos = load_multi3d_blocks(MULTI3D_VAL_DATA)

        build_dataset_ffno(
            multi3d_atmos['temp_list'],
            multi3d_atmos['vx_list'],
            multi3d_atmos['vy_list'],
            multi3d_atmos['vz_list'],
            multi3d_atmos['ne_list'],
            multi3d_atmos['lte_list'],
            multi3d_atmos['nlte_list'],
            multi3d_atmos['rho_list'],
            multi3d_atmos['z_list'],
            multi3d_atmos['dx_list'],
            multi3d_atmos['dy_list'],
            save_path=TEST_FILE,
            chi=chi,
            lines=lines,
            wave=wave,
            levels=levels,
            patch=PATCH,
            stride=STRIDE,
            scales=SCALES,
            stat_file=TRAIN_FILE
        )


def train_model(*, resume=False, bestpath=False, expand=False):
    ensure_built_datasets_exist()

    ffno_train_model(
        model=MODEL,
        train_h5=TRAIN_FILE,
        val_h5=TEST_FILE,
        save_path=MODEL_FILE,
        lines=lines,
        wave=wave,
        chi=chi,
        levels=levels,
        atom_names=atom_names,
        model_config=MODEL_CONFIG,
        dataset_type=DATASET_TYPE,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        resume_last_epoch=RESUME_LAST_EPOCH,
        resume_last_lr=RESUME_LAST_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        grad_clip=GRAD_CLIP,
        device=DEVICE,
        multi_gpu=MULTI_GPU,
        debug_loss=DEBUG_LOSS,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        use_cosine=USE_COSINE,
        min_learning_rate=MIN_LEARNING_RATE,
        resume=resume,
        bestpath=bestpath,
        load_earlier_val=LOAD_EARLIER_VAL,
        expand_from_checkpoint=EXPAND_FROM_CHECKPOINT if expand else None,
        zero_init_new_blocks=ZERO_INIT_NEW_BLOCKS,
        force_expand_validation_baseline=FORCE_EXPAND_VALIDATION_BASELINE,
        train_select=TRAINSELECT,
        train_select_seed=TRAINSELECT_SEED,
    )


def test_model():
    ensure_validation_dataset_exists()
    ensure_checkpoint_exists()

    if MULTI_GPU:
        init_distributed_runtime()

    diagnostic_path = MODEL_DIR + f"val_diagnostics_{MODEL}.json"

    ffno_test_model(
        model=MODEL,
        checkpoint_path=MODEL_FILE,
        val_h5=TEST_FILE,
        diagnostic_path=diagnostic_path,
        lines=lines,
        wave=wave,
        chi=chi,
        levels=levels,
        atom_names=atom_names,
        model_config=MODEL_CONFIG,
        dataset_type=DATASET_TYPE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        device=DEVICE,
        multi_gpu=MULTI_GPU,
    )


def is_main_process():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def barrier_if_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def init_distributed_runtime():
    if dist.is_available() and dist.is_initialized():
        return

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    try:
        dist.init_process_group(
            "nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
        )
    except TypeError:
        dist.init_process_group("nccl")


def init_distributed_prediction():
    init_distributed_runtime()


def select_prediction_entries(prediction_name=None, *, allow_prebuilt=False):
    if prediction_name is None:
        return MULTI3D_PRED_DATA

    selected = [
        pred_atmos
        for pred_atmos in MULTI3D_PRED_DATA
        if pred_atmos["NAME"] == prediction_name
    ]
    if selected:
        return selected

    if allow_prebuilt:
        return [{"NAME": prediction_name}]

    configured_names = ", ".join(pred_atmos["NAME"] for pred_atmos in MULTI3D_PRED_DATA)
    raise KeyError(
        f"No configured prediction atmosphere named {prediction_name!r}. "
        f"Configured names: {configured_names}"
    )


def build_prediction_solving_sets(prediction_name=None):
    for PRED_ATMOS in select_prediction_entries(prediction_name):

        PREDICT_FILE = MODEL_DIR + f"3D_sim_predict_{PRED_ATMOS['NAME']}.hdf5"

        if os.path.exists(PREDICT_FILE):
            print(f"Prediction solving set already exists, skipping: {PREDICT_FILE}")
            continue

        rho, z_scale, temp, vx, vy, vz, ne, dx, dy = load_pred_data(
            mesh_file=PRED_ATMOS['MESH'],
            atmos_file=PRED_ATMOS['MULTI3D_ATMOS']
        )

        build_solving_set_ffno(
            rho=rho,
            z_scale=z_scale,
            temp=temp,
            vx=vx,
            vy=vy,
            vz=vz,
            ne=ne,
            dx=dx,
            dy=dy,
            save_path=PREDICT_FILE,
        )


def run_predictions(*, distributed_full=False, prediction_name=None):
    ensure_checkpoint_exists()

    if distributed_full:
        init_distributed_prediction()

    for PRED_ATMOS in select_prediction_entries(
        prediction_name,
        allow_prebuilt=True,
    ):

        PREDICT_FILE = MODEL_DIR + f"3D_sim_predict_{PRED_ATMOS['NAME']}.hdf5"

        OUTPUT_FILE  = MODEL_DIR + f"output_3D_sim_s5_{PRED_ATMOS['NAME']}_{MODEL}.hdf5"

        DIAGNOSTIC_PATH = MODEL_DIR + f"diagnostics_3D_sim_s5_{PRED_ATMOS['NAME']}_{MODEL}.npz"

        if not os.path.exists(OUTPUT_FILE):

            if not os.path.exists(PREDICT_FILE):
                raise FileNotFoundError(
                    "With the current config, the required prediction solving-set file does not exist.\n"
                    f"  - predict: {PREDICT_FILE}\n"
                    "Run pipeline.py --buildforpredict first."
                )

            if distributed_full:
                ffno_predict_populations_distributed_full(
                    model=MODEL,
                    checkpoint_path=MODEL_FILE,
                    solve_h5=PREDICT_FILE,
                    save_path=OUTPUT_FILE,
                    model_config=MODEL_CONFIG,
                    lines=lines,
                    wave=wave,
                    chi=chi,
                    levels=levels,
                    atom_names=atom_names,
                    cuda=CUDA,
                )
            else:
                ffno_predict_populations(
                    model=MODEL,
                    checkpoint_path=MODEL_FILE,
                    solve_h5=PREDICT_FILE,
                    save_path=OUTPUT_FILE,
                    model_config=MODEL_CONFIG,
                    lines=lines,
                    wave=wave,
                    chi=chi,
                    levels=levels,
                    atom_names=atom_names,
                    diagnostic_path=DIAGNOSTIC_PATH,
                    cuda=CUDA,
                    tiled=TILED,
                    patch=PATCH,
                    stride=STRIDE
                )


def inspect_frequency_gate(*, H=None, W=None, dx=None, dy=None, save_path=None):
    ensure_checkpoint_exists()

    H = PATCH if H is None else H
    W = PATCH if W is None else W

    if dx is None or dy is None:
        if not MULTI3D_PRED_DATA:
            raise RuntimeError(
                "Set --inspect-dx and --inspect-dy when MULTI3D_PRED_DATA is empty."
            )
        mesh_path = MULTI3D_PRED_DATA[0]["MESH"]
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(
                f"Cannot infer dx/dy because the configured mesh does not exist: {mesh_path}\n"
                "Pass --inspect-dx and --inspect-dy explicitly."
            )
        mesh_dx, mesh_dy = compute_dx_dy(mesh_path)
        mesh_dx = abs(mesh_dx)
        mesh_dy = abs(mesh_dy)
        dx = mesh_dx if dx is None else dx
        dy = mesh_dy if dy is None else dy

    if save_path is None:
        save_path = MODEL_DIR + f"freq_gate_{MODEL}_H{int(H)}_W{int(W)}.npz"

    ffno_inspect_freq_gate(
        model=MODEL,
        checkpoint_path=MODEL_FILE,
        model_config=MODEL_CONFIG,
        lines=lines,
        wave=wave,
        chi=chi,
        levels=levels,
        atom_names=atom_names,
        H=H,
        W=W,
        dx=dx,
        dy=dy,
        save_path=save_path,
        cuda=CUDA,
    )


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def read_mesh(mesh_file):
    """
    Reads mesh file from Bifrost or MULTI3D.
    Equivalent to Julia read_mesh().
    """

    # Read ALL whitespace-separated numbers
    tmp = np.fromfile(mesh_file, sep=" ", dtype=np.float32)

    inc = 0

    nx = int(tmp[inc])
    inc += 1
    x = tmp[inc:inc + nx]
    inc += nx

    ny = int(tmp[inc])
    inc += 1
    y = tmp[inc:inc + ny]
    inc += ny

    nz = int(tmp[inc])
    inc += 1
    z = tmp[inc:inc + nz]

    return nx, ny, nz, x, y, z


def compute_dx_dy(mesh_file):

    nx, ny, nz, x, y, z = read_mesh(mesh_file)

    if nx < 2 or ny < 2:
        raise ValueError(f"Mesh too small: {mesh_file}")

    # mesh usually in cm → convert to m
    x = x * 1e-2
    y = y * 1e-2

    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))

    return dx, dy


def load_multi3d_blocks(dataset_entries):

    atmos_list = []
    rho_list = []
    z_list = []
    temp_list = []
    vx_list = []
    vy_list = []
    vz_list = []
    ne_list = []
    lte_list = []
    nlte_list = []
    dx_list = []
    dy_list = []

    for dataset in dataset_entries:

        lte_block = None
        nlte_block = None
        atmos = None

        atmos_path = dataset["MULTI3D_ATMOS"]
        mesh_path = dataset["MESH"]

        for mpath in dataset["MULTI3D_PATHS"]:

            m3d = Multi3dOut(directory=mpath)
            m3d.readall()

            lte = m3d.atom.nstar[:] * 1e6
            nlte = m3d.atom.n[:] * 1e6

            if lte_block is None:
                lte_block = lte
                nlte_block = nlte
            else:
                lte_block = np.concatenate([lte_block, lte], axis=-1)
                nlte_block = np.concatenate([nlte_block, nlte], axis=-1)

            if atmos is None:

                nx, ny, nz, _ = lte.shape

                atmos = Multi3dAtmos(atmos_path, nx, ny, nz)

                if hasattr(atmos, "readall"):
                    atmos.readall()

                rho = atmos.rho[:] * 1e3
                temp = atmos.temp[:]
                vx = atmos.vx[:]
                vy = atmos.vy[:]
                vz = atmos.vz[:]
                ne = atmos.ne[:] * 1e6
                z_scale = m3d.geometry.z[:] * 1e-2
                dx, dy = compute_dx_dy(mesh_path)

                dx = np.abs(dx)
                dy = np.abs(dy)

        atmos_list.append(atmos)
        rho_list.append(rho)
        z_list.append(z_scale)
        temp_list.append(temp)
        vx_list.append(vx)
        vy_list.append(vy)
        vz_list.append(vz)
        ne_list.append(ne)
        lte_list.append(lte_block)
        nlte_list.append(nlte_block)
        dx_list.append(dx)
        dy_list.append(dy)

    return dict(
        atmos_list=atmos_list,
        rho_list=rho_list,
        z_list=z_list,
        temp_list=temp_list,
        vx_list=vx_list,
        vy_list=vy_list,
        vz_list=vz_list,
        ne_list=ne_list,
        lte_list=lte_list,
        nlte_list=nlte_list,
        dx_list=dx_list,
        dy_list=dy_list,
    )


def load_pred_data(mesh_file, atmos_file):
    """
    Loads atmosphere for prediction (no MULTI3D output required).

    Parameters
    ----------
    mesh_file : str
        Path to mesh file (Bifrost/Multi3D mesh)
    atmos_file : str
        Path to atmosphere file (atm3d)

    Returns
    -------
    rho, z_scale, temp, vx, vy, vz, ne
    """

    print("\n=== LOADING PREDICTION ATMOSPHERE ===")

    # --- Read mesh ---
    nx, ny, nz, x, y, z = read_mesh(mesh_file)

    dx, dy = compute_dx_dy(mesh_file)

    dx = np.abs(dx)
    dy = np.abs(dy)

    print(f"Grid size from mesh: nx={nx}, ny={ny}, nz={nz}")

    # --- Load atmosphere ---
    atmos = Multi3dAtmos(atmos_file, nx, ny, nz)

    # --- Extract physical variables ---
    rho = atmos.rho * 1e3       # g/cm^3 → kg/m^3
    temp = atmos.temp
    vx = atmos.vx
    vy = atmos.vy
    vz = atmos.vz
    ne = atmos.ne * 1e6         # cm^-3 → m^-3

    # z from mesh is usually in cm
    z_scale = z * 1e-2          # cm → m

    print("Atmosphere loaded successfully.")
    print("rho shape:", rho.shape)

    return rho, z_scale, temp, vx, vy, vz, ne, dx, dy


if __name__ == "__main__":

    args = parse_args()

    selected_modes = [
        args.build,
        args.train,
        args.test,
        args.predict,
        args.fsdppredict,
        args.buildforpredict,
        args.inspectfreqgate,
    ]
    if sum(bool(mode) for mode in selected_modes) > 1:
        raise RuntimeError("Specify only one of: --build, --train, --test, --predict, --fsdppredict, --buildforpredict, --inspectfreqgate")

    if args.resume and not args.train:
        raise RuntimeError("--resume can only be used together with --train")
    if args.bestpath and not args.train:
        raise RuntimeError("--bestpath can only be used together with --train")
    if args.expand and not args.train:
        raise RuntimeError("--expand can only be used together with --train")
    if args.test and (args.resume or args.bestpath):
        raise RuntimeError("--resume/--bestpath are only valid together with --train")
    if args.expand and (args.resume or args.bestpath):
        raise RuntimeError("--expand cannot be combined with --resume or --bestpath")
    if args.expand and not EXPAND_FROM_CHECKPOINT:
        raise RuntimeError("Set EXPAND_FROM_CHECKPOINT in config.py before using --expand")

    try:
        if args.build:
            build_datasets()

        elif args.buildforpredict:
            build_prediction_solving_sets(prediction_name=args.predname)

        elif args.train:
            validate_runtime_device()
            train_model(resume=args.resume, bestpath=args.bestpath, expand=args.expand)

        elif args.test:
            validate_runtime_device()
            test_model()

        elif args.predict:
            validate_runtime_device()
            run_predictions(prediction_name=args.predname)

        elif args.fsdppredict:
            validate_runtime_device()
            run_predictions(distributed_full=True, prediction_name=args.predname)

        elif args.inspectfreqgate:
            validate_runtime_device()
            inspect_frequency_gate(
                H=args.inspect_h,
                W=args.inspect_w,
                dx=args.inspect_dx,
                dy=args.inspect_dy,
                save_path=args.inspect_out,
            )

        else:
            raise RuntimeError("Specify one of: --build, --train, --test, --predict, --fsdppredict, --inspectfreqgate, --buildforpredict")

    finally:
        cleanup_distributed()
