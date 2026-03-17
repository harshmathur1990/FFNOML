import os
import numpy as np
import h5py
import json
from helita.sim.multi3d import Multi3dAtmos, Multi3dOut
import matplotlib.pyplot as plt
from interp_utils import interpolate_everything
from config import *
from FFNONet import (
    build_dataset_ffno,
    build_solving_set_ffno,
    ffno_train_model,
    ffno_predict_populations
)
import torch.distributed as dist
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    return parser.parse_args()


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
            ndep=NDEP,
            patch=PATCH,
            stride=STRIDE,
            scales=SCALES,
            tscale=TENSOR_SCALE
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
            ndep=NDEP,
            patch=PATCH,
            stride=STRIDE,
            scales=SCALES,
            tscale=TENSOR_SCALE
        )


def train_model():

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
        weight_decay=WEIGHT_DECAY,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        amp=AMP,
        grad_clip=GRAD_CLIP,
        device=DEVICE,
        multi_gpu=MULTI_GPU,
        debug_loss=DEBUG_LOSS,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        tscale=TENSOR_SCALE
    )


def run_predictions():

    for PRED_ATMOS in MULTI3D_PRED_DATA:

        PREDICT_FILE = MODEL_DIR + f"3D_sim_predict_{PRED_ATMOS['NAME']}.hdf5"

        OUTPUT_FILE  = MODEL_DIR + f"output_3D_sim_s5_{PRED_ATMOS['NAME']}_{MODEL}.hdf5"

        DIAGNOSTIC_PATH = MODEL_DIR + f"diagnostics_3D_sim_s5_{PRED_ATMOS['NAME']}_{MODEL}.npz"

        if not os.path.exists(OUTPUT_FILE):

            if not os.path.exists(PREDICT_FILE):

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
                    ndep=NDEP,
                    tscale=TENSOR_SCALE
                )

            ffno_predict_populations(
                model=MODEL,
                checkpoint_path=MODEL_FILE,
                solve_h5=PREDICT_FILE,
                train_h5=TRAIN_FILE,
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
                stride=STRIDE,
                tscale=TENSOR_SCALE
            )


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

    if args.build:
        build_datasets()

    elif args.train:
        train_model()

    elif args.predict:
        run_predictions()

    else:
        raise RuntimeError("Specify one of: --build, --train, --predict")
