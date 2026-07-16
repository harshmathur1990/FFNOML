#!/usr/bin/env python3

import csv
import os

import h5py
import numpy as np

from config import ACTIVE_ATOMS, MODEL, MULTI3D_PRED_DATA
from errorplots import (
    active_atom_names_tag,
    build_truth_paths,
    compute_muspel_lte,
    load_true_multi3d_departures,
    prepare_forward_lte_populations,
)


OLD_DIR = "training_FFNO3D_zscale_expand"
NEW_DIR = "training_FFNO3D_zscale_expand_lognlte"
OUTPUT_CSV = "population_nrmse_per_level.csv"


def prediction_path(directory, dataset_name):
    filename = (
        f"output_3D_sim_s5_{dataset_name}_{MODEL}_"
        f"{active_atom_names_tag()}.hdf5"
    )
    return os.path.join(directory, filename)


def load_prediction(path, key):
    with h5py.File(path, "r") as file:
        return file[key][...]


def normalized_rmse_percent(prediction, truth):
    difference = np.asarray(prediction, dtype=np.float64) - np.asarray(
        truth, dtype=np.float64
    )
    truth = np.asarray(truth, dtype=np.float64)
    squared_error = np.sum(difference**2, dtype=np.float64)
    squared_truth = np.sum(truth**2, dtype=np.float64)
    if squared_truth == 0:
        return np.nan
    return float(100.0 * np.sqrt(squared_error / squared_truth))


def main():
    rows = []

    for configured_dataset in MULTI3D_PRED_DATA:
        dataset = dict(configured_dataset)
        dataset["MULTI3D_PATHS"] = build_truth_paths(dataset)

        old_path = prediction_path(OLD_DIR, dataset["NAME"])
        new_path = prediction_path(NEW_DIR, dataset["NAME"])
        if not os.path.exists(old_path) or not os.path.exists(new_path):
            print(f"Skipping {dataset['NAME']}: prediction file missing")
            continue

        required_truth_files = [dataset["MULTI3D_ATMOS"], dataset["MESH"]]
        for atom_path in dataset["MULTI3D_PATHS"].values():
            required_truth_files.extend(
                os.path.join(atom_path, filename)
                for filename in (
                    "multi3d.input",
                    "out_par",
                    "out_nu",
                    "out_pop",
                    "out_atm",
                )
            )
        missing_truth_files = [
            path for path in required_truth_files if not os.path.exists(path)
        ]
        if missing_truth_files:
            print(
                f"Skipping {dataset['NAME']}: missing Multi3D truth file "
                f"{missing_truth_files[0]}"
            )
            continue

        print(f"\nProcessing {dataset['NAME']}")

        old_departure = load_prediction(old_path, "departure_coefficients")
        new_nlte = load_prediction(new_path, "nlte_populations")

        try:
            _, _, _, truth_nlte, level_names = load_true_multi3d_departures(
                dataset,
                active_atoms=ACTIVE_ATOMS,
            )
        except FileNotFoundError as error:
            print(f"Skipping {dataset['NAME']}: {error}")
            continue
        muspel_lte = compute_muspel_lte(
            dataset,
            active_atoms=ACTIVE_ATOMS,
        )
        muspel_lte = prepare_forward_lte_populations(
            muspel_lte,
            truth_nlte.shape,
        )

        if old_departure.shape != truth_nlte.shape:
            raise ValueError(
                f"Old prediction shape {old_departure.shape} does not match "
                f"truth shape {truth_nlte.shape} for {dataset['NAME']}"
            )
        if new_nlte.shape != truth_nlte.shape:
            raise ValueError(
                f"New prediction shape {new_nlte.shape} does not match "
                f"truth shape {truth_nlte.shape} for {dataset['NAME']}"
            )

        for level_index, level_name in enumerate(level_names):
            old_nlte = (
                old_departure[..., level_index]
                * muspel_lte[..., level_index]
            )
            old_nrmse = normalized_rmse_percent(
                old_nlte,
                truth_nlte[..., level_index],
            )
            new_nrmse = normalized_rmse_percent(
                new_nlte[..., level_index],
                truth_nlte[..., level_index],
            )
            old_nrmse_log = normalized_rmse_percent(
                np.log10(old_nlte),
                np.log10(truth_nlte[..., level_index]),
            )
            new_nrmse_log = normalized_rmse_percent(
                np.log10(new_nlte[..., level_index]),
                np.log10(truth_nlte[..., level_index]),
            )

            print(
                f"  {level_name}: old={old_nrmse:.4f}%, "
                f"new={new_nrmse:.4f}%, "
                f"old_log={old_nrmse_log:.4f}%, "
                f"new_log={new_nrmse_log:.4f}%"
            )
            rows.append(
                {
                    "simulation": dataset["NAME"],
                    "level": level_name,
                    "old_nrmse_percent": old_nrmse,
                    "new_nrmse_percent": new_nrmse,
                    "old_log_nrmse_percent": old_nrmse_log,
                    "new_log_nrmse_percent": new_nrmse_log,
                }
            )

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "simulation",
                "level",
                "old_nrmse_percent",
                "new_nrmse_percent",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
