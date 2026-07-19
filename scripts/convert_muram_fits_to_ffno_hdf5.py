#!/usr/bin/env python3
"""
Convert a MURaM FITS atmosphere into the FFNO prediction HDF5 layout.

The generated file is the solving-set consumed by pipeline.py --predict and
--fsdppredict:

    inputs  [1, 6, nz, nx, ny]
    z_scale [1, nz, nx, ny]   in Mm, matching build_solving_set_ffno
    dx      [1]               in m
    dy      [1]               in m

Input channels are:
    log10(T), ux, uy, uz, log10(ne), log10(rho)
"""

from __future__ import annotations

import argparse
import ctypes
import math
import multiprocessing as mp
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


K_BOLTZMANN_CGS = 1.3806488e-16
_WITT_EOS = None


def _import_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise SystemExit(
            "h5py is required to write the FFNO prediction HDF5 file."
        ) from exc
    return h5py


def _read_fits(path: Path) -> tuple[np.ndarray, dict[str, Any], np.ndarray | None]:
    """
    Read a FITS file.

    Prefer astropy because it supports memmap on large MURaM cubes. Fall back to
    sunpy.io for environments that already use the RH15D conversion script.
    """
    try:
        from astropy.io import fits

        hdul = fits.open(path, memmap=True)
        data = hdul[0].data
        header = dict(hdul[0].header)
        height = None
        if len(hdul) > 1 and hdul[1].data is not None:
            height = np.asarray(hdul[1].data)
        return data, header, height
    except ImportError:
        pass

    try:
        import sunpy.io
    except ImportError as exc:
        raise SystemExit(
            "Reading FITS requires either astropy or sunpy. Install one of them "
            "in the environment where you run this converter."
        ) from exc

    records = sunpy.io.read_file(path)
    data, header = records[0]
    height = records[1][0] if len(records) > 1 else None
    return data, dict(header), None if height is None else np.asarray(height)


def _file_path(
    folder: Path,
    simulation_code: str,
    simulation_name: str,
    quantity: str,
    snap: str,
) -> Path:
    return folder / f"{simulation_code}_{simulation_name}_{quantity}_{snap}.fits"


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _find_electron_density_source(
    folder: Path,
    simulation_code: str,
    simulation_name: str,
    snap: str,
) -> tuple[str, Path]:
    for quantity in ("lgne", "lgp", "lgr"):
        path = _file_path(
            folder, simulation_code, simulation_name, quantity, snap
        )
        if path.is_file():
            return quantity, path
    raise FileNotFoundError(
        "Could not determine electron density: none of the expected lgne, "
        "lgp, or lgr FITS files is available"
    )


def _parse_slice(start: int | None, end: int | None, size: int, axis_name: str) -> slice:
    start = 0 if start is None else start
    end = size if end is None else end
    if not (0 <= start < end <= size):
        raise ValueError(f"Invalid {axis_name} range {start}:{end} for size {size}")
    return slice(start, end)


def _height_indices(
    height_m: np.ndarray,
    height_min_m: float,
    height_max_m: float,
) -> np.ndarray:
    indices = np.flatnonzero((height_m >= height_min_m) & (height_m <= height_max_m))
    if indices.size == 0:
        raise ValueError(
            f"No height points selected by range [{height_min_m}, {height_max_m}] m"
        )
    return indices


def _maybe_infer_spacing(header: dict[str, Any], key: str, scale: float) -> float | None:
    value = header.get(key)
    if value is None:
        return None
    try:
        spacing = abs(float(value)) * scale
    except (TypeError, ValueError):
        return None
    if not math.isfinite(spacing) or spacing <= 0:
        return None
    return spacing


def _read_log_quantity(
    folder: Path,
    simulation_code: str,
    simulation_name: str,
    quantity: str,
    snap: str,
    z_indices: np.ndarray,
    x_slice: slice,
    y_slice: slice,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    path = _require_file(
        _file_path(folder, simulation_code, simulation_name, quantity, snap)
    )
    data, _, _ = _read_fits(path)
    arr = (
        10.0 ** np.asarray(data[z_indices, x_slice, y_slice], dtype=np.float32)
    ).astype(np.float32, copy=False)
    if scale != 1.0:
        arr *= np.float32(scale)
    return np.transpose(arr, (1, 2, 0)).astype(np.float32, copy=False)


def _read_linear_quantity(
    folder: Path,
    simulation_code: str,
    simulation_name: str,
    quantity: str,
    snap: str,
    z_indices: np.ndarray,
    x_slice: slice,
    y_slice: slice,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    path = _require_file(
        _file_path(folder, simulation_code, simulation_name, quantity, snap)
    )
    data, _, _ = _read_fits(path)
    arr = np.asarray(data[z_indices, x_slice, y_slice], dtype=np.float32)
    if scale != 1.0:
        arr *= np.float32(scale)
    return np.transpose(arr, (1, 2, 0)).astype(np.float32, copy=False)


def _witt_candidate_paths(witt_path: str | None) -> list[Path]:
    candidate_paths = []
    if witt_path:
        candidate_paths.append(Path(witt_path))
    candidate_paths.extend(
        [
            Path(__file__).resolve().parent,
            Path.cwd() / "scripts",
            Path.cwd(),
        ]
    )
    return candidate_paths


def _find_witt_path(witt_path: str | None) -> Path | None:
    for candidate_path in _witt_candidate_paths(witt_path):
        if (candidate_path / "witt.py").is_file():
            return candidate_path
    return None


def _find_pf_path(witt_path: str | None) -> Path:
    for candidate_path in _witt_candidate_paths(witt_path):
        pf_path = candidate_path / "pf_Kurucz.input"
        if pf_path.is_file():
            return pf_path
    raise FileNotFoundError(
        "Could not find pf_Kurucz.input next to the converter, in ./scripts, "
        "or in the current directory. Pass --witt-path if it lives elsewhere."
    )


def _import_witt(witt_path: str | None):
    found_path = _find_witt_path(witt_path)
    if found_path is not None:
        sys.path.insert(0, os.fspath(found_path))

    try:
        from witt import witt
    except ImportError as exc:
        raise SystemExit(
            "Witt EOS mode requires witt.py and pf_Kurucz.input. The converter "
            "looked next to itself, in ./scripts, and in the current directory. "
            "Pass --witt-path if the files live somewhere else."
        ) from exc
    return witt()


def _shared_library_suffix() -> str:
    if platform.system() == "Darwin":
        return ".dylib"
    if platform.system() == "Windows":
        return ".dll"
    return ".so"


def _compile_witt_cpp_library(source_path: Path, library_path: Path) -> None:
    compiler = os.environ.get("CXX", "c++")
    command = [
        compiler,
        "-O3",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-pthread",
        os.fspath(source_path),
        "-o",
        os.fspath(library_path),
    ]
    subprocess.run(command, check=True)


def _load_witt_cpp_library(args) -> ctypes.CDLL:
    source_path = Path(__file__).resolve().parent / "witt_eos_cpp.cpp"
    library_dir = source_path.parent / "__pycache__"
    library_path = library_dir / f"libwitt_eos_cpp{_shared_library_suffix()}"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    needs_build = not library_path.is_file()
    if not needs_build:
        needs_build = source_path.stat().st_mtime > library_path.stat().st_mtime
    if needs_build:
        library_dir.mkdir(parents=True, exist_ok=True)
        _compile_witt_cpp_library(source_path, library_path)

    lib = ctypes.CDLL(os.fspath(library_path))
    lib.witt_ne_from_rho.argtypes = [
        ctypes.c_char_p,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.witt_ne_from_rho.restype = ctypes.c_int
    lib.witt_ne_from_pgas.argtypes = [
        ctypes.c_char_p,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.witt_ne_from_pgas.restype = ctypes.c_int
    return lib


def _init_witt_worker(witt_path: str | None):
    global _WITT_EOS
    _WITT_EOS = _import_witt(witt_path)


def _witt_ne_chunk(task):
    start, temp_chunk, rho_chunk = task
    if _WITT_EOS is None:
        raise RuntimeError("Witt EOS worker was not initialized")

    ne_chunk = np.empty(temp_chunk.shape, dtype=np.float32)
    for i, (t_cell, rho_cell) in enumerate(zip(temp_chunk, rho_chunk)):
        t_cell = float(t_cell)
        rho_cell = float(rho_cell)
        pgas_cgs = _WITT_EOS.pg_from_rho(t_cell, rho_cell)
        pe_cgs = _WITT_EOS.pe_from_pg(t_cell, pgas_cgs)
        ne_chunk[i] = np.float32(pe_cgs / (K_BOLTZMANN_CGS * t_cell) * 1e6)

    return start, ne_chunk


def _iter_witt_chunks(temp_flat, rho_flat, chunk_size):
    for start in range(0, temp_flat.size, chunk_size):
        end = min(start + chunk_size, temp_flat.size)
        yield start, temp_flat[start:end], rho_flat[start:end]


def _electron_density_from_witt_rho_cpp(args, temp: np.ndarray, rho: np.ndarray) -> np.ndarray:
    lib = _load_witt_cpp_library(args)
    pf_path = _find_pf_path(args.witt_path)
    temp_flat = np.ascontiguousarray(np.asarray(temp, dtype=np.float64).reshape(-1))
    rho_flat = np.ascontiguousarray(np.asarray(rho, dtype=np.float64).reshape(-1))
    ne_flat = np.empty(temp_flat.shape, dtype=np.float32)
    threads = 0 if args.eos_workers is None else int(args.eos_workers)
    status = lib.witt_ne_from_rho(
        os.fsencode(pf_path),
        temp_flat,
        rho_flat,
        ne_flat,
        temp_flat.size,
        threads,
        int(args.show_eos_progress),
    )
    if status != 0:
        raise RuntimeError("C++ Witt EOS calculation failed")
    return ne_flat.reshape(temp.shape)


def _electron_density_from_witt_rho_python(args, temp: np.ndarray, rho: np.ndarray) -> np.ndarray:
    temp_flat = np.asarray(temp, dtype=np.float64).reshape(-1)
    rho_flat = (np.asarray(rho, dtype=np.float64).reshape(-1) * 1e-3)
    ne_flat = np.empty(temp_flat.shape, dtype=np.float32)

    worker_count = args.eos_workers
    if worker_count is None:
        worker_count = os.cpu_count() or 1
    worker_count = max(1, int(worker_count))

    chunk_size = max(1, int(args.eos_chunk_size))
    chunks = _iter_witt_chunks(temp_flat, rho_flat, chunk_size)

    progress = None
    if args.show_eos_progress:
        try:
            from tqdm import tqdm

            progress = tqdm(total=temp_flat.size, desc="witt-rho ne")
        except ImportError:
            progress = None

    try:
        if worker_count == 1:
            _init_witt_worker(args.witt_path)
            for start, ne_chunk in map(_witt_ne_chunk, chunks):
                ne_flat[start:start + ne_chunk.size] = ne_chunk
                if progress is not None:
                    progress.update(ne_chunk.size)
        else:
            context = mp.get_context(args.eos_start_method)
            with context.Pool(
                processes=worker_count,
                initializer=_init_witt_worker,
                initargs=(args.witt_path,),
            ) as pool:
                for start, ne_chunk in pool.imap_unordered(
                    _witt_ne_chunk,
                    chunks,
                    chunksize=max(1, int(args.eos_pool_chunksize)),
                ):
                    ne_flat[start:start + ne_chunk.size] = ne_chunk
                    if progress is not None:
                        progress.update(ne_chunk.size)
    finally:
        if progress is not None:
            progress.close()

    return ne_flat.reshape(temp.shape)


def _electron_density_from_witt_rho(args, temp: np.ndarray, rho: np.ndarray) -> np.ndarray:
    if args.eos_backend in ("auto", "cpp"):
        try:
            return _electron_density_from_witt_rho_cpp(args, temp, rho)
        except Exception as exc:
            if args.eos_backend == "cpp":
                raise
            print(
                f"WARNING: C++ Witt EOS backend unavailable ({exc}); "
                "falling back to Python."
            )

    return _electron_density_from_witt_rho_python(args, temp, rho)


def _electron_density_from_witt_pgas_cpp(
    args, temp: np.ndarray, pgas: np.ndarray
) -> np.ndarray:
    lib = _load_witt_cpp_library(args)
    pf_path = _find_pf_path(args.witt_path)
    temp_flat = np.ascontiguousarray(np.asarray(temp, dtype=np.float64).reshape(-1))
    pgas_flat = np.ascontiguousarray(np.asarray(pgas, dtype=np.float64).reshape(-1))
    ne_flat = np.empty(temp_flat.shape, dtype=np.float32)
    threads = 0 if args.eos_workers is None else int(args.eos_workers)
    status = lib.witt_ne_from_pgas(
        os.fsencode(pf_path),
        temp_flat,
        pgas_flat,
        ne_flat,
        temp_flat.size,
        threads,
        int(args.show_eos_progress),
    )
    if status != 0:
        raise RuntimeError("C++ Witt EOS gas-pressure calculation failed")
    return ne_flat.reshape(temp.shape)


def _electron_density_from_witt_pgas_python(
    args, temp: np.ndarray, pgas: np.ndarray
) -> np.ndarray:
    eos = _import_witt(args.witt_path)
    temp_flat = np.asarray(temp, dtype=np.float64).reshape(-1)
    # The converter stores pressure in Pa; Witt uses dyn cm^-2.
    pgas_flat = np.asarray(pgas, dtype=np.float64).reshape(-1) * 10.0
    ne_flat = np.empty(temp_flat.shape, dtype=np.float32)
    for i, (t_cell, pgas_cell) in enumerate(zip(temp_flat, pgas_flat)):
        pe_cgs = eos.pe_from_pg(float(t_cell), float(pgas_cell))
        ne_flat[i] = np.float32(pe_cgs / (K_BOLTZMANN_CGS * t_cell) * 1e6)
    return ne_flat.reshape(temp.shape)


def _electron_density_from_witt_pgas(
    args, temp: np.ndarray, pgas: np.ndarray
) -> np.ndarray:
    if args.eos_backend in ("auto", "cpp"):
        try:
            return _electron_density_from_witt_pgas_cpp(args, temp, pgas)
        except Exception as exc:
            if args.eos_backend == "cpp":
                raise
            print(f"WARNING: C++ Witt EOS backend unavailable ({exc}); falling back to Python.")
    return _electron_density_from_witt_pgas_python(args, temp, pgas)


def _validate_positive(name: str, arr: np.ndarray) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf")
    if np.any(arr <= 0):
        raise ValueError(f"{name} must be positive before log10")


def _write_multi3d_atmosphere(
    atmos_path: Path,
    mesh_path: Path | None,
    *,
    temp: np.ndarray,
    rho: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    ne: np.ndarray,
    dx_m: float,
    dy_m: float,
    height_m: np.ndarray,
    velocity_to_kms: float,
    overwrite: bool,
) -> None:
    try:
        from helita.sim.multi3d import Multi3dAtmos
    except ImportError as exc:
        raise SystemExit(
            "Writing Multi3D atmosphere requires helita. Run this in the same "
            "environment used for Multi3D/Bifrost files."
        ) from exc

    if atmos_path.exists() and not overwrite:
        raise FileExistsError(f"Multi3D atmosphere exists: {atmos_path}")
    if mesh_path is not None and mesh_path.exists() and not overwrite:
        raise FileExistsError(f"Multi3D mesh exists: {mesh_path}")

    atmos_path.parent.mkdir(parents=True, exist_ok=True)
    if mesh_path is not None:
        mesh_path.parent.mkdir(parents=True, exist_ok=True)

    nx, ny, nz = temp.shape
    atmos = Multi3dAtmos(os.fspath(atmos_path), nx, ny, nz, mode="w+", read_nh=False)
    atmos.ne[:] = (ne * 1e-6).astype(np.float32, copy=False)
    atmos.temp[:] = temp.astype(np.float32, copy=False)
    atmos.vx[:] = (vx * velocity_to_kms).astype(np.float32, copy=False)
    atmos.vy[:] = (vy * velocity_to_kms).astype(np.float32, copy=False)
    atmos.vz[:] = (vz * velocity_to_kms).astype(np.float32, copy=False)
    atmos.rho[:] = (rho * 1e-3).astype(np.float32, copy=False)

    if mesh_path is not None:
        x = np.arange(nx, dtype=np.float64) * dx_m * 1e2
        y = np.arange(ny, dtype=np.float64) * dy_m * 1e2
        z = np.asarray(height_m, dtype=np.float64) * 1e2

        with open(mesh_path, "w") as mesh_file:
            mesh_file.write(f"{nx}\n")
            x.tofile(mesh_file, sep="  ", format="%11.5e")
            mesh_file.write(f"\n{ny}\n")
            y.tofile(mesh_file, sep="  ", format="%11.5e")
            mesh_file.write(f"\n{nz}\n")
            z.tofile(mesh_file, sep="  ", format="%11.5e")

    print(f"Wrote Multi3D atmosphere: {atmos_path}")
    if mesh_path is not None:
        print(f"Wrote Multi3D mesh: {mesh_path}")


def convert(args) -> None:
    h5py = _import_h5py()

    folder = Path(args.folder)
    snap = str(args.snap)
    electron_source, _ = _find_electron_density_source(
        folder, args.simulation_code, args.simulation_name, snap
    )
    temp_path = _require_file(
        _file_path(folder, args.simulation_code, args.simulation_name, "lgtg", snap)
    )

    temp_data, temp_header, height = _read_fits(temp_path)
    if temp_data.ndim != 3:
        raise ValueError(f"Expected 3D FITS cube in {temp_path}, got {temp_data.shape}")
    if height is None:
        raise ValueError(
            f"{temp_path} has no height extension. Pass a FITS file with height "
            "as extension 1, matching the existing RH15D converter."
        )

    nz_full, nx_full, ny_full = temp_data.shape
    height_m = np.asarray(height, dtype=np.float64).reshape(-1) * args.height_unit_scale
    if height_m.size != nz_full:
        raise ValueError(
            f"Height size {height_m.size} does not match FITS z dimension {nz_full}"
        )

    z_indices = _height_indices(height_m, args.height_min_m, args.height_max_m)
    x_slice = _parse_slice(args.start_x, args.end_x, nx_full, "x")
    y_slice = _parse_slice(args.start_y, args.end_y, ny_full, "y")

    args._z_indices = z_indices
    args._x_slice = x_slice
    args._y_slice = y_slice

    dx = args.dx
    dy = args.dy
    if dx is None:
        dx = _maybe_infer_spacing(temp_header, "CDELT1", args.xy_unit_scale)
    if dy is None:
        dy = _maybe_infer_spacing(temp_header, "CDELT2", args.xy_unit_scale)
    if dx is None or dy is None:
        raise ValueError(
            "Could not infer dx/dy from FITS CDELT1/CDELT2. Pass --dx and --dy in meters."
        )

    temp = _read_log_quantity(
        folder,
        args.simulation_code,
        args.simulation_name,
        "lgtg",
        snap,
        z_indices,
        x_slice,
        y_slice,
    )
    rho = _read_log_quantity(
        folder,
        args.simulation_code,
        args.simulation_name,
        "lgr",
        snap,
        z_indices,
        x_slice,
        y_slice,
        scale=args.rho_scale,
    )
    vx = _read_linear_quantity(
        folder,
        args.simulation_code,
        args.simulation_name,
        "ux",
        snap,
        z_indices,
        x_slice,
        y_slice,
        scale=args.velocity_scale,
    )
    vy = _read_linear_quantity(
        folder,
        args.simulation_code,
        args.simulation_name,
        "uy",
        snap,
        z_indices,
        x_slice,
        y_slice,
        scale=args.velocity_scale,
    )
    vz = _read_linear_quantity(
        folder,
        args.simulation_code,
        args.simulation_name,
        "uz",
        snap,
        z_indices,
        x_slice,
        y_slice,
        scale=args.velocity_scale,
    )

    if electron_source == "lgne":
        ne = _read_log_quantity(
            folder,
            args.simulation_code,
            args.simulation_name,
            "lgne",
            snap,
            z_indices,
            x_slice,
            y_slice,
            scale=args.electron_density_scale,
        )
        electron_density_source = "lgne"
    elif electron_source == "lgp":
        pgas = _read_log_quantity(
            folder,
            args.simulation_code,
            args.simulation_name,
            "lgp",
            snap,
            z_indices,
            x_slice,
            y_slice,
            scale=args.pressure_scale,
        )
        ne = _electron_density_from_witt_pgas(args, temp, pgas)
        electron_density_source = "witt-pgas"
    else:
        ne = _electron_density_from_witt_rho(args, temp, rho)
        electron_density_source = "witt-rho"

    # Match the attached RH15D converter: reverse selected z so index 0 is top.
    height_selected = height_m[z_indices]
    if args.reverse_z:
        temp = temp[:, :, ::-1]
        rho = rho[:, :, ::-1]
        vx = vx[:, :, ::-1]
        vy = vy[:, :, ::-1]
        vz = vz[:, :, ::-1]
        ne = ne[:, :, ::-1]
        height_selected = height_selected[::-1]

    _validate_positive("temperature", temp)
    _validate_positive("electron density", ne)
    _validate_positive("density", rho)

    inputs = np.stack(
        [
            np.log10(temp),
            vx,
            vy,
            vz,
            np.log10(ne),
            np.log10(rho),
        ],
        axis=0,
    )
    inputs = np.transpose(inputs, (0, 3, 1, 2)).astype(np.float32, copy=False)

    z_native = np.broadcast_to(
        (height_selected.astype(np.float32) / np.float32(1e6)).reshape(-1, 1, 1),
        (inputs.shape[1], inputs.shape[2], inputs.shape[3]),
    )

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    mode = "w" if args.overwrite else "x"
    with h5py.File(output, mode) as f:
        f.create_dataset(
            "inputs",
            data=inputs[None, ...],
            compression="gzip",
            compression_opts=args.compression,
            shuffle=True,
        )
        f.create_dataset(
            "z_scale",
            data=z_native[None, ...],
            compression="gzip",
            compression_opts=args.compression,
            shuffle=True,
        )
        f.create_dataset("dx", data=np.array([dx], dtype=np.float32))
        f.create_dataset("dy", data=np.array([dy], dtype=np.float32))
        f.attrs["N"] = 1
        f.attrs["Cin"] = inputs.shape[0]
        f.attrs["D"] = inputs.shape[1]
        f.attrs["nx"] = inputs.shape[2]
        f.attrs["ny"] = inputs.shape[3]
        f.attrs["native_depth"] = inputs.shape[1]
        f.attrs["source_format"] = "MURaM FITS"
        f.attrs["source_folder"] = os.fspath(folder)
        f.attrs["simulation_code"] = args.simulation_code
        f.attrs["simulation_name"] = args.simulation_name
        f.attrs["snap"] = snap
        f.attrs["electron_density_source"] = electron_density_source

    print(f"Wrote {output}")
    print(f"inputs shape: {(1,) + inputs.shape}")
    print(f"z_scale shape: {(1,) + z_native.shape}")
    print(f"dx={dx} m, dy={dy} m")
    print(f"height range: {float(height_selected.min())} .. {float(height_selected.max())} m")

    if args.multi3d_atmos_out:
        _write_multi3d_atmosphere(
            Path(args.multi3d_atmos_out),
            None if args.multi3d_mesh_out is None else Path(args.multi3d_mesh_out),
            temp=temp,
            rho=rho,
            vx=vx,
            vy=vy,
            vz=vz,
            ne=ne,
            dx_m=dx,
            dy_m=dy,
            height_m=height_selected,
            velocity_to_kms=args.multi3d_velocity_to_kms,
            overwrite=args.overwrite,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MURaM FITS cubes to FFNO prediction HDF5."
    )
    parser.add_argument("--folder", required=True, help="Directory containing MURaM FITS files")
    parser.add_argument("--simulation-code", default="MURaM")
    parser.add_argument("--simulation-name", required=True)
    parser.add_argument("--snap", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-x", type=int, default=None)
    parser.add_argument("--end-x", type=int, default=None)
    parser.add_argument("--start-y", type=int, default=None)
    parser.add_argument("--end-y", type=int, default=None)
    parser.add_argument("--height-min-m", type=float, default=-np.inf)
    parser.add_argument("--height-max-m", type=float, default=np.inf)
    parser.add_argument(
        "--height-unit-scale",
        type=float,
        default=1e6,
        help="Multiplier from FITS height extension units to meters. Default assumes Mm.",
    )
    parser.add_argument(
        "--xy-unit-scale",
        type=float,
        default=1e6,
        help="Multiplier from FITS CDELT1/CDELT2 units to meters. Default assumes Mm.",
    )
    parser.add_argument("--dx", type=float, default=None, help="Horizontal x spacing in meters")
    parser.add_argument("--dy", type=float, default=None, help="Horizontal y spacing in meters")
    parser.add_argument(
        "--rho-scale",
        type=float,
        default=1e3,
        help="Multiplier after 10**lgr. Default converts g/cm^3 to kg/m^3.",
    )
    parser.add_argument(
        "--velocity-scale",
        type=float,
        default=1.0,
        help="Multiplier for ux/uy/uz. Use 1e3 if FITS velocities are km/s.",
    )
    parser.add_argument(
        "--pressure-scale",
        type=float,
        default=10.0,
        help="Multiplier after 10**lgp. Default converts dyn/cm^2 to Pa.",
    )
    parser.add_argument(
        "--electron-density-scale",
        type=float,
        default=1e6,
        help=(
            "Multiplier after 10**lgne. Default converts cm^-3 to m^-3. "
            "Used only when an lgne FITS file is available."
        ),
    )
    parser.add_argument(
        "--witt-path",
        default=None,
        help="Directory containing witt.py and pf_Kurucz.input for Witt EOS modes.",
    )
    parser.add_argument(
        "--show-eos-progress",
        action="store_true",
        help="Show progress for the Witt EOS calculation when supported.",
    )
    parser.add_argument(
        "--eos-backend",
        choices=["auto", "cpp", "python"],
        default="auto",
        help=(
            "Backend for Witt electron-density calculations. 'auto' uses "
            "the C++ full-atmosphere backend when it can be compiled and "
            "falls back to Python."
        ),
    )
    parser.add_argument(
        "--eos-workers",
        type=int,
        default=None,
        help=(
            "Worker/thread count for Witt electron-density calculation. "
            "Default uses all visible CPUs. Use 1 for serial execution."
        ),
    )
    parser.add_argument(
        "--eos-chunk-size",
        type=int,
        default=4096,
        help="Number of cells per worker task for the Python Witt EOS backend.",
    )
    parser.add_argument(
        "--eos-pool-chunksize",
        type=int,
        default=1,
        help="Number of EOS tasks batched per Python multiprocessing dispatch.",
    )
    parser.add_argument(
        "--eos-start-method",
        choices=mp.get_all_start_methods(),
        default="fork" if "fork" in mp.get_all_start_methods() else mp.get_start_method(),
        help="Multiprocessing start method for Python Witt EOS workers.",
    )
    parser.add_argument("--no-reverse-z", dest="reverse_z", action="store_false")
    parser.set_defaults(reverse_z=True)
    parser.add_argument(
        "--multi3d-atmos-out",
        default=None,
        help="Optional Multi3D atmosphere output path, e.g. ar098192/270000/atm3d.",
    )
    parser.add_argument(
        "--multi3d-mesh-out",
        default=None,
        help="Optional Multi3D mesh output path. Used only with --multi3d-atmos-out.",
    )
    parser.add_argument(
        "--multi3d-velocity-to-kms",
        type=float,
        default=1e-5,
        help=(
            "Multiplier from internal converter velocity units to km/s for "
            "Multi3D output. Default assumes ux/uy/uz are cm/s."
        ),
    )
    parser.add_argument("--compression", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
