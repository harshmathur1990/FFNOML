#!/usr/bin/env python3
"""
Convert a MURaM FITS atmosphere into the FFNO prediction HDF5 layout.

The generated file is the solving-set consumed by pipeline.py --predict and
--fsdppredict:

    inputs  [1, 6, nz, nx, ny]
    z_scale [1, nz, nx, ny]   in Mm, matching build_solving_set_ffno
    dx      [1]               in m
    dy      [1]               in m

Input channels are normalized from FITS BUNIT/CUNIT metadata to:
    log10(T [K]), ux [km/s], uy [km/s], uz [km/s],
    log10(ne [m^-3]), log10(rho [kg/m^3])
"""

from __future__ import annotations

import ctypes
import math
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


K_BOLTZMANN_CGS = 1.3806488e-16
_WITT_EOS = None
_INTERPOLATION_METHODS = {"nearest", "linear", "cubic"}
_OUTPUT_CHANNELS = ("temperature", "rho", "vx", "vy", "vz", "ne")
_RESAMPLED_CHANNELS = _OUTPUT_CHANNELS + ("pgas", "hydrogen_populations")
_DEFAULT_LOG_INTERPOLATED_CHANNELS = {
    "temperature",
    "rho",
    "ne",
    "pgas",
    "hydrogen_populations",
}


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


def _find_hydrogen_population_paths(
    folder: Path,
    simulation_code: str,
    simulation_name: str,
    snap: str,
) -> tuple[Path, ...] | None:
    """Return lgn1..lgn6 paths when the complete population set is present."""
    paths = tuple(
        _file_path(
            folder,
            simulation_code,
            simulation_name,
            f"lgn{level}",
            snap,
        )
        for level in range(1, 7)
    )
    return paths if all(path.is_file() for path in paths) else None


def _parse_slice_text(value: str) -> slice:
    """Parse ``START:STOP[:STEP]`` or ``slice(START, STOP[, STEP])``."""
    text = value.strip()
    if text.startswith("slice(") and text.endswith(")"):
        parts = [part.strip() for part in text[6:-1].split(",")]
    else:
        parts = [part.strip() for part in text.split(":")]
    if len(parts) not in (2, 3):
        raise ValueError(
            "slice must be START:STOP[:STEP] or slice(START, STOP[, STEP])"
        )
    try:
        values = [None if part in ("", "None") else int(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"invalid slice {value!r}") from exc
    if len(values) == 2:
        values.append(None)
    return slice(*values)


def _parse_slice(
    start: int | None,
    end: int | None,
    size: int,
    axis_name: str,
    *,
    step: int | None = None,
) -> slice:
    start = 0 if start is None else start
    end = size if end is None else end
    step = 1 if step is None else step
    if not (0 <= start < end <= size) or step <= 0:
        raise ValueError(
            f"Invalid {axis_name} slice {start}:{end}:{step} for size {size}"
        )
    return slice(start, end, step)


def _resolve_slice(
    requested: slice | None,
    start: int | None,
    end: int | None,
    size: int,
    axis_name: str,
) -> slice:
    """Validate an explicit slice or fall back to legacy start/end options."""
    if requested is not None:
        if start is not None or end is not None:
            raise ValueError(
                f"selection.{axis_name} cannot be combined with legacy "
                f"start_{axis_name}/end_{axis_name} settings"
            )
        return _parse_slice(
            requested.start,
            requested.stop,
            size,
            axis_name,
            step=requested.step,
        )
    return _parse_slice(start, end, size, axis_name)


def _height_indices(
    height_m: np.ndarray,
    height_min_m: float,
    height_max_m: float,
    candidates: np.ndarray | None = None,
) -> np.ndarray:
    if candidates is None:
        candidates = np.arange(height_m.size)
    candidates = np.asarray(candidates, dtype=np.intp)
    selected_heights = height_m[candidates]
    indices = candidates[
        (selected_heights >= height_min_m) & (selected_heights <= height_max_m)
    ]
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


def _import_astropy_units():
    try:
        from astropy import units as u
    except ImportError as exc:
        raise SystemExit(
            "astropy is required to interpret FITS units and convert them to SI."
        ) from exc
    return u


def _parse_header_unit(raw_unit: Any):
    """Parse a FITS unit, accepting the parenthesized powers in MURaM headers."""
    u = _import_astropy_units()
    unit_text = str(raw_unit).strip().replace("−", "-")
    try:
        return u.Unit(unit_text)
    except ValueError:
        # MURaM writes e.g. ``m s^(-1)``; Astropy's generic/FITS-like syntax
        # represents the same unit as ``m s^-1``.
        normalized = unit_text.replace("^(", "^").replace(")", "")
        return u.Unit(normalized)


def _unit_scale_from_header(
    header: dict[str, Any],
    key: str,
    target_unit: str,
    quantity: str,
) -> float:
    """Return Astropy's conversion factor from a FITS header unit."""
    raw_unit = header.get(key)
    if raw_unit is None or not str(raw_unit).strip():
        raise ValueError(f"FITS header has no {key} for {quantity}")

    try:
        source = _parse_header_unit(raw_unit)
        target = _import_astropy_units().Unit(target_unit)
        return float(source.to(target))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Cannot convert {key}={raw_unit!r} for {quantity} to {target_unit}"
        ) from exc


def _convert_values(values: np.ndarray, source_unit, target_unit) -> np.ndarray:
    """Convert array values using an Astropy-derived multiplicative factor."""
    factor = float(source_unit.to(target_unit))
    return values * factor


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
    si_unit: str,
    quantity_name: str,
) -> np.ndarray:
    path = _require_file(
        _file_path(folder, simulation_code, simulation_name, quantity, snap)
    )
    data, header, _ = _read_fits(path)
    scale = _unit_scale_from_header(header, "BUNIT", si_unit, quantity_name)
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
    si_unit: str,
    quantity_name: str,
) -> np.ndarray:
    path = _require_file(
        _file_path(folder, simulation_code, simulation_name, quantity, snap)
    )
    data, header, _ = _read_fits(path)
    scale = _unit_scale_from_header(header, "BUNIT", si_unit, quantity_name)
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
        "or in the current directory. Set [eos].witt_path if it lives elsewhere."
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
            "Set [eos].witt_path if the files live somewhere else."
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


def _target_axis_size(size: int, factor: float | None, target: int | None) -> int:
    if target is not None:
        target = int(target)
        if target < size:
            raise ValueError(
                f"Resampling target {target} would downsample an axis of size {size}"
            )
        return target
    if factor is None:
        return size
    factor = float(factor)
    if not math.isfinite(factor) or factor < 1.0:
        raise ValueError(f"Upsampling factors must be finite and >= 1, got {factor}")
    if size == 1:
        return 1
    return int(round((size - 1) * factor)) + 1


def _natural_cubic_second_derivatives(
    coordinates: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Natural-cubic second derivatives for values shaped (n, n_series)."""
    n = coordinates.size
    second = np.zeros_like(values, dtype=np.float64)
    if n <= 2:
        return second

    h = np.diff(coordinates)
    diagonal = 2.0 * (h[:-1] + h[1:])
    lower = h[1:-1]
    upper = h[1:-1]
    rhs = 6.0 * (
        (values[2:] - values[1:-1]) / h[1:, None]
        - (values[1:-1] - values[:-2]) / h[:-1, None]
    )

    # Thomas algorithm. The tridiagonal coefficients are shared by all series.
    c_prime = np.empty(max(0, n - 2), dtype=np.float64)
    d_prime = np.empty_like(rhs, dtype=np.float64)
    c_prime[0] = upper[0] / diagonal[0] if n > 3 else 0.0
    d_prime[0] = rhs[0] / diagonal[0]
    for row in range(1, n - 2):
        denominator = diagonal[row] - lower[row - 1] * c_prime[row - 1]
        c_prime[row] = upper[row] / denominator if row < n - 3 else 0.0
        d_prime[row] = (rhs[row] - lower[row - 1] * d_prime[row - 1]) / denominator

    second[-2] = d_prime[-1]
    for row in range(n - 4, -1, -1):
        second[row + 1] = d_prime[row] - c_prime[row] * second[row + 2]
    return second


def _interpolate_series(
    coordinates: np.ndarray,
    target_coordinates: np.ndarray,
    values: np.ndarray,
    method: str,
) -> np.ndarray:
    """Interpolate a matrix whose first dimension follows coordinates."""
    if coordinates.size == 1:
        if target_coordinates.size != 1:
            raise ValueError("Cannot upsample an axis containing only one point")
        return values.copy()

    indices = np.searchsorted(coordinates, target_coordinates, side="right") - 1
    indices = np.clip(indices, 0, coordinates.size - 2)
    left = coordinates[indices]
    right = coordinates[indices + 1]

    if method == "nearest":
        choose_right = (target_coordinates - left) > (right - target_coordinates)
        nearest = indices + choose_right.astype(np.intp)
        return values[nearest]

    fraction = ((target_coordinates - left) / (right - left))[:, None]
    if method == "linear":
        return values[indices] * (1.0 - fraction) + values[indices + 1] * fraction

    second = _natural_cubic_second_derivatives(coordinates, values)
    width = (right - left)[:, None]
    a = (right[:, None] - target_coordinates[:, None]) / width
    b = (target_coordinates[:, None] - left[:, None]) / width
    return (
        a * values[indices]
        + b * values[indices + 1]
        + ((a**3 - a) * second[indices] + (b**3 - b) * second[indices + 1])
        * width**2
        / 6.0
    )


def _interpolate_axis(
    values: np.ndarray,
    coordinates: np.ndarray,
    target_coordinates: np.ndarray,
    axis: int,
    method: str,
    *,
    log_space: bool,
) -> np.ndarray:
    """Interpolate one cube axis, chunking independent series to bound memory."""
    if method not in _INTERPOLATION_METHODS:
        raise ValueError(
            f"Unknown interpolation method {method!r}; choose nearest, linear, or cubic"
        )
    coordinates = np.asarray(coordinates, dtype=np.float64)
    target_coordinates = np.asarray(target_coordinates, dtype=np.float64)
    if coordinates.ndim != 1 or coordinates.size != values.shape[axis]:
        raise ValueError("Interpolation coordinates do not match the selected axis")
    if np.any(~np.isfinite(coordinates)) or np.any(np.diff(coordinates) == 0):
        raise ValueError("Interpolation coordinates must be finite and strictly monotonic")

    reversed_axis = coordinates[0] > coordinates[-1]
    if reversed_axis:
        coordinates = coordinates[::-1]
        target_coordinates = target_coordinates[::-1]
        values = np.flip(values, axis=axis)
    if np.any(np.diff(coordinates) <= 0):
        raise ValueError("Interpolation coordinates must be strictly monotonic")

    moved = np.moveaxis(values, axis, 0)
    original_shape = moved.shape[1:]
    matrix = moved.reshape(moved.shape[0], -1).astype(np.float64, copy=False)
    if log_space:
        _validate_positive("log-space interpolation input", matrix)
        matrix = np.log10(matrix)

    output = np.empty((target_coordinates.size, matrix.shape[1]), dtype=np.float32)
    chunk_size = 65536
    for start in range(0, matrix.shape[1], chunk_size):
        stop = min(start + chunk_size, matrix.shape[1])
        result = _interpolate_series(
            coordinates, target_coordinates, matrix[:, start:stop], method
        )
        if log_space:
            result = 10.0**result
        output[:, start:stop] = result.astype(np.float32, copy=False)

    reshaped = output.reshape((target_coordinates.size, *original_shape))
    if reversed_axis:
        reshaped = reshaped[::-1]
    return np.moveaxis(reshaped, 0, axis)


def _channel_methods(args, channel: str) -> dict[str, str]:
    methods = dict(args.interpolation_default)
    methods.update(args.interpolation_channels.get(channel, {}))
    for axis_name in ("x", "y", "z"):
        if methods[axis_name] not in _INTERPOLATION_METHODS:
            raise ValueError(
                f"Invalid {axis_name} interpolation method for {channel}: "
                f"{methods[axis_name]!r}"
            )
    return methods


def _resample_cube(
    values: np.ndarray,
    old_coordinates: tuple[np.ndarray, np.ndarray, np.ndarray],
    new_coordinates: tuple[np.ndarray, np.ndarray, np.ndarray],
    methods: dict[str, str],
    *,
    log_space: bool,
) -> np.ndarray:
    result = values
    for axis, axis_name in enumerate(("x", "y", "z")):
        if np.array_equal(old_coordinates[axis], new_coordinates[axis]):
            continue
        result = _interpolate_axis(
            result,
            old_coordinates[axis],
            new_coordinates[axis],
            axis,
            methods[axis_name],
            log_space=log_space,
        )
    return result


def _resample_atmosphere(
    args,
    fields: dict[str, np.ndarray],
    height_m: np.ndarray,
    dx_m: float,
    dy_m: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, float, float]:
    """Upsample all spatial dimensions while preserving the physical extent."""
    nx, ny, nz = fields["temperature"].shape
    source_sizes = {"x": nx, "y": ny, "z": nz}
    old_coordinates = (
        np.arange(nx, dtype=np.float64) * dx_m,
        np.arange(ny, dtype=np.float64) * dy_m,
        np.asarray(height_m, dtype=np.float64),
    )
    new_coordinates_list = []
    for coordinates, axis in zip(old_coordinates, ("x", "y", "z")):
        target_spacing = args.upsample_target_spacing_m.get(axis)
        if target_spacing is not None:
            target_spacing = float(target_spacing)
            if not math.isfinite(target_spacing) or target_spacing <= 0:
                raise ValueError(
                    f"Target spacing for {axis} must be finite and positive"
                )
            span = abs(float(coordinates[-1] - coordinates[0]))
            target_size = int(math.floor(span / target_spacing + 1e-12)) + 1
            if target_size < source_sizes[axis]:
                native_mean = span / max(1, source_sizes[axis] - 1)
                raise ValueError(
                    f"Target {axis} spacing {target_spacing:g} m would downsample "
                    f"the axis (mean native spacing {native_mean:g} m)"
                )
            direction = 1.0 if coordinates[-1] >= coordinates[0] else -1.0
            target_coordinates = (
                coordinates[0]
                + direction * target_spacing * np.arange(target_size, dtype=np.float64)
            )
        else:
            target_size = _target_axis_size(
                source_sizes[axis],
                args.upsample_factors.get(axis),
                args.upsample_target_shape.get(axis),
            )
            target_coordinates = np.linspace(
                coordinates[0], coordinates[-1], target_size
            )
        new_coordinates_list.append(target_coordinates)
    new_coordinates = tuple(new_coordinates_list)
    target_sizes = {
        axis: coordinates.size
        for axis, coordinates in zip(("x", "y", "z"), new_coordinates)
    }

    result = {}
    for channel, values in fields.items():
        result[channel] = _resample_cube(
            values,
            old_coordinates,
            new_coordinates,
            _channel_methods(args, channel),
            log_space=channel in args.log_interpolation_channels,
        )

    new_dx = dx_m if target_sizes["x"] == 1 else abs(
        new_coordinates[0][1] - new_coordinates[0][0]
    )
    new_dy = dy_m if target_sizes["y"] == 1 else abs(
        new_coordinates[1][1] - new_coordinates[1][0]
    )
    print(
        "Resampled atmosphere: "
        f"{(nx, ny, nz)} -> "
        f"{(target_sizes['x'], target_sizes['y'], target_sizes['z'])}"
    )
    return result, new_coordinates[2], float(new_dx), float(new_dy)


def _validate_atmosphere(args, fields: dict[str, np.ndarray]) -> None:
    """Check finiteness, positivity, and configured adjacent-cell continuity."""
    spatial_shapes = {field.shape[:3] for field in fields.values()}
    if len(spatial_shapes) != 1:
        raise ValueError(
            f"Atmosphere channels have inconsistent spatial shapes: {spatial_shapes}"
        )
    for channel, values in fields.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{channel} contains NaN or Inf")
    for channel in args.positive_channels:
        if channel not in fields:
            raise ValueError(f"Unknown positive-validation channel {channel!r}")
        _validate_positive(channel, fields[channel])

    for channel, thresholds in args.continuity_max_log10_step.items():
        if channel not in fields:
            raise ValueError(f"Unknown continuity-validation channel {channel!r}")
        _validate_positive(channel, fields[channel])
        logged = np.log10(fields[channel].astype(np.float64, copy=False))
        for axis, axis_name in enumerate(("x", "y", "z")):
            if axis_name not in thresholds or logged.shape[axis] < 2:
                continue
            threshold = float(thresholds[axis_name])
            if not math.isfinite(threshold) or threshold <= 0:
                raise ValueError(
                    f"Continuity threshold for {channel}.{axis_name} must be positive"
                )
            jumps = np.abs(np.diff(logged, axis=axis))
            maximum = float(jumps.max())
            if maximum > threshold:
                location = np.unravel_index(int(jumps.argmax()), jumps.shape)
                raise ValueError(
                    f"{channel} is discontinuous along {axis_name}: maximum adjacent "
                    f"jump is {maximum:.6g} dex at {location}, exceeding {threshold:g} dex"
                )


def _apply_temperature_floor_and_calculate_ne(
    args, electron_source: str, fields: dict[str, np.ndarray]
) -> np.ndarray:
    """Floor final-grid temperature, then derive or select electron density."""
    temp = fields["temperature"]
    floor_count = int(np.count_nonzero(temp < args.temperature_floor_k))
    np.maximum(temp, np.float32(args.temperature_floor_k), out=temp)
    if floor_count:
        print(
            f"Raised {floor_count} interpolated temperature cells to "
            f"{args.temperature_floor_k:g} K"
        )

    if electron_source == "lgp":
        ne = _electron_density_from_witt_pgas(args, temp, fields["pgas"])
    elif electron_source == "lgr":
        ne = _electron_density_from_witt_rho(args, temp, fields["rho"])
    elif electron_source == "lgne":
        ne = fields["ne"]
    else:
        raise ValueError(f"Unknown electron-density source {electron_source!r}")
    _validate_positive("electron density", ne)
    fields["ne"] = ne
    return ne


def _rotate_spatial_dimensions(arr: np.ndarray) -> np.ndarray:
    """Apply ``[::-1, :].T`` to every horizontal plane of an (x, y, z) cube."""
    return np.transpose(arr[::-1, :, :], (1, 0, 2))


def _reverse_to_target_coordinates(
    temp: np.ndarray,
    rho: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    ne: np.ndarray,
    height_m: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Rotate horizontal planes and reverse depth for the target coordinates."""
    return (
        _rotate_spatial_dimensions(temp[:, :, ::-1]),
        _rotate_spatial_dimensions(rho[:, :, ::-1]),
        _rotate_spatial_dimensions(vx[:, :, ::-1]),
        _rotate_spatial_dimensions(vy[:, :, ::-1]),
        _rotate_spatial_dimensions(vz[:, :, ::-1]),
        _rotate_spatial_dimensions(ne[:, :, ::-1]),
        height_m[::-1],
    )


def _write_ffno_hdf5(
    output: Path,
    *,
    temp: np.ndarray,
    rho: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    ne: np.ndarray,
    height_m: np.ndarray,
    dx_m: float,
    dy_m: float,
    source_folder: Path,
    simulation_code: str,
    simulation_name: str,
    snap: str,
    electron_density_source: str,
    compression: int,
    overwrite: bool,
) -> None:
    """Build and write one FFNO solving-set HDF5 file.

    Atmosphere inputs use SI units: K, kg m^-3, m s^-1, m^-3, and m.
    Velocity channels are converted to km s^-1 and z_scale is converted to Mm
    for compatibility with FFNO training data.
    """
    _validate_positive("temperature", temp)
    _validate_positive("electron density", ne)
    _validate_positive("density", rho)

    u = _import_astropy_units()
    vx_kms = _convert_values(vx, u.m / u.s, u.km / u.s)
    vy_kms = _convert_values(vy, u.m / u.s, u.km / u.s)
    vz_kms = _convert_values(vz, u.m / u.s, u.km / u.s)

    inputs = np.stack(
        [
            np.log10(temp),
            vx_kms,
            vy_kms,
            vz_kms,
            np.log10(ne),
            np.log10(rho),
        ],
        axis=0,
    )
    inputs = np.transpose(inputs, (0, 3, 1, 2)).astype(np.float32, copy=False)

    height_mm = _convert_values(
        np.asarray(height_m, dtype=np.float32), u.m, u.Mm
    )
    z_native = np.broadcast_to(
        height_mm.reshape(-1, 1, 1),
        (inputs.shape[1], inputs.shape[2], inputs.shape[3]),
    )

    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    h5py = _import_h5py()
    mode = "w" if overwrite else "x"
    with h5py.File(output, mode) as f:
        f.create_dataset(
            "inputs",
            data=inputs[None, ...],
            compression="gzip",
            compression_opts=compression,
            shuffle=True,
        )
        f.create_dataset(
            "z_scale",
            data=z_native[None, ...],
            compression="gzip",
            compression_opts=compression,
            shuffle=True,
        )
        f.create_dataset("dx", data=np.array([dx_m], dtype=np.float32))
        f.create_dataset("dy", data=np.array([dy_m], dtype=np.float32))
        f.attrs["N"] = 1
        f.attrs["Cin"] = inputs.shape[0]
        f.attrs["D"] = inputs.shape[1]
        f.attrs["nx"] = inputs.shape[2]
        f.attrs["ny"] = inputs.shape[3]
        f.attrs["native_depth"] = inputs.shape[1]
        f.attrs["source_format"] = "MURaM FITS"
        f.attrs["source_folder"] = os.fspath(source_folder)
        f.attrs["simulation_code"] = simulation_code
        f.attrs["simulation_name"] = simulation_name
        f.attrs["snap"] = snap
        f.attrs["electron_density_source"] = electron_density_source
        f.attrs["velocity_unit"] = "km s^-1"

    print(f"Wrote {output}")
    print(f"inputs shape: {(1,) + inputs.shape}")
    print(f"z_scale shape: {(1,) + z_native.shape}")
    print(f"dx={dx_m} m, dy={dy_m} m")
    print(f"height range: {float(height_m.min())} .. {float(height_m.max())} m")


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
    nh: np.ndarray | None,
    dx_m: float,
    dy_m: float,
    height_m: np.ndarray,
    overwrite: bool,
) -> None:
    """Write the converter's SI-valued atmosphere in Multi3D units.

    Expected input units
    --------------------
    temp
        Temperature in kelvin (K).
    rho
        Mass density in kilograms per cubic metre (kg m^-3).
    vx, vy, vz
        Velocity components in metres per second (m s^-1), the converter's
        internal velocity unit.
    ne
        Electron number density in inverse cubic metres (m^-3).
    nh
        Optional hydrogen populations in inverse cubic metres, ordered as
        ``(nx, ny, nz, 6)``. ``None`` writes an atmosphere without populations.
    dx_m, dy_m
        Horizontal grid spacing in metres (m).
    height_m
        Vertical grid coordinates in metres (m).
    Units written
    -------------
    The Multi3D atmosphere fields are written as: ``temp`` in K (unchanged),
    ``rho`` in g cm^-3, ``ne`` and ``nh`` in cm^-3, and ``vx``, ``vy``,
    ``vz`` in km s^-1. If ``mesh_path`` is provided, its x, y, and z
    coordinates are written in centimetres. Astropy performs all conversions
    from the SI input units listed above.

    All scalar atmosphere arrays must have the same ``(nx, ny, nz)`` shape;
    ``nh`` has one additional trailing level axis. The ``atmos_path``,
    ``mesh_path``, and ``overwrite`` parameters control file output only and
    therefore have no physical units.
    """
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

    u = _import_astropy_units()
    nx, ny, nz = temp.shape
    if nh is not None and nh.shape != (nx, ny, nz, 6):
        raise ValueError(
            "Hydrogen populations must have shape "
            f"{(nx, ny, nz, 6)}, got {nh.shape}"
        )
    read_nh = nh is not None
    atmos = Multi3dAtmos(
        os.fspath(atmos_path), nx, ny, nz, mode="w+", read_nh=read_nh
    )
    atmos.ne[:] = _convert_values(ne, u.m**-3, u.cm**-3).astype(
        np.float32, copy=False
    )
    atmos.temp[:] = temp.astype(np.float32, copy=False)
    atmos.vx[:] = _convert_values(vx, u.m / u.s, u.km / u.s).astype(
        np.float32, copy=False
    )
    atmos.vy[:] = _convert_values(vy, u.m / u.s, u.km / u.s).astype(
        np.float32, copy=False
    )
    atmos.vz[:] = _convert_values(vz, u.m / u.s, u.km / u.s).astype(
        np.float32, copy=False
    )
    atmos.rho[:] = _convert_values(rho, u.kg / u.m**3, u.g / u.cm**3).astype(
        np.float32, copy=False
    )
    if nh is not None:
        atmos.nh[:] = _convert_values(nh, u.m**-3, u.cm**-3).astype(
            np.float32, copy=False
        )

    if mesh_path is not None:
        x_m = np.arange(nx, dtype=np.float64) * dx_m
        y_m = np.arange(ny, dtype=np.float64) * dy_m
        x = _convert_values(x_m, u.m, u.cm)
        y = _convert_values(y_m, u.m, u.cm)
        z = _convert_values(np.asarray(height_m, dtype=np.float64), u.m, u.cm)

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
    folder = Path(args.folder)
    snap = str(args.snap)
    electron_source, _ = _find_electron_density_source(
        folder, args.simulation_code, args.simulation_name, snap
    )
    hydrogen_population_paths = _find_hydrogen_population_paths(
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
    height_scale = _unit_scale_from_header(
        temp_header, "CUNIT3", "m", "vertical coordinate"
    )
    height_m = np.asarray(height, dtype=np.float64).reshape(-1) * height_scale
    if height_m.size != nz_full:
        raise ValueError(
            f"Height size {height_m.size} does not match FITS z dimension {nz_full}"
        )

    x_slice = _resolve_slice(
        args.x_slice, args.start_x, args.end_x, nx_full, "x"
    )
    y_slice = _resolve_slice(
        args.y_slice, args.start_y, args.end_y, ny_full, "y"
    )
    z_slice = _resolve_slice(args.z_slice, None, None, nz_full, "z")
    z_candidates = np.arange(nz_full, dtype=np.intp)[z_slice]
    z_indices = _height_indices(
        height_m,
        args.height_min_m,
        args.height_max_m,
        candidates=z_candidates,
    )

    args._z_indices = z_indices
    args._x_slice = x_slice
    args._y_slice = y_slice

    dx = args.dx
    dy = args.dy
    if dx is None:
        x_scale = _unit_scale_from_header(
            temp_header, "CUNIT1", "m", "x coordinate"
        )
        dx = _maybe_infer_spacing(temp_header, "CDELT1", x_scale)
    if dy is None:
        y_scale = _unit_scale_from_header(
            temp_header, "CUNIT2", "m", "y coordinate"
        )
        dy = _maybe_infer_spacing(temp_header, "CDELT2", y_scale)
    if dx is None or dy is None:
        raise ValueError(
            "Could not infer dx/dy from FITS CDELT1/CDELT2. Set [grid].dx_m "
            "and [grid].dy_m in meters."
        )
    dx *= x_slice.step
    dy *= y_slice.step

    temp = _read_log_quantity(
        folder,
        args.simulation_code,
        args.simulation_name,
        "lgtg",
        snap,
        z_indices,
        x_slice,
        y_slice,
        si_unit="K",
        quantity_name="temperature",
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
        si_unit="kg / m3",
        quantity_name="mass density",
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
        si_unit="m / s",
        quantity_name="x velocity",
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
        si_unit="m / s",
        quantity_name="y velocity",
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
        si_unit="m / s",
        quantity_name="z velocity",
    )
    _validate_positive("temperature", temp)
    _validate_positive("density", rho)

    ne = None
    pgas = None
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
            si_unit="1 / m3",
            quantity_name="electron density",
        )
        _validate_positive("electron density", ne)
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
            si_unit="Pa",
            quantity_name="gas pressure",
        )
        _validate_positive("gas pressure", pgas)
        electron_density_source = "witt-pgas"
    else:
        electron_density_source = "witt-rho"

    height_selected = height_m[z_indices]
    # The target convention rotates each horizontal plane with [::-1, :].T and
    # stores index 0 at the top.
    source_fields = {
        "temperature": temp,
        "rho": rho,
        "vx": vx,
        "vy": vy,
        "vz": vz,
    }
    if ne is not None:
        source_fields["ne"] = ne
    if pgas is not None:
        source_fields["pgas"] = pgas
    resampling_fields = {
        channel: _rotate_spatial_dimensions(values[:, :, ::-1])
        for channel, values in source_fields.items()
    }
    height_selected = height_selected[::-1]
    # The horizontal rotation exchanges the x and y axes and their spacing.
    dx, dy = dy, dx

    nh = None
    if args.multi3d_atmos_out and hydrogen_population_paths is not None:
        print("Detected lgn1 through lgn6; reading hydrogen populations.")
        nh = np.empty((*resampling_fields["temperature"].shape, 6), dtype=np.float32)
        for level in range(1, 7):
            population = _read_log_quantity(
                folder,
                args.simulation_code,
                args.simulation_name,
                f"lgn{level}",
                snap,
                z_indices,
                x_slice,
                y_slice,
                si_unit="1 / m3",
                quantity_name=f"hydrogen level {level} population",
            )
            nh[..., level - 1] = _rotate_spatial_dimensions(
                population[:, :, ::-1]
            )

    if nh is not None:
        resampling_fields["hydrogen_populations"] = nh
    resampling_fields, height_selected, dx, dy = _resample_atmosphere(
        args, resampling_fields, height_selected, dx, dy
    )

    # Apply the temperature floor only on the final interpolated grid. EOS-derived
    # electron density must use this floored temperature and the interpolated rho
    # or gas pressure, rather than values calculated on the source grid.
    ne = _apply_temperature_floor_and_calculate_ne(
        args, electron_source, resampling_fields
    )

    # Validate the complete final-grid atmosphere before narrowing it to the six
    # FFNO input channels. This keeps internal EOS inputs such as pgas available
    # to configurable positivity and continuity checks.
    _validate_atmosphere(args, resampling_fields)
    fields = {channel: resampling_fields[channel] for channel in _OUTPUT_CHANNELS}
    temp = fields["temperature"]
    rho = fields["rho"]
    vx = fields["vx"]
    vy = fields["vy"]
    vz = fields["vz"]
    ne = fields["ne"]
    if nh is not None:
        nh = resampling_fields["hydrogen_populations"]

    _write_ffno_hdf5(
        Path(args.output),
        temp=temp,
        rho=rho,
        vx=vx,
        vy=vy,
        vz=vz,
        ne=ne,
        height_m=height_selected,
        dx_m=dx,
        dy_m=dy,
        source_folder=folder,
        simulation_code=args.simulation_code,
        simulation_name=args.simulation_name,
        snap=snap,
        electron_density_source=electron_density_source,
        compression=args.compression,
        overwrite=args.overwrite,
    )

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
            nh=nh,
            dx_m=dx,
            dy_m=dy,
            height_m=height_selected,
            overwrite=args.overwrite,
        )


def _check_keys(table: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(table) - allowed
    if unknown:
        raise ValueError(f"Unknown key(s) in [{name}]: {', '.join(sorted(unknown))}")


def _config_path(value: Any, base_dir: Path) -> str | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return os.fspath(path.resolve())


def _configured_slice(selection: dict[str, Any], axis: str) -> slice | None:
    value = selection.get(axis)
    if value is None:
        return None
    if isinstance(value, str):
        return _parse_slice_text(value)
    if not isinstance(value, dict):
        raise ValueError(f"[selection].{axis} must be a slice string or table")
    _check_keys(value, {"start", "stop", "step"}, f"selection.{axis}")
    return slice(value.get("start"), value.get("stop"), value.get("step"))


def load_config(config_path: Path) -> SimpleNamespace:
    """Load every converter setting from a TOML file."""
    config_path = config_path.expanduser().resolve()
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)
    _check_keys(
        config,
        {"input", "selection", "grid", "eos", "output", "resampling", "validation"},
        "root",
    )
    for required_table in ("input", "output"):
        if required_table not in config or not isinstance(config[required_table], dict):
            raise ValueError(f"Configuration requires a [{required_table}] table")

    input_config = config["input"]
    _check_keys(
        input_config,
        {"folder", "simulation_code", "simulation_name", "snap"},
        "input",
    )
    missing = {"folder", "simulation_name", "snap"} - set(input_config)
    if missing:
        raise ValueError(f"Missing required [input] key(s): {', '.join(sorted(missing))}")

    selection = config.get("selection", {})
    _check_keys(selection, {"x", "y", "z", "height_min_m", "height_max_m"}, "selection")
    grid = config.get("grid", {})
    _check_keys(grid, {"dx_m", "dy_m"}, "grid")
    eos = config.get("eos", {})
    _check_keys(
        eos,
        {
            "witt_path", "backend", "workers", "chunk_size", "pool_chunksize",
            "start_method", "show_progress",
        },
        "eos",
    )
    output = config["output"]
    _check_keys(
        output,
        {"hdf5", "multi3d_atmos", "multi3d_mesh", "compression", "overwrite"},
        "output",
    )
    if "hdf5" not in output:
        raise ValueError("Missing required [output] key: hdf5")

    resampling = config.get("resampling", {})
    _check_keys(
        resampling,
        {
            "factors",
            "target_shape",
            "target_spacing_m",
            "default",
            "channels",
            "log_space",
        },
        "resampling",
    )
    factors = dict(resampling.get("factors", {}))
    targets = dict(resampling.get("target_shape", {}))
    target_spacing_m = dict(resampling.get("target_spacing_m", {}))
    _check_keys(factors, {"x", "y", "z"}, "resampling.factors")
    _check_keys(targets, {"x", "y", "z"}, "resampling.target_shape")
    _check_keys(
        target_spacing_m, {"x", "y", "z"}, "resampling.target_spacing_m"
    )
    conflicts = (
        (set(factors) & set(targets))
        | (set(factors) & set(target_spacing_m))
        | (set(targets) & set(target_spacing_m))
    )
    if conflicts:
        raise ValueError(
            "Specify only one of factor, target_shape, or target_spacing_m "
            "for each axis: "
            + ", ".join(sorted(conflicts))
        )
    interpolation_default = {"x": "linear", "y": "linear", "z": "linear"}
    configured_default = dict(resampling.get("default", {}))
    _check_keys(configured_default, {"x", "y", "z"}, "resampling.default")
    interpolation_default.update(configured_default)
    interpolation_channels = {
        channel: dict(methods)
        for channel, methods in resampling.get("channels", {}).items()
    }
    _check_keys(
        interpolation_channels, set(_RESAMPLED_CHANNELS), "resampling.channels"
    )
    for channel, methods in interpolation_channels.items():
        _check_keys(methods, {"x", "y", "z"}, f"resampling.channels.{channel}")
    configured_log_space = dict(resampling.get("log_space", {}))
    _check_keys(
        configured_log_space,
        set(_RESAMPLED_CHANNELS),
        "resampling.log_space",
    )
    log_interpolation_channels = set(_DEFAULT_LOG_INTERPOLATED_CHANNELS)
    for channel, enabled in configured_log_space.items():
        if not isinstance(enabled, bool):
            raise ValueError(
                f"[resampling.log_space].{channel} must be true or false"
            )
        if enabled:
            log_interpolation_channels.add(channel)
        else:
            log_interpolation_channels.discard(channel)

    validation = config.get("validation", {})
    _check_keys(
        validation,
        {"temperature_floor_k", "positive_channels", "continuity_max_log10_step"},
        "validation",
    )
    temperature_floor_k = float(validation.get("temperature_floor_k", 3250.0))
    if not math.isfinite(temperature_floor_k) or temperature_floor_k <= 0:
        raise ValueError("[validation].temperature_floor_k must be finite and positive")
    positive_channels = tuple(
        validation.get("positive_channels", ["temperature", "rho", "ne"])
    )
    continuity = {
        channel: dict(thresholds)
        for channel, thresholds in validation.get(
            "continuity_max_log10_step",
            {
                "rho": {"x": 3.0, "y": 3.0, "z": 3.0},
                "ne": {"x": 3.0, "y": 3.0, "z": 3.0},
            },
        ).items()
    }
    for channel, thresholds in continuity.items():
        _check_keys(thresholds, {"x", "y", "z"}, f"validation.continuity_max_log10_step.{channel}")

    backend = str(eos.get("backend", "auto"))
    if backend not in {"auto", "cpp", "python"}:
        raise ValueError("[eos].backend must be auto, cpp, or python")
    default_start = "fork" if "fork" in mp.get_all_start_methods() else mp.get_start_method()
    start_method = str(eos.get("start_method", default_start))
    if start_method not in mp.get_all_start_methods():
        raise ValueError(
            f"Unsupported [eos].start_method {start_method!r}; "
            f"choose from {mp.get_all_start_methods()}"
        )

    base_dir = config_path.parent
    multi3d_atmos = _config_path(output.get("multi3d_atmos"), base_dir)
    multi3d_mesh = _config_path(output.get("multi3d_mesh"), base_dir)
    if multi3d_mesh is not None and multi3d_atmos is None:
        raise ValueError("[output].multi3d_mesh requires [output].multi3d_atmos")

    return SimpleNamespace(
        folder=_config_path(input_config["folder"], base_dir),
        simulation_code=str(input_config.get("simulation_code", "MURaM")),
        simulation_name=str(input_config["simulation_name"]),
        snap=str(input_config["snap"]),
        output=_config_path(output["hdf5"], base_dir),
        start_x=None,
        end_x=None,
        start_y=None,
        end_y=None,
        x_slice=_configured_slice(selection, "x"),
        y_slice=_configured_slice(selection, "y"),
        z_slice=_configured_slice(selection, "z"),
        height_min_m=float(selection.get("height_min_m", -np.inf)),
        height_max_m=float(selection.get("height_max_m", np.inf)),
        dx=None if grid.get("dx_m") is None else float(grid["dx_m"]),
        dy=None if grid.get("dy_m") is None else float(grid["dy_m"]),
        witt_path=_config_path(eos.get("witt_path"), base_dir),
        show_eos_progress=bool(eos.get("show_progress", False)),
        eos_backend=backend,
        eos_workers=None if eos.get("workers") is None else int(eos["workers"]),
        eos_chunk_size=int(eos.get("chunk_size", 4096)),
        eos_pool_chunksize=int(eos.get("pool_chunksize", 1)),
        eos_start_method=start_method,
        multi3d_atmos_out=multi3d_atmos,
        multi3d_mesh_out=multi3d_mesh,
        compression=int(output.get("compression", 4)),
        overwrite=bool(output.get("overwrite", False)),
        upsample_factors=factors,
        upsample_target_shape=targets,
        upsample_target_spacing_m=target_spacing_m,
        interpolation_default=interpolation_default,
        interpolation_channels=interpolation_channels,
        log_interpolation_channels=frozenset(log_interpolation_channels),
        temperature_floor_k=temperature_floor_k,
        positive_channels=positive_channels,
        continuity_max_log10_step=continuity,
    )


def _main(argv: list[str]) -> None:
    if len(argv) != 2 or argv[1] in {"-h", "--help"}:
        print(
            f"Usage: {Path(argv[0]).name} CONFIG.toml\n"
            "All input, crop, resampling, validation, EOS, and output settings "
            "are read from CONFIG.toml."
        )
        if len(argv) == 2:
            return
        raise SystemExit(2)
    convert(load_config(Path(argv[1])))


if __name__ == "__main__":
    _main(sys.argv)
