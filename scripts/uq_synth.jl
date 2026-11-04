#!/usr/bin/env julia
# -----------------------------------------------------------------------------
# uq_synth.jl
#
# Line profiles for every column with the same Muspel settings as Forward.jl
# (atom files, line index, level pair, background atoms, Voigt table), from
# either MULTI3D out_pop (truth) or a scripts/uq_predict.py file (det_log,
# log10 m^-3, [nx, ny, nz, nlev] in Python order). Forward.jl is not included
# because it requires Plots.
#
# Usage:
#   julia --threads=auto scripts/uq_synth.jl --atom H --mesh M --atmos A \
#       (--out-pop DIR/out_pop | --uq FILE.h5) --out OUT.h5 [--atom-dir DIR]
# Output: group <atom> with "intensity" [nλ, ny, nx] and "wave" (as Forward.jl).
# -----------------------------------------------------------------------------

using Muspel
using AtomicData
using HDF5
using Base.Threads

const ATOMS = Dict(
    "H"  => (atom_file = "atom.h6_tiago2.yaml", nlevels = 6, line_index = 5, lower_level = 2, upper_level = 3),
    "CA" => (atom_file = "atom.ca2.yaml",       nlevels = 6, line_index = 5, lower_level = 3, upper_level = 5),
)
const VOIGT = (a_min = 1f-4, a_max = 1f1, a_n = 20000, v_min = 0f0, v_max = 5f2, v_n = 2500)
const BACKGROUND = ["Al.yaml", "C.yaml", "Ca.yaml", "Fe.yaml", "H_6.yaml", "He.yaml", "KI.yaml", "Mg.yaml",
                    "N.yaml", "Na.yaml", "NiI.yaml", "O.yaml", "S.yaml", "Si.yaml"]

function parse_args(args)
    opts = Dict{String,String}("--atom-dir" => "/mn/stornext/u3/harshm/Documents/WorkRepo/multi3d/input/atoms")
    i = 1
    while i <= length(args)
        i < length(args) || error("missing value for $(args[i])")
        opts[args[i]] = args[i + 1]
        i += 2
    end
    for k in ("--atom", "--mesh", "--atmos", "--out")
        haskey(opts, k) || error("$k is required")
    end
    xor(haskey(opts, "--out-pop"), haskey(opts, "--uq")) || error("give exactly one of --out-pop, --uq")
    return opts
end

# (nz, ny, nx, nlev) populations in m^-3, the layout read_pops_multi3d returns
function load_populations(opts, atmos, nlevels)
    if haskey(opts, "--out-pop")
        nlte, _ = read_pops_multi3d(opts["--out-pop"], atmos.nx, atmos.ny, atmos.nz, nlevels)
        return nlte
    end
    raw = h5open(f -> read(f["det_log"]), opts["--uq"], "r")  # Julia order (nlev, nz, ny, nx)
    return Float32.(exp10.(permutedims(raw, (2, 3, 4, 1))))
end

# Same as Forward.jl synthesize_intensity_3d
function synthesize(atms, atom, cfg, pops)
    my_line = atom.lines[cfg.line_index]
    a = LinRange(VOIGT.a_min, VOIGT.a_max, VOIGT.a_n)
    v = LinRange(VOIGT.v_min, VOIGT.v_max, VOIGT.v_n)
    voigt_itp = create_voigt_itp(a, v)
    σ_itp = get_σ_itp(atms, my_line.λ0, [joinpath(AtomicData.get_atom_dir(), b) for b in BACKGROUND])

    intensity = Array{Float32,3}(undef, my_line.nλ, atms.ny, atms.nx)
    ncolumns = atms.nx * atms.ny
    nworkers = min(Threads.nthreads(), ncolumns)
    n_u = pops[:, :, :, cfg.upper_level]
    n_l = pops[:, :, :, cfg.lower_level]
    Threads.@threads :static for worker in 1:nworkers
        buf = RTBuffer(atms.nz, my_line.nλ, Float32)
        for column in worker:nworkers:ncolumns
            j = mod1(column, atms.ny)
            i = fld(column - 1, atms.ny) + 1
            calc_line_prep!(my_line, buf, atms[:, j, i], σ_itp)
            calc_line_1D!(my_line, buf, my_line.λ, atms[:, j, i], n_u[:, j, i], n_l[:, j, i], voigt_itp)
            intensity[:, j, i] = buf.intensity
        end
    end
    return intensity, my_line.λ
end

function main(args)
    opts = parse_args(args)
    name = opts["--atom"]
    cfg = ATOMS[name]
    t0 = time()
    atmos = read_atmos_multi3d(opts["--mesh"], opts["--atmos"])
    atom = Muspel.read_atom(joinpath(opts["--atom-dir"], cfg.atom_file))
    pops = load_populations(opts, atmos, cfg.nlevels)
    size(pops) == (atmos.nz, atmos.ny, atmos.nx, cfg.nlevels) ||
        error("populations $(size(pops)) do not match atmosphere ($(atmos.nz), $(atmos.ny), $(atmos.nx))")
    intensity, wave = synthesize(atmos, atom, cfg, pops)
    tmp = opts["--out"] * ".tmp"
    h5open(tmp, "w") do f
        g = create_group(f, name)
        g["intensity"] = intensity
        g["wave"] = wave
        attrs(f)["source"] = get(opts, "--out-pop", get(opts, "--uq", ""))
        attrs(f)["atmos"] = opts["--atmos"]
    end
    mv(tmp, opts["--out"]; force = true)
    println("wrote $(opts["--out"]) in $(round(time() - t0; digits = 1)) s with $(Threads.nthreads()) threads")
end

main(ARGS)
