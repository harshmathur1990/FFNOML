# What the FFNO receives on the population mesh from the z route and the log tau route (FALC): the mesh model wrapped around
# a recorder, on the atmospheres each run wrote. Writes falc_mesh_inputs.h5 and prints the two side by side by height.
# usage (hyades, after ~/loadgcc.sh): julia --project=FFNOInversion.jl response_run/falc/compare_mesh_inputs.jl
using FFNOInversion, HDF5, Printf
import HSEMHS
const R = joinpath(@__DIR__)
mutable struct Recorder <: AbstractPopulationModel
    seen::Any
end
FFNOInversion.predict_populations!(out, m::Recorder, a, cache=nothing) = (m.seen = deepcopy(a); fill!(out, 1.0); out)
function atmosphere(name)
    h5open(joinpath(R, name)) do f
        field(n) = permutedims(read(f[n])[1, :, :, :], (1, 3, 2))
        kind = haskey(f, "z_m") ? :z_m : :log10tau500
        axis = read(f[kind === :z_m ? "z_m" : "logtau_500"])
        grid = Grid3D(VerticalAxis(kind, axis), read(f["x_m"]), read(f["y_m"]))
        FFNOInversion.Atmosphere3D(grid, field("temperature"), field("vx"), field("vy"), field("vz"), field("vturb");
            pgas=field("pgas"), rho=field("rho"), ne=field("ne"), z=field("z"))
    end
end
eos = HSEMHS.load_table("/mn/stornext/d9/data/harshm/hsemhs_env112/work/e2e/inputs/eos_table_fullz.h5")
config = PopulationMeshConfig()
mesh = population_mesh(config)
runs = Dict{String,Any}()
for (label, file) in (("z", "falc_zmesh_atmosphere.h5"), ("ltau", "falc_ltau_vturb_atmosphere.h5"))
    a = atmosphere(file); inner = Recorder(nothing)
    model = MeshPopulationModel(inner, config, eos, 274.0; native_z=true)
    out = zeros(size(a.temperature)..., 1); predict_populations!(out, model, a)
    s = inner.seen
    runs[label] = (T=s.temperature[:, 3, 3], ne=s.ne[:, 3, 3], rho=s.rho[:, 3, 3], vz=s.vz[:, 3, 3], ztop=maximum(a.z[:, 3, 3]), zbot=minimum(a.z[:, 3, 3]),
        coltop=a.ne[1, 3, 3], rhotop=a.rho[1, 3, 3])
    @printf("%-5s levels %d, column z %.0f to %.0f km, top ne %.3e rho %.3e\n", label, length(mesh), minimum(a.z[:, 3, 3]) / 1e3, maximum(a.z[:, 3, 3]) / 1e3, a.ne[1, 3, 3], a.rho[1, 3, 3])
end
h5open(joinpath(R, "falc_mesh_inputs.h5"), "w") do f
    f["z_mesh_m"] = mesh
    for (k, v) in runs, n in (:T, :ne, :rho, :vz); f["$k/$n"] = getfield(v, n); end
end
println("   z_km |    T z      T ltau |  ne z/ltau  rho z/ltau |      ne z     rho z")
z, l = runs["z"], runs["ltau"]
for h in (14000, 10000, 6000, 5000, 4000, 3000, 2500, 2200, 2100, 2000, 1800, 1500, 1000, 700, 500, 250, 100, 0, -100, -200, -500, -1000)
    k = argmin(abs.(mesh .- h * 1e3))
    @printf("%7.0f | %8.0f %9.0f | %9.2e  %9.2e | %9.2e %9.2e\n", mesh[k] / 1e3, z.T[k], l.T[k], z.ne[k] / l.ne[k], z.rho[k] / l.rho[k], z.ne[k], z.rho[k])
end
