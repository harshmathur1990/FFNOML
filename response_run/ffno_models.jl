# Run-local model files for scripts/ffno_population_tangents.jl on eagle7: the paths of paths.jl plus the FFNO
# checkpoints and level names of examples/inversion_run/model_factory.jl.
include(joinpath(@__DIR__,"paths.jl"))
const CHECKPOINTS = Dict(
    :H => "/mn/stornext/d9/data/harshm/FFNOML/training_FFNO3D_zscale_expand_lognlte/3D_sim_train_H.pt",
    :CA => "/mn/stornext/d9/data/harshm/FFNOML/training_FFNO3D_zscale_expand_lognlte/3D_sim_train_CA.pt",
)
const LEVEL_NAMES = Dict(species => Tuple("$species level $index" for index in 1:6) for species in (:H, :CA))
