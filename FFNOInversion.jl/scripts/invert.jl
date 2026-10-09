using FFNOInversion

length(ARGS)==2 || error("usage: julia --project=. scripts/invert.jl CONFIG.toml MODEL_FACTORY.jl")
include(abspath(ARGS[2]))
isdefined(Main,:FFNO_INVERSION_FACTORY) || error(
    "MODEL_FACTORY.jl must define FFNO_INVERSION_FACTORY::InversionModelFactory")
factory=getfield(Main,:FFNO_INVERSION_FACTORY)
factory isa InversionModelFactory || error("FFNO_INVERSION_FACTORY has the wrong type")
run_inversion_files!(ARGS[1],factory)
