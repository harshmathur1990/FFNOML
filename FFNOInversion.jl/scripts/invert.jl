using FFNOInversion

# --finish: do not iterate; write the outputs of the L-BFGS checkpoint's parameters (a run stopped by its time limit)
finish="--finish" in ARGS; arguments=filter(!=("--finish"),ARGS)
length(arguments)==2 || error("usage: julia --project=. scripts/invert.jl [--finish] CONFIG.toml MODEL_FACTORY.jl")
include(abspath(arguments[2]))
isdefined(Main,:FFNO_INVERSION_FACTORY) || error(
    "MODEL_FACTORY.jl must define FFNO_INVERSION_FACTORY::InversionModelFactory")
factory=getfield(Main,:FFNO_INVERSION_FACTORY)
factory isa InversionModelFactory || error("FFNO_INVERSION_FACTORY has the wrong type")
run_inversion_files!(arguments[1],factory;finish=finish)
