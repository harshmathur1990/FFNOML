# Run-local inputs for scripts/column_response.jl (the parts of model_factory.jl it needs, without the GPU service).
using Libdl
const RUN_DIR = abspath(get(ENV, "FFNOML_RUN_DIR", @__DIR__))
const ATOM_FILES = Dict(:H => joinpath(RUN_DIR, "inputs", "atoms", "atom.h6_tiago2.yaml"),
                        :CA => joinpath(RUN_DIR, "inputs", "atoms", "atom.ca2.yaml"))
const EOS_LIBRARY = joinpath(RUN_DIR, "inputs", "wittmann", "libwitt_ffno.$(Libdl.dlext)")
const PARTITION_FUNCTIONS = joinpath(RUN_DIR, "inputs", "pf_Kurucz.input")
