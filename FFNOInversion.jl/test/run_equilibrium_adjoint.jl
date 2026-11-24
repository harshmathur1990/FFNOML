# Only the equilibrium-adjoint tests: julia --project=. -t 16 test/run_equilibrium_adjoint.jl  (from FFNOInversion.jl/)
using Test, FFNOInversion
cd(@__DIR__)
include("equilibrium_adjoint.jl")
