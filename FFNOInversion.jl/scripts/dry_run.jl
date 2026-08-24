using FFNOInversion

length(ARGS) == 1 || error("usage: julia --project scripts/dry_run.jl CONFIG.toml")
config = load_config(ARGS[1])
println(dry_run_summary(config))
