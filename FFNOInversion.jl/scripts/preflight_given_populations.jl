using FFNOInversion

# Checks [population.given] of a forward configuration before a run: each file's format and grid, and the orientation from tau500 against the
# atmosphere's temperature. --read also reads the populations and checks that they are finite and not negative, as the run does.
args=filter(!=("--read"),ARGS); read_all="--read" in ARGS
length(args)==2 || error("usage: julia --project=. scripts/preflight_given_populations.jl CONFIG.toml MODEL_FACTORY.jl [--read]")
config=load_config(args[1])
config.given_populations===nothing && error("$(args[1]) has no [population.given]")
include(abspath(args[2]))
isdefined(Main,:FFNO_INVERSION_FACTORY) || error("MODEL_FACTORY.jl must define FFNO_INVERSION_FACTORY::InversionModelFactory")
factory=getfield(Main,:FFNO_INVERSION_FACTORY)
levels=factory.population_levels isa Function ? factory.population_levels(config) : factory.population_levels
atmosphere=read_inversion_inputs(config).atmosphere
result=read_all ? read_given_populations(config.given_populations,atmosphere,levels) :
    preflight_given_populations(config.given_populations,atmosphere,levels)
foreach(note->println("GIVEN_POPULATIONS ",note),result.notes)
println("GIVEN_POPULATIONS preflight passed",read_all ? "" : " (populations not read: add --read)")
