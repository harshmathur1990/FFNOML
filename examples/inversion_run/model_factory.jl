using FFNOInversion
using Libdl
using Muspel

const RUN_DIR = abspath(get(ENV, "FFNOML_RUN_DIR", @__DIR__))

required_path(parts...) = begin
    path = joinpath(RUN_DIR, parts...)
    isfile(path) || error("Required inversion asset is missing: $path")
    path
end

const CHECKPOINTS = Dict(
    :H => required_path("training_FFNO3D_zscale_expand_lognlte", "3D_sim_train_H.pt"),
    :CA => required_path("training_FFNO3D_zscale_expand_lognlte", "3D_sim_train_CA.pt"),
)
const ATOM_FILES = Dict(
    :H => required_path("inputs", "atoms", "atom.h6_tiago2.yaml"),
    :CA => required_path("inputs", "atoms", "atom.ca2.yaml"),
)
const EOS_LIBRARY = required_path("inputs", "wittmann", "libwitt_ffno.$(Libdl.dlext)")
const PARTITION_FUNCTIONS = required_path("inputs", "pf_Kurucz.input")

const LEVELS = Dict(:H => 6, :CA => 6)
const LEVEL_NAMES = Dict(
    species => Tuple("$species level $index" for index in 1:count)
    for (species, count) in LEVELS
)

function build_production_model(config, distributed, workspace, local_pressure_top, context)
    metadata(species) = PopulationMetadata(
        FFNO_INPUT_CHANNELS,
        LEVEL_NAMES[species],
        get(ENV, "FFNO_$(species)_CHECKPOINT_HASH", "unrecorded-example-checkpoint"),
    )
    specs = [FSDPModelSpec(species, CHECKPOINTS[species], metadata(species))
             for species in (:H, :CA)]

    diagnostics = get(ENV, "FFNO_GPU_DIAGNOSTICS_DIR", joinpath(RUN_DIR, "diagnostics"))
    populations = launch_fsdp_population_models(
        specs,
        context;
        timeout_seconds=max(180.0, config.parallel.gpu_connect_timeout_seconds),
        diagnostics_directory=diagnostics,
    )

    eos = WittmannEOS(EOS_LIBRARY, PARTITION_FUNCTIONS)
    opacity = WittmannOpacity500(eos)
    top_density = parse(Float64, get(ENV, "FFNO_TOP_DENSITY_KG_M3", "1e-10"))
    boundary = HE3DBoundaryState(top_density, local_pressure_top, :top)
    force_options = ForceBalanceOptions()

    try
        # Muspel caches are built from a thermodynamically complete initial state.
        reconstruct_force_balance_distributed!(
            distributed, boundary, eos, opacity, context; options=force_options
        )
        predict_distributed_populations!(workspace.populations, populations, distributed, context)
        synthesizer = build_synthesis_setup(
            config,
            distributed.local_atmosphere,
            workspace.populations,
            eos;
            atom_files=ATOM_FILES,
        )
        return HybridForwardModel(
            populations,
            NonPRD(),
            synthesizer,
            config.observation_model,
            eos,
            opacity,
            boundary,
            force_options,
            CapabilityManifest(),
        )
    catch
        close_distributed_population_model!(populations, context)
        rethrow()
    end
end

const FFNO_INVERSION_FACTORY = InversionModelFactory(LEVELS, build_production_model)
