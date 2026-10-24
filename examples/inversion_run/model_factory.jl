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
const EOS_LIBRARY = joinpath(RUN_DIR, "inputs", "wittmann", "libwitt_ffno.$(Libdl.dlext)")
const PARTITION_FUNCTIONS = required_path("inputs", "pf_Kurucz.input")
const FFNO_INVERSION_ASSETS = Dict(
    "H checkpoint" => CHECKPOINTS[:H],
    "Ca checkpoint" => CHECKPOINTS[:CA],
    "H atom" => ATOM_FILES[:H],
    "Ca atom" => ATOM_FILES[:CA],
    "partition functions" => PARTITION_FUNCTIONS,
)
const FFNO_INVERSION_GENERATED_ASSETS = Dict("EOS library" => EOS_LIBRARY)

const LEVELS = Dict(:H => 6, :CA => 6)
const LEVEL_NAMES = Dict(
    species => Tuple("$species level $index" for index in 1:count)
    for (species, count) in LEVELS
)

function build_production_model(config, distributed, workspace, context)
    metadata(species) = PopulationMetadata(
        FFNO_INPUT_CHANNELS,
        LEVEL_NAMES[species],
        get(ENV, "FFNO_$(species)_CHECKPOINT_HASH", "unrecorded-example-checkpoint"),
    )
    specs = [FSDPModelSpec(species, CHECKPOINTS[species], metadata(species))
             for species in (:H, :CA)]

    # Synthesis uses the Wittmann library; the equilibrium (pressure, density, electron density, heights) comes from
    # HSEMHS, configured in [equilibrium] (HSEMHS sets the pressure boundary). prepare_equilibrium checks that the
    # HSEMHS EOS table was built with the same partition functions and abundances as this library.
    eos = WittmannEOS(EOS_LIBRARY, PARTITION_FUNCTIONS)
    equilibrium = prepare_equilibrium(config, distributed, context; synthesis_eos=eos)

    diagnostics = get(ENV, "FFNO_GPU_DIAGNOSTICS_DIR", joinpath(RUN_DIR, "diagnostics"))
    populations = try
        launch_fsdp_population_models(
            specs,
            context;
            timeout_seconds=max(180.0, config.parallel.gpu_connect_timeout_seconds),
            diagnostics_directory=diagnostics,
        )
    catch
        close_equilibrium!(equilibrium)
        rethrow()
    end

    try
        # Muspel caches are built from a thermodynamically complete initial state.
        outcome = reconstruct_equilibrium!(equilibrium, distributed, context)
        outcome.valid || throw(ArgumentError(
            "the initial atmosphere has no valid equilibrium: $(outcome.reason); $(join(outcome.messages, "; "))"))
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
            equilibrium,
            CapabilityManifest(),
            prepared_equilibrium=outcome,
        )
    catch
        close_distributed_population_model!(populations, context)
        close_equilibrium!(equilibrium)
        rethrow()
    end
end

const FFNO_INVERSION_FACTORY = InversionModelFactory(LEVELS, build_production_model)
(;factory=FFNO_INVERSION_FACTORY,assets=FFNO_INVERSION_ASSETS,
  generated_assets=FFNO_INVERSION_GENERATED_ASSETS)
