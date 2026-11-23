# FALC with the zaug H model (FFNOMLTrainingWeights/zaug/H) (H only), from the production FALC factory: H populations, run assets from response_run,
# the FSDP service from FFNO_FSDP_LAUNCHER.
using FFNOInversion
using Libdl
using Muspel

const R = "/mn/stornext/d9/data/harshm/FFNOML/response_run"
const CHECKPOINTS = Dict(:H => "/mn/stornext/d9/data/harshm/FFNOMLTrainingWeights/zaug/H/3D_sim_train_H.pt")
const ATOM_FILES = Dict(:H => joinpath(R, "inputs", "atoms", "atom.h6_tiago2.yaml"))
const EOS_LIBRARY = joinpath(R, "inputs", "wittmann", "libwitt_ffno.$(Libdl.dlext)")
const PARTITION_FUNCTIONS = joinpath(R, "inputs", "pf_Kurucz.input")
const LEVELS = Dict(:H => 6)

function build_falc_model(config, distributed, workspace, context)
    metadata(s) = PopulationMetadata(FFNO_INPUT_CHANNELS, Tuple("$s level $i" for i in 1:6), "unrecorded-example-checkpoint")
    eos = WittmannEOS(EOS_LIBRARY, PARTITION_FUNCTIONS)
    equilibrium = prepare_equilibrium(config, distributed, context; synthesis_eos=eos)
    launcher = get(ENV, "FFNO_FSDP_LAUNCHER", "")
    populations = launch_fsdp_population_models([FSDPModelSpec(s, CHECKPOINTS[s], metadata(s)) for s in (:H,)], context;
        population_mesh=population_mesh_options(config), timeout_seconds=900.0,
        diagnostics_directory=get(ENV, "FFNO_GPU_DIAGNOSTICS_DIR", "/mn/stornext/d9/data/harshm/ffnoml_runs/falc_zaug_native/diagnostics"),
        (isempty(launcher) ? (;) : (;launcher=launcher))...)
    outcome = reconstruct_equilibrium!(equilibrium, distributed, context)
    outcome.valid || error("no valid equilibrium: $(outcome.reason)")
    predict_distributed_populations!(FFNOInversion.cartesian_populations(workspace), populations, distributed, context)
    FFNOInversion.footprint_state!(workspace, distributed, context)
    synthesizer = build_synthesis_setup(config, FFNOInversion.synthesis_atmosphere(workspace, distributed),
        workspace.populations, eos; atom_files=ATOM_FILES, nlte_hydrogen=synthesis_hydrogen(equilibrium))
    HybridForwardModel(populations, NonPRD(), synthesizer, config.observation_model, equilibrium, CapabilityManifest(),
        prepared_equilibrium=outcome)
end

InversionModelFactory(LEVELS, build_falc_model)
