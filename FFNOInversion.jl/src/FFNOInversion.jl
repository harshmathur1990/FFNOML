module FFNOInversion

using LinearAlgebra
using Libdl
using Serialization
using Sockets
using TOML
import MPI

include("Types.jl")
include("Parallel.jl")
include("Nodes.jl")
include("EOS.jl")
include("Opacity500.jl")
include("ForceBalance.jl")
include("PopulationModels.jl")
include("LinePhysics.jl")
include("Synthesis.jl")
include("Observations.jl")
include("ForwardModel.jl")
include("Objectives.jl")
include("Regularization.jl")
include("DistributedForward.jl")
include("IO.jl")

export Grid3D, MagneticField3D, Atmosphere3D, HE3DBoundaryState
export ParallelOptions, ParallelContext, Tile2D, DistributedField
export serial_context, initialize_parallel, finalize_parallel!, isroot, barrier, mpi_broadcast
export allreduce_sum, allreduce_max, process_grid, tile_for_rank, local_tile, local_atmosphere
export distribute_field, gather_field, exchange_halos
export AbstractGPUCoordinator, RootGPUCoordinator, launch_gpu!
export parallel_provenance, write_parallel_provenance
export HE3DMode, MHSMode, select_force_balance
export AbstractEOS, WittmannEOS, IdealGasEOS, thermodynamics!
export AbstractOpacity500, WittmannOpacity500, ReferenceOpacity500, opacity500!
export ForceBalanceOptions, ForceBalanceDiagnostics, lorentz_force!, reconstruct_force_balance!
export StokesSet, SpectralCube, ObservationCube, CapabilityManifest
export NodeField, expand_nodes
export AbstractPopulationModel, predict_populations!, MockPopulationModel
export FFNO_INPUT_CHANNELS, PopulationMetadata, population_features, PythonFFNOModel, RecordedPopulationModel, load_python_ffno_model
export save_population_record, load_population_record
export AbstractRedistributionModel, NonPRD, MockPRD
export AbstractLineOpacityModel, AbstractFormalSolver, FFNOTransition, MuspelLineOpacityModel, MuspelFormalSolver, build_muspel_line_model
export KuruczLine, KuruczLTEModel, WittmannKuruczState, TabulatedOpacityModel
export KuruczLineCache, load_kurucz_linelist, select_kurucz_lines, add_opacity_emissivity!, planck_lambda
export AbstractSynthesizer, synthesize!, formal_solve!, ScalarFormalSolver, OpacityContributor, MixedIntensitySynthesizer, SynthesisCache, ThreadedSynthesisCache
export RegionSynthesisSetup, build_synthesis_setup
export MockIntensitySynthesizer, MockPolarizedSynthesizer
export AbstractObservationModel, apply_observation!, IdentityObservation
export GaussianPSFObservation, build_inversion_weights
export ForwardWorkspace, MockForwardModel, forward!
export DistributedAtmosphere, distribute_atmosphere, gather_atmosphere
export reconstruct_force_balance_distributed!, AbstractDistributedPopulationModel
export LocalDistributedPopulationModel, RootDistributedPopulationModel, predict_distributed_populations!
export HybridForwardWorkspace, HybridForwardModel, HybridForwardTimings, gather_spectrum, distributed_chi2, distributed_regularization_penalty
export distributed_memory_report
export distribute_observation
export ResidualLayout, residual!, VerticalRegularizationSpec, RegularizationSpec, regularization_penalty
export AtmosphereInputConfig, ObservedDataConfig, WeightInputConfig, OutputConfig, SynthesisGridConfig, SpectralSourceConfig, SpectralRegionConfig, wavelengths, RunConfig
export load_config, dry_run_summary
export checkpoint!, restore_checkpoint

end
