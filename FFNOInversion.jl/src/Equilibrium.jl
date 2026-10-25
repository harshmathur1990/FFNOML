"""Atmospheric equilibrium reconstruction behind one interface.

HSEMHS is the only production backend (`[equilibrium] backend = "HSEMHS"`); the
abstract interface also admits test doubles. A backend owns its solver state and
communicators; FFNO owns MPI, threads and the phase order (controls, equilibrium,
populations, synthesis, observation).

    backend = prepare_equilibrium(config, distributed, context)        # collective, once
    outcome = reconstruct_equilibrium!(backend, distributed, context)  # collective, every evaluation
    close_equilibrium!(backend)                                        # collective

`reconstruct_equilibrium!` fills `pgas`, `rho`, `ne`, `z`, `logtau500` and `chi500` of the
rank-local atmosphere from its temperature (and B) without changing T, B,
velocities or microturbulence, and returns an `EquilibriumOutcome` that is
identical on every rank. It does not throw for physics failures (EOS envelope,
non-finite or folded columns): `outcome.valid` is false and the fields are not
to be used.
"""
abstract type AbstractEquilibriumBackend end

function reconstruct_equilibrium! end
close_equilibrium!(::AbstractEquilibriumBackend) = nothing

"""Rank-identical result of one reconstruction. `detail` holds the backend's own
status (an `HSEMHS.EquilibriumStatus` for the HSEMHS backend)."""
struct EquilibriumOutcome
    valid::Bool
    reason::Symbol
    requested_model::Symbol
    effective_mode::Symbol
    steps::Int
    seconds::Float64
    messages::Vector{String}
    detail::Any
end

# ---------------------------------------------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------------------------------------------

"""Parsed `[equilibrium]` section. `mode` is derived from the native vertical
coordinate and `model` (brief §2): column_hse_z, column_hse_ltau, mhs_poisson_z,
mhs_poisson_ltau. The column options use HSEMHS's names, defaults and
semantics; HSEMHS is the authority on the pressure boundary."""
struct EquilibriumConfig
    backend::String
    model::Symbol
    mode::Symbol
    gravity_m_s2::Float64
    eos_table_file::String
    on_nonconvergence::Symbol
    column::Any                       # HSEMHS.ColumnZOptions or HSEMHS.ColumnTauOptions
    poisson::Any                      # nothing in Phase 3 (mhs arrives in Phase 4)
end

const _EQ_KEYS = ("backend","model","gravity_m_s2","eos_table_file","on_nonconvergence","column_z","column_ltau")
const _COLUMN_Z_KEYS = Dict("anchor"=>:anchor,"P_ref_pa"=>:P_ref,"z_ref_m"=>:z_ref,"n_sub"=>:n_sub,"adapt"=>:adapt,
    "rk_tol"=>:rk_tol,"n_sub_max"=>:n_sub_max,"tau_top"=>:tau_top,"H_max_m"=>:H_max,"mu"=>:mu)
const _COLUMN_LTAU_KEYS = Dict("align"=>:align)

function _option_value(field,value)
    field in (:anchor,:tau_top) && return Symbol(lowercase(String(value)))
    field===:align && return String(value)
    field===:adapt && return Bool(value)
    field in (:n_sub,:n_sub_max) && return Int(value)
    Float64(value)
end

function _options(raw,keys,section)
    unknown=setdiff(collect(Base.keys(raw)),collect(Base.keys(keys)))
    isempty(unknown) || throw(ArgumentError("unknown key(s) in [$section]: $(join(sort(unknown),", "))"))
    (; (keys[k]=>_option_value(keys[k],v) for (k,v) in raw)...)
end

"""Parse `[equilibrium]` for native coordinate `vertical_coordinate`, relative to `base` (paths)."""
function _equilibrium_config(raw,vertical_coordinate::Symbol,base::AbstractString)
    unknown=setdiff(collect(keys(raw)),collect(_EQ_KEYS))
    isempty(unknown) || throw(ArgumentError("unknown key(s) in [equilibrium]: $(join(sort(unknown),", "))"))
    backend=String(get(raw,"backend","HSEMHS"))
    backend=="HSEMHS" || throw(ArgumentError("equilibrium.backend must be \"HSEMHS\" (the only equilibrium backend), got \"$backend\""))
    model=Symbol(lowercase(String(_required(raw,"model"))))
    model in (:column_hse,:mhs) || throw(ArgumentError("equilibrium.model must be \"column_hse\" or \"mhs\", got \"$model\""))
    model===:mhs && throw(ArgumentError("equilibrium.model = \"mhs\" (mhs-poisson-z/-ltau) is not available yet (HSEMHS integration Phase 4)"))
    height=vertical_coordinate===:z_m
    mode=height ? (model===:column_hse ? :column_hse_z : :mhs_poisson_z) : (model===:column_hse ? :column_hse_ltau : :mhs_poisson_ltau)
    g=Float64(get(raw,"gravity_m_s2",274.0)); isfinite(g)&&g>0 || throw(ArgumentError("equilibrium.gravity_m_s2 must be positive"))
    table=String(_required(raw,"eos_table_file")); table=isabspath(table) ? table : normpath(joinpath(base,table))
    policy=Symbol(lowercase(String(get(raw,"on_nonconvergence","reject_trial"))))
    policy in (:reject_trial,:abort) || throw(ArgumentError("equilibrium.on_nonconvergence must be reject_trial or abort"))
    column=if height
        haskey(raw,"column_ltau") && throw(ArgumentError("[equilibrium.column_ltau] does not apply to a native-z run ($mode)"))
        cz=get(raw,"column_z",Dict{String,Any}())
        haskey(cz,"P_ref_pa") || throw(ArgumentError(
            "[equilibrium.column_z] P_ref_pa is required for $mode (HSEMHS column_hse.P_ref; the pressure boundary is set by HSEMHS)"))
        HSEMHS.ColumnZOptions(; _options(cz,_COLUMN_Z_KEYS,"equilibrium.column_z")...)
    else
        haskey(raw,"column_z") && throw(ArgumentError(
            "[equilibrium.column_z] does not apply to a native log-tau run ($mode): the τ-hydrostatic top closure sets the pressure"))
        HSEMHS.ColumnTauOptions(; _options(get(raw,"column_ltau",Dict{String,Any}()),_COLUMN_LTAU_KEYS,"equilibrium.column_ltau")...)
    end
    errors=column isa HSEMHS.ColumnTauOptions ? HSEMHS.option_errors(column,[0.0,1.0]) :
        filter(e->!occursin("z_ref",e),HSEMHS.option_errors(column,[1.0,0.0]))     # z_ref is checked against the axis at prepare
    isempty(errors) || throw(ArgumentError("[equilibrium] options: "*join(errors,"; ")))
    EquilibriumConfig(backend,model,mode,g,table,policy,column,nothing)
end

hsemhs_mode_name(mode::Symbol) = replace(String(mode),"_"=>"-")

# ---------------------------------------------------------------------------------------------------------------------
# HSEMHS backend
# ---------------------------------------------------------------------------------------------------------------------

"""HSEMHS in process: an embedded workspace on the inversion's communicator (a
duplicate of it, owned by HSEMHS) and this rank's x slab."""
mutable struct HSEMHSBackend <: AbstractEquilibriumBackend
    config::EquilibriumConfig
    workspace::Any                    # HSEMHS.EmbeddedWorkspace
    provenance::Dict{String,Any}
    closed::Bool
end

function _backend_comm(context::ParallelContext)
    context.enabled && return context.comm
    MPI.Initialized() || throw(ArgumentError(
        "the HSEMHS backend needs MPI; create the context with initialize_parallel (serial runs use a single-rank MPI)"))
    MPI.COMM_SELF
end

_file_sha256(path)=bytes2hex(open(SHA.sha256,path))

"""Provenance of the equilibrium EOS table and its compatibility with the
synthesis-side Wittmann EOS: the same partition functions are required (hard
mismatch → error); the table's other build settings are recorded."""
function _eos_provenance(table,synthesis_eos)
    meta=table.meta
    p=Dict{String,Any}("eos_table_sha256"=>HSEMHS.table_sha256(table),"T_sw"=>table.T_sw,"hot_branch"=>table.hot_branch,
        "abundances"=>string(get(meta,"abundances","")),"fix_gasc_h2"=>get(meta,"fix_gasc_h2",-1),
        "wittmann_source_sha256"=>string(get(meta,"source_sha256","")),"pf_sha256"=>string(get(meta,"pf_sha256","")),
        "chi500"=>"extinction per length [m^-1] at 5000 A, absorption + scattering")
    if synthesis_eos!==nothing
        synthesis_pf=_file_sha256(synthesis_eos.partition_functions)
        p["synthesis_pf_sha256"]=synthesis_pf
        synthesis_pf==p["pf_sha256"] || throw(ArgumentError(
            "the HSEMHS EOS table was built with partition functions $(p["pf_sha256"]) but synthesis uses $(synthesis_pf) ($(synthesis_eos.partition_functions))"))
        p["abundances"]=="wittmann_default" || throw(ArgumentError(
            "the HSEMHS EOS table uses abundances \"$(p["abundances"])\"; the synthesis Wittmann library uses its built-in defaults"))
    end
    p
end

"""
    prepare_equilibrium(config, distributed, context; synthesis_eos=nothing) -> HSEMHSBackend

Collective. Requires the `x_slabs` decomposition (HSEMHS solves on x slabs) for
multi-rank runs; checks that HSEMHS's slab equals this rank's tile, loads the EOS
table once per rank and, if `synthesis_eos` is given, compares partition functions
and abundances with the synthesis Wittmann EOS. Local failures on any rank
become the same error on every rank (HSEMHS two-stage protocol).
"""
function prepare_equilibrium(config,distributed::DistributedAtmosphere,context::ParallelContext;
        synthesis_eos=nothing)                                 # config::RunConfig (defined later, in IO.jl)
    eq=config.equilibrium
    eq===nothing && throw(ArgumentError("the configuration has no [equilibrium] section"))
    prepare_equilibrium(eq,distributed,context;synthesis_eos=synthesis_eos)
end

function prepare_equilibrium(eq::EquilibriumConfig,distributed::DistributedAtmosphere,context::ParallelContext;
        synthesis_eos=nothing)
    try
        _prepare_hsemhs(eq,distributed,context,synthesis_eos)
    catch e
        # HSEMHS agrees on failures collectively, so every rank throws the same error here
        e isa HSEMHS.InvalidInput ? throw(ArgumentError(e.msg)) : rethrow()
    end
end

function _prepare_hsemhs(eq,distributed,context,synthesis_eos)
    _assert_mpi_thread(context)
    grid=distributed.global_grid; comm=_backend_comm(context)
    expected=eq.mode in (:column_hse_z,:mhs_poisson_z) ? :z_m : :log10tau500
    local_state=HSEMHS.agree(comm,"prepare_equilibrium") do
        vertical_coordinate(grid)===expected || throw(ArgumentError(
            "equilibrium mode $(hsemhs_mode_name(eq.mode)) needs a native $expected grid, got $(vertical_coordinate(grid))"))
        context.enabled && context.size>1 && context.options.decomposition!==:x_slabs && throw(ArgumentError(
            "the HSEMHS equilibrium needs parallel.decomposition = \"x_slabs\" (got $(context.options.decomposition))"))
        isfile(eq.eos_table_file) || throw(ArgumentError("EOS table not found: $(eq.eos_table_file)"))
        table=HSEMHS.load_table(eq.eos_table_file)
        (table,_eos_provenance(table,synthesis_eos))
    end
    table,provenance=local_state
    column=eq.column
    ws=HSEMHS.prepare_embedded(comm,eq.mode;axis=native_depth(grid),x=grid.x,y=grid.y,eos=table,g=eq.gravity_m_s2,
        column=column,poisson=eq.poisson,progress=isroot(context))
    ok=HSEMHS.agree(comm,"equilibrium slab ownership") do
        HSEMHS.owned_columns(ws)==distributed.tile.xrange && distributed.tile.yrange==1:length(grid.y) ||
            throw(ArgumentError("HSEMHS owns x $(HSEMHS.owned_columns(ws)) but this rank's tile is x $(distributed.tile.xrange), y $(distributed.tile.yrange)"))
        true
    end
    provenance["mode"]=hsemhs_mode_name(eq.mode); provenance["gravity_m_s2"]=eq.gravity_m_s2
    provenance["eos_table_file"]=abspath(eq.eos_table_file); provenance["hsemhs_version"]=string(pkgversion(HSEMHS))
    HSEMHSBackend(eq,ws,provenance,false)
end

_zxy_to_zyx(a)=permutedims(a,(1,3,2))

function _set_field!(atmosphere,name,value)
    destination=getfield(atmosphere,name)
    destination===nothing ? setfield!(atmosphere,name,copy(value)) : copyto!(destination,value)
    nothing
end

function reconstruct_equilibrium!(backend::HSEMHSBackend,distributed::DistributedAtmosphere,context::ParallelContext)
    backend.closed && throw(ArgumentError("the equilibrium backend is closed"))
    _assert_mpi_thread(context)
    a=distributed.local_atmosphere
    lnT=_zxy_to_zyx(log.(a.temperature))
    B=a.magnetic_field
    started=time()
    _forward_stage(context,"equilibrium begin mode=$(hsemhs_mode_name(backend.config.mode))")
    res=try
        if B===nothing
            # no magnetic data means B = 0 (the τ-grid alignment then uses the gas pressure alone)
            backend.config.mode===:column_hse_z ? HSEMHS.solve_embedded!(backend.workspace,lnT) :
                HSEMHS.solve_embedded!(backend.workspace,lnT,zero(lnT),zero(lnT),zero(lnT))
        else
            HSEMHS.solve_embedded!(backend.workspace,lnT,_zxy_to_zyx(B.Bx),_zxy_to_zyx(B.By),_zxy_to_zyx(B.Bz))
        end
    catch e
        e isa HSEMHS.InvalidInput ? throw(ArgumentError(e.msg)) : rethrow()
    end
    st=res.status
    _forward_stage(context,"equilibrium done seconds=$(round(time()-started;digits=1)) valid=$(st.valid) reason=$(st.reason)")
    if st.valid
        _set_field!(a,:pgas,_zxy_to_zyx(res.Pgas)); _set_field!(a,:rho,_zxy_to_zyx(res.rho))
        _set_field!(a,:ne,_zxy_to_zyx(res.ne)); _set_field!(a,:z,_zxy_to_zyx(res.z))
        _set_field!(a,:logtau500,_zxy_to_zyx(res.logtau500)); _set_field!(a,:chi500,_zxy_to_zyx(res.chi500))
        check_coordinate_consistency(a)
    end
    EquilibriumOutcome(st.valid,st.reason,backend.config.model,backend.config.mode,st.steps,st.seconds,copy(st.messages),
        deepcopy(st))
end

function close_equilibrium!(backend::HSEMHSBackend)
    backend.closed && return nothing
    HSEMHS.free_embedded!(backend.workspace); backend.closed=true
    nothing
end
