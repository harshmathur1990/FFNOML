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
coordinate and `model` (brief §2): column_hse_z, column_hse_ltau, column_hse_logm,
mhs_poisson_z, mhs_poisson_ltau, mhs_poisson_logm. The column options use HSEMHS's names, defaults and
semantics; HSEMHS is the authority on the pressure boundary."""
struct EquilibriumConfig
    backend::String
    model::Symbol
    mode::Symbol
    gravity_m_s2::Float64
    eos_table_file::String
    on_nonconvergence::Symbol
    column::Any                       # HSEMHS.ColumnZOptions or HSEMHS.ColumnTauOptions (= ColumnMassOptions)
    poisson::Any                      # nothing in Phase 3 (mhs arrives in Phase 4)
end

const _EQ_KEYS = ("backend","model","gravity_m_s2","eos_table_file","on_nonconvergence","column_z","column_ltau","column_logm")
const _COLUMN_Z_KEYS = Dict("anchor"=>:anchor,"P_ref_pa"=>:P_ref,"z_ref_m"=>:z_ref,"n_sub"=>:n_sub,"adapt"=>:adapt,
    "rk_tol"=>:rk_tol,"n_sub_max"=>:n_sub_max,"tau_top"=>:tau_top,"H_max_m"=>:H_max,"mu"=>:mu)
const _COLUMN_LTAU_KEYS = Dict("align"=>:align)
const _COLUMN_LOGM_KEYS = Dict("align"=>:align)

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
    model===:mhs && throw(ArgumentError("equilibrium.model = \"mhs\" (mhs-poisson-z/-ltau/-logm) is not available yet (HSEMHS integration Phase 4)"))
    height=vertical_coordinate===:z_m; mass=vertical_coordinate===:log10m
    suffix=height ? "z" : mass ? "logm" : "ltau"
    mode=Symbol(model===:column_hse ? "column_hse_" : "mhs_poisson_",suffix)
    g=Float64(get(raw,"gravity_m_s2",274.0)); isfinite(g)&&g>0 || throw(ArgumentError("equilibrium.gravity_m_s2 must be positive"))
    table=String(_required(raw,"eos_table_file")); table=isabspath(table) ? table : normpath(joinpath(base,table))
    policy=Symbol(lowercase(String(get(raw,"on_nonconvergence","reject_trial"))))
    policy in (:reject_trial,:abort) || throw(ArgumentError("equilibrium.on_nonconvergence must be reject_trial or abort"))
    column=if height
        haskey(raw,"column_ltau") && throw(ArgumentError("[equilibrium.column_ltau] does not apply to a native-z run ($mode)"))
        haskey(raw,"column_logm") && throw(ArgumentError("[equilibrium.column_logm] does not apply to a native-z run ($mode)"))
        cz=get(raw,"column_z",Dict{String,Any}())
        haskey(cz,"P_ref_pa") || throw(ArgumentError(
            "[equilibrium.column_z] P_ref_pa is required for $mode (HSEMHS column_hse.P_ref; the pressure boundary is set by HSEMHS)"))
        HSEMHS.ColumnZOptions(; _options(cz,_COLUMN_Z_KEYS,"equilibrium.column_z")...)
    elseif mass
        haskey(raw,"column_z") && throw(ArgumentError(
            "[equilibrium.column_z] does not apply to a native log-m run ($mode): P = g m sets the pressure"))
        haskey(raw,"column_ltau") && throw(ArgumentError("[equilibrium.column_ltau] does not apply to a native log-m run ($mode)"))
        HSEMHS.ColumnMassOptions(; _options(get(raw,"column_logm",Dict{String,Any}()),_COLUMN_LOGM_KEYS,"equilibrium.column_logm")...)
    else
        haskey(raw,"column_logm") && throw(ArgumentError("[equilibrium.column_logm] does not apply to a native log-tau run ($mode)"))
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
# a given atmosphere: nₑ and hydrogen populations, no HSEMHS
# ---------------------------------------------------------------------------------------------------------------------

# Wittmann's abundances (ABUND_RAW and AMASS in scripts/witt_eos_cpp.cpp), so that a given atmosphere has the composition of the synthesis EOS
const _GIVEN_TOTAL_ABUNDANCE = 1.0950187898409673
const _GIVEN_MASS_PER_H_KG = 2.3396723599405787e-27
const _KB_SI = 1.380649e-23

"""χ₅₀₀ [m⁻¹] from T, nₑ, neutral H and protons (nz, nx, ny); the method is in the Muspel extension."""
function given_chi500 end

"""
Fill `pgas`, `rho` and `chi500` of an atmosphere from its T, its nₑ and the hydrogen populations (nz, nx, ny, level; the last level the
protons), as RH's `MULTIatmos` and `convertScales`: n_H = Σ n, ρ = (mass per H) n_H, P = (n_H Σ abundances + nₑ) k T. Neutral hydrogen and
protons are kept for the continuum and broadening (`build_synthesis_setup`).
"""
function set_given_state!(a,hydrogen)
    T=eltype(a.temperature)
    size(hydrogen)[1:3]==size(a.temperature) && size(hydrogen,4)>=2 || throw(DimensionMismatch(
        "hydrogen populations have size $(size(hydrogen)); expected $(size(a.temperature)) and at least two levels, the last the protons"))
    all(isfinite,hydrogen) && all(>=(0),hydrogen) && all(>(0),sum(hydrogen;dims=4)) || throw(ArgumentError(
        "hydrogen populations must be finite, not negative, and sum to a positive density"))
    neutral=dropdims(sum(@view(hydrogen[:,:,:,1:end-1]);dims=4);dims=4); protons=hydrogen[:,:,:,end]; nh=neutral.+protons
    a.rho=nh.*T(_GIVEN_MASS_PER_H_KG)
    a.pgas=(nh.*T(_GIVEN_TOTAL_ABUNDANCE).+a.ne).*T(_KB_SI).*a.temperature
    a.chi500=T.(given_chi500(a.temperature,a.ne,neutral,protons))
    a.h_neutral=neutral; a.h_proton=protons
    a
end

"""
Fill `rho` and `pgas` of an atmosphere from its given ρ and nₑ: n_H = ρ/(mass per H), P = (n_H Σ abundances + nₑ) k T. χ₅₀₀ needs hydrogen,
which here comes from the EOS at (T, P), so `GivenAtmosphereBackend` adds it when the synthesis EOS is known. As hydrogen is not given,
`h_neutral` and `h_proton` stay empty, and the continuum and broadening take hydrogen from the EOS as in any other run.
"""
function set_given_density!(a,rho)
    T=eltype(a.temperature)
    a.rho=rho
    a.pgas=(rho./T(_GIVEN_MASS_PER_H_KG).*T(_GIVEN_TOTAL_ABUNDANCE).+a.ne).*T(_KB_SI).*a.temperature
    a
end

"""χ₅₀₀ of an atmosphere with a given ρ and nₑ: Muspel's continuum at 500 nm, with neutral H and protons from the EOS at (T, P)."""
function _given_density_chi500!(a,synthesis_eos)
    synthesis_eos===nothing && throw(ArgumentError("a given nₑ and ρ need the synthesis EOS: prepare_equilibrium(...; synthesis_eos)"))
    state=WittmannKuruczState(synthesis_eos); nz,nx,ny=size(a.temperature)
    neutral=zeros(Float64,nz,nx,ny); protons=zeros(Float64,nz,nx,ny)
    Threads.@threads for y in 1:ny
        for x in 1:nx
            temperature=Float64.(a.temperature[:,x,y]); pressure=Float64.(a.pgas[:,x,y])
            neutral[:,x,y].=eos_neutral_hydrogen!(zeros(Float64,nz),state,temperature,pressure)
            protons[:,x,y].=eos_protons(state,temperature,pressure)
        end
    end
    a.chi500=eltype(a.temperature).(given_chi500(a.temperature,a.ne,neutral,protons))
    a
end

"""
Geometry of one column from its ρ and χ₅₀₀, RH's `convertScales`: dz = −2 μ Δm/(ρₖ₋₁+ρₖ) with τ₁ = (χ₅₀₀/ρ) m on a mass axis;
dz = −2 μ Δτ/(χ₅₀₀ₖ₋₁+χ₅₀₀ₖ) on a τ axis; the axis itself on a height axis, with τ₁ = χ₅₀₀ |Δz|/(2μ), 0 if that exceeds 1 (written one dex
above the next level, its logarithm being undefined). Elsewhere dτ = (χ₅₀₀ₖ₋₁+χ₅₀₀ₖ)/2 · |dz|/μ. m and τ are along the line of sight, z vertical. On the
mass and τ axes z(τ = 1) = 0 in every column (linear in τ).
"""
function given_geometry!(z,logtau,kind::Symbol,axis,rho,chi,mu)
    n=length(axis); tau=zeros(Float64,n)
    if kind===:z_m
        z.=axis; tau[1]=chi[1]*(axis[1]-axis[2])/(2mu); tau[1]>1 && (tau[1]=0.0)
    elseif kind===:log10m
        m=exp10.(Float64.(axis)); z[1]=0; tau[1]=chi[1]/rho[1]*m[1]
        for k in 2:n; z[k]=z[k-1]-2mu*(m[k]-m[k-1])/(rho[k-1]+rho[k]); end
    else
        tau.=exp10.(Float64.(axis)); z[1]=0
        for k in 2:n; z[k]=z[k-1]-2mu*(tau[k]-tau[k-1])/(chi[k-1]+chi[k]); end
    end
    kind===:log10tau500 || for k in 2:n
        tau[k]=tau[k-1]+(chi[k-1]+chi[k])/2*(z[k-1]-z[k])/mu
    end
    if kind===:log10tau500
        logtau.=axis
    else
        logtau.=log10.(tau)
        tau[1]==0 && (logtau[1]=logtau[2]-1)
    end
    if kind!==:z_m
        j=searchsortedfirst(tau,1.0)
        zero_level=j==1 ? z[1] : j>n ? z[n] : z[j-1]+(z[j]-z[j-1])*(1-tau[j-1])/(tau[j]-tau[j-1])
        z.-=zero_level
    end
    nothing
end

"""`z` and `logtau500` of every column of `a` from its `rho` and `chi500`."""
function _given_columns!(a)
    T=eltype(a.temperature); a.z===nothing && (a.z=zeros(T,size(a.temperature))); a.logtau500===nothing && (a.logtau500=zeros(T,size(a.temperature)))
    kind=vertical_coordinate(a.grid); axis=native_depth(a.grid); mu=a.grid.view.mu; _,nx,ny=size(a.temperature)
    Threads.@threads for y in 1:ny
        for x in 1:nx
            @views given_geometry!(a.z[:,x,y],a.logtau500[:,x,y],kind,axis,a.rho[:,x,y],a.chi500[:,x,y],mu)
        end
    end
    check_coordinate_consistency(a)
end

"""
The atmosphere is given, by nₑ with either the hydrogen populations (`set_given_state!`) or ρ (`set_given_density!`, χ₅₀₀ here from the
EOS's hydrogen): `pgas`, `rho` and `chi500` are fixed, and every axis's geometry follows column by column. No HSEMHS, no EOS table and no MPI. Forward only.
"""
mutable struct GivenAtmosphereBackend <: AbstractEquilibriumBackend
    config::EquilibriumConfig
    provenance::Dict{String,Any}
    function GivenAtmosphereBackend(eq::EquilibriumConfig,distributed::DistributedAtmosphere;synthesis_eos=nothing)
        a=distributed.local_atmosphere
        hydrogen_given=a.h_neutral!==nothing
        hydrogen_given || a.chi500!==nothing || a.rho===nothing || _given_density_chi500!(a,synthesis_eos)
        all(n->getfield(a,n)!==nothing,(:ne,:rho,:pgas,:chi500)) || throw(ArgumentError(
            "a given atmosphere needs ne, rho, pgas and chi500 (read_inversion_inputs fills them from ne with the hydrogen populations or rho)"))
        new(eq,Dict{String,Any}("mode"=>hsemhs_mode_name(eq.mode),"ne_given"=>true,"hydrogen_populations_given"=>hydrogen_given,"rho_given"=>true,
            "total_abundance"=>_GIVEN_TOTAL_ABUNDANCE,"mass_per_H_kg"=>_GIVEN_MASS_PER_H_KG))
    end
end

function reconstruct_equilibrium!(backend::GivenAtmosphereBackend,distributed::DistributedAtmosphere,context::ParallelContext)
    started=time()
    _forward_stage(context,"equilibrium begin mode=$(hsemhs_mode_name(backend.config.mode))")
    _given_columns!(distributed.local_atmosphere)
    _forward_stage(context,"equilibrium done seconds=$(round(time()-started;digits=1)) valid=true reason=given")
    EquilibriumOutcome(true,:given,:given,backend.config.mode,0,time()-started,String[],nothing)
end

"""`[equilibrium]` of a given atmosphere (`ne` with `hydrogen_populations` or `rho` in the datasets): optional, and only `gravity_m_s2` and
`eos_table_file` have a use, both for the population mesh's extensions of the columns."""
function _given_equilibrium_config(raw,vertical_coordinate::Symbol,base::AbstractString)
    unknown=setdiff(collect(keys(raw)),["backend","model","gravity_m_s2","eos_table_file","on_nonconvergence"])
    isempty(unknown) || throw(ArgumentError(
        "[equilibrium] $(join(sort(unknown),", ")) do(es) not apply: a given atmosphere builds the geometry without HSEMHS"))
    g=Float64(get(raw,"gravity_m_s2",274.0)); isfinite(g)&&g>0 || throw(ArgumentError("equilibrium.gravity_m_s2 must be positive"))
    table=String(get(raw,"eos_table_file","")); table=isempty(table)||isabspath(table) ? table : normpath(joinpath(base,table))
    mode=vertical_coordinate===:z_m ? :given_z : vertical_coordinate===:log10m ? :given_logm : :given_ltau
    EquilibriumConfig("given",:given,mode,g,table,:abort,nothing,nothing)
end

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
    ne_given::Bool                    # the input supplies nₑ: HSEMHS derives P and rho from it on every reconstruction
    given_rho::Any                    # the input supplies only ρ (forward only): nₑ from the EOS at (T, ρ), then as a given nₑ; else nothing
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
    eq.backend=="given" && return GivenAtmosphereBackend(eq,distributed;synthesis_eos=synthesis_eos)
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
    expected=eq.mode in (:column_hse_z,:mhs_poisson_z) ? :z_m : eq.mode in (:column_hse_logm,:mhs_poisson_logm) ? :log10m : :log10tau500
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
        column=column,poisson=eq.poisson,view=hsemhs_view(grid.view),progress=isroot(context))
    ok=HSEMHS.agree(comm,"equilibrium slab ownership") do
        HSEMHS.owned_columns(ws)==distributed.tile.xrange && distributed.tile.yrange==1:length(grid.y) ||
            throw(ArgumentError("HSEMHS owns x $(HSEMHS.owned_columns(ws)) but this rank's tile is x $(distributed.tile.xrange), y $(distributed.tile.yrange)"))
        true
    end
    provenance["mode"]=hsemhs_mode_name(eq.mode); provenance["gravity_m_s2"]=eq.gravity_m_s2
    is_vertical(grid.view) || merge!(provenance,view_attributes(grid.view,view_columns(grid)))
    provenance["eos_table_file"]=abspath(eq.eos_table_file); provenance["hsemhs_version"]=string(pkgversion(HSEMHS))
    a=distributed.local_atmosphere
    given_rho=a.ne===nothing && a.rho!==nothing ? copy(a.rho) : nothing
    ne_given=a.ne!==nothing || given_rho!==nothing
    a.ne!==nothing && (provenance["ne_given"]=true)
    given_rho===nothing || (provenance["rho_given"]=true)
    HSEMHSBackend(eq,ws,provenance,false,ne_given,given_rho)
end

_zxy_to_zyx(a)=permutedims(a,(1,3,2))

function _set_field!(atmosphere,name,value)
    destination=getfield(atmosphere,name)
    destination===nothing ? setfield!(atmosphere,name,copy(value)) : copyto!(destination,value)
    nothing
end

"""
nₑ of an atmosphere with only ρ given: the EOS state at (T, ρ) on every level. Fills `a.ne`, which the given-nₑ route then takes, and returns
(valid, reason, messages), identical on every rank. That route's P reproduces ρ through the same table.
"""
function _ne_from_given_rho!(backend,a,context)
    eos=backend.workspace.eos; T=eltype(a.temperature); nz,nx,ny=size(a.temperature); ne=fill(T(NaN),nz,nx,ny)
    Threads.@threads for y in 1:ny
        for x in 1:nx, k in 1:nz
            lnT=log(Float64(a.temperature[k,x,y])); lnrho=log(Float64(backend.given_rho[k,x,y]))
            lnP=try _mesh_lnP(eos,lnT,lnrho) catch e; e isa ErrorException || rethrow(); NaN end
            isfinite(lnP) || continue
            r,n,_,_=HSEMHS.state(eos,lnT,lnP)
            isfinite(r) && abs(r-lnrho)<1e-8 && (ne[k,x,y]=T(exp(n)))
        end
    end
    nbad=allreduce_sum(count(!isfinite,ne),context)
    nbad>0 && return false,:eos_out_of_envelope,["$nbad levels have a given density no EOS state reaches at their temperature"]
    _set_field!(a,:ne,ne)
    true,:converged,String[]
end

"""
Column-mass axis with a given nₑ and no hydrogen populations: P is the pressure at which the EOS gives nₑ at T and ρ, χ₅₀₀ the EOS's there
(as HSEMHS does for a given nₑ on the other axes); `given_geometry!` builds the rest. Fills `a` and returns (valid, reason, messages),
identical on every rank.
"""
function _given_ne_column_mass!(backend,a,context)
    eos=backend.workspace.eos; T=eltype(a.temperature); nz,nx,ny=size(a.temperature)
    P=zeros(T,nz,nx,ny); rho=zeros(T,nz,nx,ny); chi=zeros(T,nz,nx,ny); z=zeros(T,nz,nx,ny); lt=zeros(T,nz,nx,ny)
    Threads.@threads for y in 1:ny
        for x in 1:nx, k in 1:nz
            lnT=log(Float64(a.temperature[k,x,y]))
            lnP=HSEMHS.lnP_from_lnne(eos,lnT,log(Float64(a.ne[k,x,y])))
            P[k,x,y]=exp(lnP); isfinite(lnP) || continue
            r,_,c,_=HSEMHS.state(eos,lnT,lnP)
            rho[k,x,y]=exp(r); chi[k,x,y]=exp(c)
        end
    end
    nbad=allreduce_sum(count(!isfinite,P),context)
    nbad>0 && return false,:eos_out_of_envelope,["$nbad levels have a given electron density no EOS state reaches"]
    _set_field!(a,:pgas,P); _set_field!(a,:rho,rho); _set_field!(a,:chi500,chi)
    _given_columns!(a)
    true,:converged,String[]
end

function reconstruct_equilibrium!(backend::HSEMHSBackend,distributed::DistributedAtmosphere,context::ParallelContext)
    backend.closed && throw(ArgumentError("the equilibrium backend is closed"))
    _assert_mpi_thread(context)
    a=distributed.local_atmosphere
    if backend.given_rho!==nothing
        started=time()
        valid,reason,messages=_ne_from_given_rho!(backend,a,context)
        valid || return EquilibriumOutcome(false,reason,backend.config.model,backend.config.mode,0,time()-started,messages,nothing)
    end
    if backend.ne_given && backend.config.mode===:column_hse_logm
        started=time()
        _forward_stage(context,"equilibrium begin mode=$(hsemhs_mode_name(backend.config.mode)) ne_given=true")
        valid,reason,messages=_given_ne_column_mass!(backend,a,context)
        _forward_stage(context,"equilibrium done seconds=$(round(time()-started;digits=1)) valid=$valid reason=$reason")
        valid && backend.given_rho!==nothing && _set_field!(a,:rho,backend.given_rho)
        return EquilibriumOutcome(valid,reason,backend.config.model,backend.config.mode,0,time()-started,messages,nothing)
    end
    lnT=_zxy_to_zyx(log.(a.temperature))
    B=a.magnetic_field
    started=time()
    _forward_stage(context,"equilibrium begin mode=$(hsemhs_mode_name(backend.config.mode))")
    given=backend.ne_given ? (;lnne=_zxy_to_zyx(log.(a.ne))) : (;)
    res=try
        if B===nothing
            # no magnetic data means B = 0 (the τ- and m-grid alignment then uses the gas pressure alone)
            backend.config.mode===:column_hse_z ? HSEMHS.solve_embedded!(backend.workspace,lnT;given...) :
                HSEMHS.solve_embedded!(backend.workspace,lnT,zero(lnT),zero(lnT),zero(lnT);given...)
        else
            HSEMHS.solve_embedded!(backend.workspace,lnT,_zxy_to_zyx(B.Bx),_zxy_to_zyx(B.By),_zxy_to_zyx(B.Bz);given...)
        end
    catch e
        e isa HSEMHS.InvalidInput ? throw(ArgumentError(e.msg)) : rethrow()
    end
    st=res.status
    _forward_stage(context,"equilibrium done seconds=$(round(time()-started;digits=1)) valid=$(st.valid) reason=$(st.reason)")
    if st.valid
        _set_field!(a,:pgas,_zxy_to_zyx(res.Pgas)); _set_field!(a,:rho,backend.given_rho===nothing ? _zxy_to_zyx(res.rho) : backend.given_rho)
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
