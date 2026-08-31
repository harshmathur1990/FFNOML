"""Rank-local cotangents of the atmosphere after force reconstruction.

Every field is allocated so pullbacks can accumulate without optional-value
branches. Magnetic cotangents exist only when the primal atmosphere has B.
"""
mutable struct AtmosphereCotangent{T<:AbstractFloat,A<:AbstractArray{T,3},B}
    temperature::A
    vx::A
    vy::A
    vz::A
    vturb::A
    pgas::A
    rho::A
    ne::A
    z::A
    magnetic_field::B
end

function AtmosphereCotangent(atmosphere::Atmosphere3D{T}) where T
    shape=size(atmosphere.temperature); field()=zeros(T,shape)
    magnetic=atmosphere.magnetic_field===nothing ? nothing :
        MagneticField3D(field(),field(),field())
    AtmosphereCotangent(field(),field(),field(),field(),field(),field(),field(),field(),field(),magnetic)
end

function _accumulate!(destination::AtmosphereCotangent,source::AtmosphereCotangent)
    for name in (:temperature,:vx,:vy,:vz,:vturb,:pgas,:rho,:ne,:z)
        getfield(destination,name).+=getfield(source,name)
    end
    if destination.magnetic_field!==nothing
        source.magnetic_field===nothing && throw(DimensionMismatch("magnetic cotangent is missing"))
        for name in (:Bx,:By,:Bz)
            getfield(destination.magnetic_field,name).+=getfield(source.magnetic_field,name)
        end
    end
    destination
end

# ---------------------------------------------------------------------------
# Observation transpose
# ---------------------------------------------------------------------------

function _convolve_axis_vjp(output_bar::AbstractArray{T,4},kernel,axis::Int) where T
    length(kernel)==1 && return copy(output_bar)
    input_bar=zeros(T,size(output_bar)); radius=(length(kernel)-1)÷2
    for index in CartesianIndices(output_bar)
        coord=Tuple(index); norm=zero(T)
        for offset in -radius:radius
            1<=coord[axis]+offset<=size(output_bar,axis) || continue
            norm+=kernel[offset+radius+1]
        end
        for offset in -radius:radius
            pos=coord[axis]+offset
            1<=pos<=size(output_bar,axis) || continue
            source=CartesianIndex(ntuple(d->d==axis ? pos : coord[d],4))
            input_bar[source]+=output_bar[index]*kernel[offset+radius+1]/norm
        end
    end
    input_bar
end

function _distributed_spatial_vjp(output_bar::AbstractArray{T,4},kernel,axis::Int,
        distributed::DistributedAtmosphere,context::ParallelContext;tag::Int) where T
    radius=(length(kernel)-1)÷2; radius==0 && return copy(output_bar)
    lx,ly=size(output_bar,3),size(output_bar,4)
    radius<=min(lx,ly) || throw(ArgumentError("PSF radius exceeds local tile extent"))
    padded=zeros(T,size(output_bar,1),size(output_bar,2),lx+2radius,ly+2radius)
    for l in axes(output_bar,1),s in axes(output_bar,2),i in 1:lx,j in 1:ly
        for offset in -radius:radius
            ii=axis==3 ? i+radius+offset : i+radius
            jj=axis==4 ? j+radius+offset : j+radius
            padded[l,s,ii,jj]+=kernel[offset+radius+1]*output_bar[l,s,i,j]
        end
    end
    local_bar=copy(@view padded[:,:,radius+1:radius+lx,radius+1:radius+ly])
    tile=distributed.tile; cx,cy=tile.coordinates
    if axis==3
        left=_rank_at(tile,cx-1,cy); right=_rank_at(tile,cx+1,cy)
        send_left=copy(@view padded[:,:,1:radius,radius+1:radius+ly])
        send_right=copy(@view padded[:,:,radius+lx+1:radius+lx+radius,radius+1:radius+ly])
        if context.enabled
            _assert_mpi_thread(context)
            from_right=zeros(T,size(send_left)); from_left=zeros(T,size(send_right))
            MPI.Sendrecv!(send_left,from_right,context.comm;dest=something(left,MPI.PROC_NULL),
                sendtag=tag,source=something(right,MPI.PROC_NULL),recvtag=tag)
            MPI.Sendrecv!(send_right,from_left,context.comm;dest=something(right,MPI.PROC_NULL),
                sendtag=tag+1,source=something(left,MPI.PROC_NULL),recvtag=tag+1)
            right===nothing || (@views local_bar[:,:,lx-radius+1:lx,:].+=from_right)
            left===nothing || (@views local_bar[:,:,1:radius,:].+=from_left)
        end
        if left===nothing; @views local_bar[:,:,1:1,:].+=sum(send_left,dims=3); end
        if right===nothing; @views local_bar[:,:,lx:lx,:].+=sum(send_right,dims=3); end
    elseif axis==4
        down=_rank_at(tile,cx,cy-1); up=_rank_at(tile,cx,cy+1)
        send_down=copy(@view padded[:,:,radius+1:radius+lx,1:radius])
        send_up=copy(@view padded[:,:,radius+1:radius+lx,radius+ly+1:radius+ly+radius])
        if context.enabled
            _assert_mpi_thread(context)
            from_up=zeros(T,size(send_down)); from_down=zeros(T,size(send_up))
            MPI.Sendrecv!(send_down,from_up,context.comm;dest=something(down,MPI.PROC_NULL),
                sendtag=tag,source=something(up,MPI.PROC_NULL),recvtag=tag)
            MPI.Sendrecv!(send_up,from_down,context.comm;dest=something(up,MPI.PROC_NULL),
                sendtag=tag+1,source=something(down,MPI.PROC_NULL),recvtag=tag+1)
            up===nothing || (@views local_bar[:,:,:,ly-radius+1:ly].+=from_up)
            down===nothing || (@views local_bar[:,:,:,1:radius].+=from_down)
        end
        if down===nothing; @views local_bar[:,:,:,1:1].+=sum(send_down,dims=4); end
        if up===nothing; @views local_bar[:,:,:,ly:ly].+=sum(send_up,dims=4); end
    else
        throw(ArgumentError("distributed spatial VJP axis must be 3 or 4"))
    end
    local_bar
end

function observation_vjp!(intrinsic_bar,output_bar,::IdentityObservation,
        intrinsic::SpectralCube,distributed,context)
    size(intrinsic_bar)==size(output_bar)==size(intrinsic.data) || throw(DimensionMismatch(
        "observation VJP arrays differ"))
    copyto!(intrinsic_bar,output_bar)
end

function observation_vjp!(intrinsic_bar,output_bar,model::GaussianPSFObservation,
        intrinsic::SpectralCube{T},distributed,context) where T
    size(intrinsic_bar)==size(output_bar)==size(intrinsic.data) || throw(DimensionMismatch(
        "observation VJP arrays differ"))
    ky=_kernel(T,T(model.spatial_fwhm_y_m),T(model.dy_m))
    kx=_kernel(T,T(model.spatial_fwhm_x_m),T(model.dx_m))
    spectral_kernel=model.spectral_fwhm_m==0 ? T[one(T)] :
        _kernel(T,T(model.spectral_fwhm_m),_uniform_spacing(intrinsic.wavelength_m))
    ybar=_distributed_spatial_vjp(output_bar,ky,4,distributed,context;tag=710)
    xbar=_distributed_spatial_vjp(ybar,kx,3,distributed,context;tag=712)
    copyto!(intrinsic_bar,_convolve_axis_vjp(xbar,spectral_kernel,1))
end

# ---------------------------------------------------------------------------
# Scalar formal-solver and opacity pullbacks
# ---------------------------------------------------------------------------

@inline function _formal_weights_derivative(dt)
    if dt>40
        w1=zero(dt); w2=inv(dt); w3=one(dt)-w2
        return w1,w2,w3,zero(dt),-inv(dt^2),inv(dt^2)
    elseif dt>0.01
        w1=exp(-dt); u0=(one(dt)-w1)/dt; w2=u0-w1; w3=one(dt)-u0
        dw1=-w1; du0=((dt+one(dt))*w1-one(dt))/dt^2
        return w1,w2,w3,dw1,du0-dw1,-du0
    end
    w1=one(dt)-dt+dt^2/2
    w2=(one(dt)/2-dt/3)*dt
    w3=(one(dt)/2-dt/6)*dt
    w1,w2,w3,-one(dt)+dt,one(dt)/2-2dt/3,one(dt)/2-dt/3
end

function _formal_solve_vjp!(chi_bar,eta_bar,z_bar,::ScalarFormalSolver,chi,eta,z,output_bar)
    nz,nlambda=size(chi); length(output_bar)==nlambda || throw(DimensionMismatch(
        "formal-solver cotangent wavelength count differs"))
    fill!(chi_bar,zero(eltype(chi_bar))); fill!(eta_bar,zero(eltype(eta_bar)))
    for l in 1:nlambda
        source=(@view eta[:,l])./(@view chi[:,l])
        intensity_before=zeros(eltype(chi),nz)
        intensity=source[nz]; intensity_before[nz]=intensity
        for k in nz-1:-1:1
            dz=z[k+1]-z[k]; average_chi=(chi[k,l]+chi[k+1,l])/2
            dt=abs(dz)*average_chi
            w1,w2,w3,_,_,_=_formal_weights_derivative(dt)
            intensity_before[k]=intensity
            intensity=w1*intensity+w2*source[k+1]+w3*source[k]
        end
        source_bar=zeros(eltype(chi),nz); intensity_bar=output_bar[l]
        for k in 1:nz-1
            dz=z[k+1]-z[k]; average_chi=(chi[k,l]+chi[k+1,l])/2
            dt=abs(dz)*average_chi
            w1,w2,w3,dw1,dw2,dw3=_formal_weights_derivative(dt)
            dt_bar=intensity_bar*(dw1*intensity_before[k]+dw2*source[k+1]+dw3*source[k])
            source_bar[k+1]+=intensity_bar*w2; source_bar[k]+=intensity_bar*w3
            intensity_bar*=w1
            chi_bar[k,l]+=dt_bar*abs(dz)/2; chi_bar[k+1,l]+=dt_bar*abs(dz)/2
            dz_bar=dt_bar*sign(dz)*average_chi
            z_bar[k+1]+=dz_bar; z_bar[k]-=dz_bar
        end
        source_bar[nz]+=intensity_bar
        for k in 1:nz
            eta_bar[k,l]+=source_bar[k]/chi[k,l]
            chi_bar[k,l]-=source_bar[k]*eta[k,l]/chi[k,l]^2
        end
    end
    chi_bar,eta_bar,z_bar
end

function add_opacity_emissivity_vjp! end

function add_opacity_emissivity_vjp!(atmosphere_bar,population_bar,population_lookup,
        chi_bar,eta_bar,::TabulatedOpacityModel,wavelength,atmosphere,x,y,populations)
    nothing
end

@inline function _planck_temperature_derivative(lambda,temperature)
    value=planck_lambda(lambda,temperature)
    argument=_H*_C/(lambda*_KB*temperature)
    value*argument/temperature/(one(argument)-exp(-argument))
end

function add_opacity_emissivity_vjp!(atmosphere_bar,population_bar,population_lookup,
        chi_bar,eta_bar,model::FFNOTransition,wavelength,atmosphere,x,y,populations)
    populations===nothing && throw(ArgumentError("FFNO transition VJP requires populations"))
    for k in axes(chi_bar,1),l in axes(chi_bar,2)
        temp=atmosphere.temperature[k,x,y]; vz=atmosphere.vz[k,x,y]
        vturb=atmosphere.vturb[k,x,y]; lambda=wavelength[l]
        thermal=2*_KB*temp/model.atomic_mass_kg+vturb^2
        width=model.wavelength0_m/_C*sqrt(thermal)
        shifted=lambda*(1-vz/_C); q=(shifted-model.wavelength0_m)/width
        profile=exp(-q^2)/(sqrt(pi)*width)
        nl=populations[k,x,y,model.lower_level]; nu=populations[k,x,y,model.upper_level]
        difference=nl-nu; difference<=0 && continue
        coefficient=model.oscillator_strength*1e-24
        extinction=coefficient*difference*profile
        source=planck_lambda(lambda,temp)
        extinction_bar=chi_bar[k,l]+eta_bar[k,l]*source
        source_bar=eta_bar[k,l]*extinction
        population_bar[k,x,y,model.lower_level]+=extinction_bar*coefficient*profile
        population_bar[k,x,y,model.upper_level]-=extinction_bar*coefficient*profile
        profile_bar=extinction_bar*coefficient*difference
        width_bar=profile_bar*profile*(2q^2-one(q))/width
        shifted_bar=profile_bar*profile*(-2q)/width
        atmosphere_bar.temperature[k,x,y]+=source_bar*_planck_temperature_derivative(lambda,temp)+
            width_bar*width*(_KB/model.atomic_mass_kg)/thermal
        atmosphere_bar.vturb[k,x,y]+=width_bar*width*vturb/thermal
        atmosphere_bar.vz[k,x,y]-=shifted_bar*lambda/_C
    end
    nothing
end

function add_opacity_emissivity_vjp!(atmosphere_bar,population_bar,population_lookup,
        chi_bar,eta_bar,model::KuruczLTEModel,wavelength,atmosphere,x,y,populations)
    model.eos===nothing || return _opaque_opacity_vjp!(atmosphere_bar,chi_bar,eta_bar,model,
        wavelength,atmosphere,x,y,populations)
    for line in model.lines,k in axes(chi_bar,1),l in axes(chi_bar,2)
        temp=atmosphere.temperature[k,x,y]; rho=atmosphere.rho[k,x,y]
        vz=atmosphere.vz[k,x,y]; vturb=atmosphere.vturb[k,x,y]; lambda=wavelength[l]
        boltzmann=exp(-line.lower_energy_j/(_KB*temp)); lower=rho*boltzmann
        thermal=2*_KB*temp/(56*1.66053906660e-27)+vturb^2
        width=line.wavelength0_m/_C*sqrt(thermal)
        shifted=lambda*(1-vz/_C); q=(shifted-line.wavelength0_m)/width
        profile=exp(-q^2)/(sqrt(pi)*width); coefficient=1e-2*10^line.loggf
        extinction=coefficient*lower*profile; source=planck_lambda(lambda,temp)
        extinction_bar=chi_bar[k,l]+eta_bar[k,l]*source
        source_bar=eta_bar[k,l]*extinction; lower_bar=extinction_bar*coefficient*profile
        profile_bar=extinction_bar*coefficient*lower
        width_bar=profile_bar*profile*(2q^2-one(q))/width
        shifted_bar=profile_bar*profile*(-2q)/width
        atmosphere_bar.rho[k,x,y]+=lower_bar*boltzmann
        atmosphere_bar.temperature[k,x,y]+=lower_bar*lower*line.lower_energy_j/(_KB*temp^2)+
            width_bar*width*(_KB/(56*1.66053906660e-27))/thermal+
            source_bar*_planck_temperature_derivative(lambda,temp)
        atmosphere_bar.vturb[k,x,y]+=width_bar*width*vturb/thermal
        atmosphere_bar.vz[k,x,y]-=shifted_bar*lambda/_C
    end
    nothing
end

"""Numerical local pullback for opaque C-backed opacity kernels.

This perturbs one column scalar at a time but never reruns force balance,
population inference, the formal solution, or the distributed forward model.
It is the correctness fallback for the Wittmann Kurucz adapter.
"""
function _opaque_opacity_vjp!(atmosphere_bar,chi_bar,eta_bar,model,wavelength,
        atmosphere,x,y,populations;relative_step=1e-5)
    nz,nlambda=size(chi_bar); T=eltype(chi_bar)
    function pairing()
        chi=zeros(T,nz,nlambda); eta=zeros(T,nz,nlambda)
        add_opacity_emissivity!(chi,eta,model,wavelength,atmosphere,x,y,populations)
        sum(chi.*chi_bar)+sum(eta.*eta_bar)
    end
    for (name,bar) in ((:temperature,atmosphere_bar.temperature),(:vz,atmosphere_bar.vz),
            (:vturb,atmosphere_bar.vturb),(:pgas,atmosphere_bar.pgas),
            (:rho,atmosphere_bar.rho),(:ne,atmosphere_bar.ne))
        values=getfield(atmosphere,name); values===nothing && continue
        for k in 1:nz
            center=values[k,x,y]; h=T(relative_step)*max(abs(center),one(T))
            values[k,x,y]=center+h; plus=pairing()
            values[k,x,y]=center-h; minus=pairing()
            values[k,x,y]=center; bar[k,x,y]+=(plus-minus)/(2h)
        end
    end
    nothing
end

_population_bar(populations::AbstractArray)=zeros(eltype(populations),size(populations))
_population_bar(populations::AbstractDict)=Dict(key=>zeros(eltype(value),size(value))
    for (key,value) in populations)

function _population_lookup(populations::AbstractArray,population_bar,source)
    source===populations || throw(ArgumentError(
        "synthesis contributor populations are not owned by the active population workspace"))
    population_bar
end


function _population_lookup(populations::AbstractDict,population_bar::AbstractDict,source)
    for (key,value) in populations
        value===source && return population_bar[key]
    end
    throw(ArgumentError(
        "synthesis contributor populations are not owned by the active population workspace"))
end

function synthesis_vjp!(atmosphere_bar::AtmosphereCotangent,model::MockIntensitySynthesizer,
        redistribution,atmosphere,populations,intrinsic,intrinsic_bar,cache=nothing)
    population_bar=_population_bar(populations)
    coupling=redistribution isa MockPRD ? redistribution.coupling : one(eltype(populations))
    for x in axes(intrinsic_bar,3),y in axes(intrinsic_bar,4)
        column_bar=zero(eltype(populations))
        for l in axes(intrinsic_bar,1)
            profile=exp(-((intrinsic.wavelength_m[l]-model.line_center_m)/model.width_m)^2)
            column_bar+=coupling*profile*intrinsic_bar[l,1,x,y]
        end
        @views population_bar[:,x,y,1].+=column_bar
    end
    population_bar
end

function synthesis_vjp!(atmosphere_bar::AtmosphereCotangent,model::MixedIntensitySynthesizer,
        redistribution,atmosphere,populations,intrinsic,intrinsic_bar,cache=nothing)
    redistribution isa NonPRD || throw(ArgumentError("mixed synthesis VJP supports non-PRD only"))
    model.solver isa ScalarFormalSolver || throw(ArgumentError(
        "production synthesis VJP requires ScalarFormalSolver"))
    population_bar=_population_bar(populations)
    lookup(source)=_population_lookup(populations,population_bar,source)
    nz,nx,ny=size(atmosphere.temperature); nlambda=length(intrinsic.wavelength_m)
    for x in 1:nx,y in 1:ny
        chi=zeros(eltype(intrinsic.data),nz,nlambda); eta=zeros(eltype(intrinsic.data),nz,nlambda)
        for contributor in model.contributors
            source=contributor.populations===nothing ? populations : contributor.populations
            add_opacity_emissivity!(chi,eta,contributor.model,intrinsic.wavelength_m,
                atmosphere,x,y,source)
        end
        chi_bar=zeros(eltype(chi),size(chi)); eta_bar=similar(chi_bar); z_bar=zeros(eltype(chi),nz)
        _formal_solve_vjp!(chi_bar,eta_bar,z_bar,model.solver,chi,eta,
            @view(atmosphere.z[:,x,y]),@view(intrinsic_bar[:,1,x,y]))
        @views atmosphere_bar.z[:,x,y].+=z_bar
        for contributor in model.contributors
            source=contributor.populations===nothing ? populations : contributor.populations
            source_bar=contributor.populations===nothing && populations isa AbstractDict ? nothing : lookup(source)
            add_opacity_emissivity_vjp!(atmosphere_bar,source_bar,lookup,chi_bar,eta_bar,
                contributor.model,intrinsic.wavelength_m,atmosphere,x,y,source)
        end
    end
    population_bar
end

# ---------------------------------------------------------------------------
# Population VJPs, including the rank-0 GPU service
# ---------------------------------------------------------------------------

function population_vjp!(feature_bar,z_bar,model::MockPopulationModel,atmosphere,population_bar)
    fill!(feature_bar,zero(eltype(feature_bar))); fill!(z_bar,zero(eltype(z_bar)))
    @views feature_bar[1,:,:,:].=model.scale/5000 .* population_bar[:,:,:,1]
    feature_bar,z_bar
end

function _features_to_atmosphere_vjp!(bar,feature_bar,z_bar,atmosphere)
    @views bar.temperature.+=feature_bar[1,:,:,:]
    @views bar.vx.+=feature_bar[2,:,:,:]
    @views bar.vy.+=feature_bar[3,:,:,:]
    @views bar.vz.+=feature_bar[4,:,:,:]
    @views bar.ne.+=feature_bar[5,:,:,:]./(atmosphere.ne.*log(10))
    @views bar.rho.+=feature_bar[6,:,:,:]./(atmosphere.rho.*log(10))
    bar.z.+=z_bar; bar
end

function predict_distributed_populations_vjp!(bar::AtmosphereCotangent,
        backend::LocalDistributedPopulationModel,distributed,population_bar,context)
    atmosphere=distributed.local_atmosphere
    feature_bar=zeros(eltype(population_bar),6,size(atmosphere.temperature)...)
    z_bar=zeros(eltype(population_bar),size(atmosphere.temperature))
    population_vjp!(feature_bar,z_bar,backend.model,atmosphere,population_bar)
    _features_to_atmosphere_vjp!(bar,feature_bar,z_bar,atmosphere)
end

function predict_distributed_populations_vjp!(bar::AtmosphereCotangent,
        backend::RootDistributedPopulationModel,distributed,population_bar,context)
    atmosphere=distributed.local_atmosphere; grid=distributed.global_grid; tile=distributed.tile
    nz=length(grid.log_tau500); nx=length(grid.x); ny=length(grid.y)
    packed=permutedims(population_bar,(1,4,2,3))
    global_packed=gather_field(_field(packed,tile,(nz,size(packed,2),nx,ny)),context;tag=720)
    global_atmosphere=gather_atmosphere(distributed,context)
    coordinator=RootGPUCoordinator(() -> begin
        backend.root_model===nothing && error("rank 0 has no FFNO population model")
        global_bar=permutedims(global_packed,(1,3,4,2))
        feature_bar=zeros(eltype(global_bar),6,nz,nx,ny); z_bar=zeros(eltype(global_bar),nz,nx,ny)
        population_vjp!(feature_bar,z_bar,backend.root_model,global_atmosphere,global_bar)
        (feature_bar=feature_bar,z_bar=z_bar)
    end;launcher_rank=context.root)
    result=launch_gpu!(coordinator,context)
    local_features=distribute_field(eltype(population_bar),isroot(context) ? result.feature_bar : nothing,
        (6,nz,nx,ny),context;tag=721).values
    local_z=distribute_field(eltype(population_bar),isroot(context) ? result.z_bar : nothing,
        (nz,nx,ny),context;tag=722).values
    _features_to_atmosphere_vjp!(bar,local_features,local_z,atmosphere)
end

function predict_distributed_populations_vjp!(bar::AtmosphereCotangent,
        backend::CompositeDistributedPopulationModel,distributed,population_bar::AbstractDict,context)
    Set(keys(population_bar))==Set(keys(backend.models)) || throw(ArgumentError(
        "population cotangent species differ from composite model"))
    for species in sort!(collect(keys(backend.models));by=string)
        predict_distributed_populations_vjp!(bar,backend.models[species],distributed,
            population_bar[species],context)
    end
    bar
end

# ---------------------------------------------------------------------------
# Force-balance/regularization composite VJP and complete objective gradient
# ---------------------------------------------------------------------------

struct HybridAdjointObjectiveGradient{T<:AbstractFloat} <: AbstractObjectiveGradient
    force_balance_step::T
    function HybridAdjointObjectiveGradient(force_balance_step::T) where T<:AbstractFloat
        isfinite(force_balance_step)&&force_balance_step>0 || throw(ArgumentError(
            "force-balance VJP step must be positive"))
        new{T}(force_balance_step)
    end
end
HybridAdjointObjectiveGradient(;force_balance_step::Real=1e-4)=
    HybridAdjointObjectiveGradient(Float64(force_balance_step))

function _derived_pairing(bar::AtmosphereCotangent,atmosphere,context)
    local_value=zero(eltype(atmosphere.temperature))
    for name in (:pgas,:rho,:ne,:z)
        value=getfield(atmosphere,name); value===nothing && continue
        local_value+=sum(value.*getfield(bar,name))
    end
    allreduce_sum(local_value,context)
end

function _reconstruct_trial!(problem,layout,parameters,context)
    _restore_trial_reference!(problem)
    apply_control_maps!(problem.distributed,layout,parameters)
    reconstruct_force_balance_distributed!(problem.distributed,problem.model.boundary,
        problem.model.eos,problem.model.opacity,context;options=problem.model.force_options)
end

function _force_regularization_vjp!(gradient,backend::HybridAdjointObjectiveGradient,
        bar,problem,layout::ControlMapLayout{T},parameters,evaluation,context) where T
    center=scaled_parameters(layout,parameters); lower,upper=_scaled_bounds(layout)
    function functional(scaled)
        physical=_physical_from_scaled(layout,scaled)
        _reconstruct_trial!(problem,layout,physical,context)
        pairing=_derived_pairing(bar,problem.distributed.local_atmosphere,context)
        regularization=distributed_regularization_penalty(problem.distributed,problem.regularization,
            problem.dx_m,problem.dy_m,context).total
        T(pairing+regularization)
    end
    f0=T(_derived_pairing(bar,problem.distributed.local_atmosphere,context)+
        evaluation.components.regularization)
    for coordinate in eachindex(center)
        hp=min(T(backend.force_balance_step),upper[coordinate]-center[coordinate])
        hm=min(T(backend.force_balance_step),center[coordinate]-lower[coordinate])
        tolerance=eps(T)*max(abs(center[coordinate]),one(T))*10
        if hp>tolerance && hm>tolerance
            plus=copy(center); plus[coordinate]+=hp
            minus=copy(center); minus[coordinate]-=hm
            fp=functional(plus); fm=functional(minus)
            gradient[coordinate]+=(hm^2*fp+(hp^2-hm^2)*f0-hp^2*fm)/(hp*hm*(hp+hm))
        elseif hp>tolerance
            plus=copy(center); plus[coordinate]+=hp
            gradient[coordinate]+=(functional(plus)-f0)/hp
        elseif hm>tolerance
            minus=copy(center); minus[coordinate]-=hm
            gradient[coordinate]+=(f0-functional(minus))/hm
        end
    end
    _reconstruct_trial!(problem,layout,parameters,context)
    gradient
end

function _direct_bar(bar::AtmosphereCotangent,variable::Symbol)
    variable===:vlos && return bar.vz
    variable in (:temperature,:vx,:vy,:vz,:vturb) && return getfield(bar,variable)
    if variable in (:Bx,:By,:Bz)
        bar.magnetic_field===nothing && throw(ArgumentError("$variable gradient requested without B"))
        return getfield(bar.magnetic_field,variable)
    end
    throw(ArgumentError("unsupported direct control cotangent $variable"))
end

function objective_gradient!(backend::HybridAdjointObjectiveGradient,
        problem::DistributedInversionProblem,layout::ControlMapLayout{T},parameters,context) where T
    evaluation=evaluate_objective!(problem,layout,parameters,context)
    synthetic=problem.workspace.output.data; observation=problem.observation
    output_bar=similar(synthetic)
    @. output_bar=(T(2)/problem.active_residual_count)*observation.inversion_weights^2*
        (synthetic-observation.spectrum.data)/observation.sigma^2
    intrinsic_bar=similar(problem.workspace.intrinsic.data)
    observation_vjp!(intrinsic_bar,output_bar,problem.model.observation,
        problem.workspace.intrinsic,problem.distributed,context)
    atmosphere_bar=AtmosphereCotangent(problem.distributed.local_atmosphere)
    population_bar=synthesis_vjp!(atmosphere_bar,problem.model.synthesizer,
        problem.model.redistribution,problem.distributed.local_atmosphere,
        problem.workspace.populations,problem.workspace.intrinsic,intrinsic_bar,
        problem.workspace.synthesis_cache)
    predict_distributed_populations_vjp!(atmosphere_bar,problem.model.populations,
        problem.distributed,population_bar,context)
    gradient=zeros(T,layout.parameter_count)
    for spec in layout.specs
        accumulate_control_vjp!(gradient,layout,spec.variable,_direct_bar(atmosphere_bar,spec.variable),
            problem.distributed,context)
    end
    _force_regularization_vjp!(gradient,backend,atmosphere_bar,problem,layout,parameters,evaluation,context)
    ObjectiveGradientEvaluation(evaluation,gradient,1)
end
