module FFNOInversionMuspelExt

using FFNOInversion
using Muspel

function _muspel_atmosphere(atmosphere,hydrogen_populations)
    nz,nx,ny=size(atmosphere.temperature)
    size(hydrogen_populations)[1:3]==(nz,nx,ny) || throw(DimensionMismatch("H-population grid differs from atmosphere"))
    size(hydrogen_populations,4)>=6 || throw(DimensionMismatch("six-level H populations are required for Muspel continuum"))
    hi=dropdims(sum(@view(hydrogen_populations[:,:,:,1:5]),dims=4),dims=4)
    hp=@view hydrogen_populations[:,:,:,6]
    p(A)=permutedims(Float32.(A),(1,3,2))
    z=atmosphere.z isa AbstractVector ? Float32.(atmosphere.z) : p(atmosphere.z)
    Muspel.Atmosphere3D(nx,ny,nz,Float32.(atmosphere.grid.x),Float32.(atmosphere.grid.y),z,
        p(atmosphere.temperature),p(atmosphere.vx),p(atmosphere.vy),p(atmosphere.vz),p(atmosphere.ne),p(hi),p(hp))
end

"""Muspel's Voigt table, with the exact value Re/Im erfcx(a - iv) outside its (a, v) range.

`Muspel.create_voigt_itp` extrapolates linearly (`Line()`), which goes negative for damping above the table
(a > 10): van der Waals broadening in an HSE-collapsed column (n_H ~ 1e25 m^-3) then made the Ca 8542 line
opacity about -1e3 m^-1 and the formal solution grow without bound.
"""
struct BoundedVoigt{I,T}
    table::I
    a_min::T
    a_max::T
    v_max::T
end
@inline (f::BoundedVoigt)(a,v)=(f.a_min<=a<=f.a_max && abs(v)<=f.v_max) ? f.table(a,abs(v)) :
    Muspel.erfcx(complex(Float64(a),-Float64(abs(v))))

function FFNOInversion.build_muspel_line_model(atmosphere,hydrogen_populations,atom_file::AbstractString,line_index::Integer,
                                                lower_level::Integer,upper_level::Integer;
                                                voigt=(a_min=1f-4,a_max=1f1,a_n=20000,v_min=0f0,v_max=5f2,v_n=2500),
                                                include_continuum=true,neutral_hydrogen=nothing)
    atom=Muspel.read_atom(FFNOInversion.resolve_abo_broadening(atom_file)); line=atom.lines[line_index]
    matmos=_muspel_atmosphere(atmosphere,hydrogen_populations)
    names=("Al.yaml","C.yaml","Ca.yaml","Fe.yaml","H_6.yaml","He.yaml","KI.yaml","Mg.yaml","N.yaml","Na.yaml","NiI.yaml","O.yaml","S.yaml","Si.yaml")
    files=[joinpath(Muspel.AtomicData.get_atom_dir(),name) for name in names]
    continuum=Muspel.get_σ_itp(matmos,line.λ0,files)
    table=Muspel.create_voigt_itp(LinRange(Float32(voigt.a_min),Float32(voigt.a_max),voigt.a_n),
                                   LinRange(Float32(voigt.v_min),Float32(voigt.v_max),voigt.v_n))
    vitp=BoundedVoigt(table,Float32(voigt.a_min),Float32(voigt.a_max),Float32(voigt.v_max))
    MuspelLineOpacityModel(line,continuum,vitp,hydrogen_populations,Int(lower_level),Int(upper_level),include_continuum,neutral_hydrogen)
end

function FFNOInversion.add_opacity_emissivity!(chi,eta,model::MuspelLineOpacityModel,wavelength,atmosphere,x,y,populations)
    populations===nothing && throw(ArgumentError("Muspel FFNO line populations are required"))
    hp=model.hydrogen_populations; line=model.line
    neutral=model.neutral_hydrogen===nothing ? nothing :
        FFNOInversion.eos_neutral_hydrogen!(zeros(Float64,size(chi,1)),model.neutral_hydrogen,
            Float64.(atmosphere.temperature[:,x,y]),Float64.(atmosphere.pgas[:,x,y]))
    @inbounds for k in axes(chi,1)
        temp=Float32(atmosphere.temperature[k,x,y]); ne=Float32(atmosphere.ne[k,x,y])
        hi=neutral===nothing ? Float32(sum(@view hp[k,x,y,1:5])) : Float32(neutral[k]); proton=Float32(hp[k,x,y,6])
        alpha_c=model.include_continuum ? Muspel.α_cont(model.continuum,temp,ne,hi,proton) : zero(temp)
        j_c=model.include_continuum ? alpha_c*Muspel.blackbody_λ(line.λ0,temp) : zero(temp)
        gamma=Muspel.calc_broadening(line.γ,temp,ne,hi)
        # thermal and microturbulent widths in quadrature; vturb = 0 is exactly the thermal width
        dwidth=hypot(Muspel.doppler_width(line.λ0,line.mass,temp),Float32(line.λ0*atmosphere.vturb[k,x,y]/299792458.0))
        nu=populations[k,x,y,model.upper_level]; nl=populations[k,x,y,model.lower_level]
        gamma_energy=6.62607015e-34*299792458.0/(4pi*line.λ0*1e-9)
        for l in eachindex(wavelength)
            lambda_nm=wavelength[l]*1e9
            a=Muspel.damping(gamma,lambda_nm,dwidth)
            v=(lambda_nm-line.λ0+line.λ0*FFNOInversion.los_velocity(atmosphere,k,x,y)/299792458.0)/dwidth
            profile=real(model.voigt(a,abs(v)))/(sqrt(pi)*dwidth)
            factor=gamma_energy*profile
            alpha=factor*(nl*line.Blu-nu*line.Bul)*1e9+(model.include_continuum ? alpha_c : 0)
            emiss=factor*nu*line.Aul*1e-3+(model.include_continuum ? j_c : 0)
            chi[k,l]+=alpha; eta[k,l]+=emiss
        end
    end
end

function _cell_pairing(model,wavelength,chi_bar,eta_bar,temp,ne,vz,hi,proton,nu,nl,vturb)
    line=model.line
    alpha_c=model.include_continuum ? Muspel.α_cont(model.continuum,temp,ne,hi,proton) : zero(temp)
    j_c=model.include_continuum ? alpha_c*Muspel.blackbody_λ(line.λ0,temp) : zero(temp)
    gamma=Muspel.calc_broadening(line.γ,temp,ne,hi)
    dwidth=hypot(Muspel.doppler_width(line.λ0,line.mass,temp),line.λ0*vturb/299792458.0)
    gamma_energy=6.62607015e-34*299792458.0/(4pi*line.λ0*1e-9)
    result=zero(promote_type(typeof(temp),eltype(chi_bar)))
    for l in eachindex(wavelength)
        lambda_nm=wavelength[l]*1e9
        a=Muspel.damping(gamma,lambda_nm,dwidth)
        v=(lambda_nm-line.λ0+line.λ0*vz/299792458.0)/dwidth
        profile=real(model.voigt(a,abs(v)))/(sqrt(pi)*dwidth)
        factor=gamma_energy*profile
        alpha=factor*(nl*line.Blu-nu*line.Bul)*1e9+
            (model.include_continuum ? alpha_c : 0)
        emiss=factor*nu*line.Aul*1e-3+(model.include_continuum ? j_c : 0)
        result+=chi_bar[l]*alpha+eta_bar[l]*emiss
    end
    result
end

"""Cell-local numerical VJP for Muspel's cached opacity primitives.

Only the scalar opacity/emissivity preparation is perturbed. The formal solve,
the distributed forward model and FFNO inference are not repeated.
"""
function FFNOInversion.add_opacity_emissivity_vjp!(atmosphere_bar,population_bar,
        population_lookup,chi_bar,eta_bar,model::MuspelLineOpacityModel,wavelength,
        atmosphere,x,y,populations)
    populations===nothing && throw(ArgumentError("Muspel VJP requires line populations"))
    hydrogen=model.hydrogen_populations; hydrogen_bar=population_lookup(hydrogen)
    line=model.line
    state=model.neutral_hydrogen; column_T=Float64.(atmosphere.temperature[:,x,y]); column_P=Float64.(atmosphere.pgas[:,x,y])
    neutral=state===nothing ? nothing : FFNOInversion.eos_neutral_hydrogen!(zeros(Float64,length(column_T)),state,column_T,column_P)
    for k in axes(chi_bar,1)
        temp=Float32(atmosphere.temperature[k,x,y]); ne=Float32(atmosphere.ne[k,x,y])
        vz=Float32(FFNOInversion.los_velocity(atmosphere,k,x,y))
        hi=neutral===nothing ? Float32(sum(@view hydrogen[k,x,y,1:5])) : Float32(neutral[k])
        proton=Float32(hydrogen[k,x,y,6]); nu=Float32(populations[k,x,y,model.upper_level])
        nl=Float32(populations[k,x,y,model.lower_level])
        cb=@view chi_bar[k,:]; eb=@view eta_bar[k,:]
        vturb=Float32(atmosphere.vturb[k,x,y])
        values=Float32[temp,ne,vz,hi,proton,nu,nl,vturb]
        derivatives=zeros(Float64,length(values))
        for q in eachindex(values)
            center=values[q]
            scale=q in (3,8) ? max(abs(center),1f3) : max(abs(center),1f0)    # velocities
            h=1f-3*scale
            q!=3 && center>0 && (h=min(h,center/2))
            plus=copy(values); minus=copy(values); plus[q]+=h; minus[q]-=h
            fp=_cell_pairing(model,wavelength,cb,eb,plus...)
            fm=_cell_pairing(model,wavelength,cb,eb,minus...)
            derivatives[q]=(fp-fm)/(2h)
        end
        atmosphere_bar.temperature[k,x,y]+=derivatives[1]
        atmosphere_bar.ne[k,x,y]+=derivatives[2]
        FFNOInversion.add_los_cotangent!(atmosphere_bar,atmosphere,k,x,y,derivatives[3])
        if state===nothing
            @views hydrogen_bar[k,x,y,1:5].+=derivatives[4]
        else
            # n_H⁰ from the EOS at this level: its derivative in T and P joins theirs
            eos_at(t,p)=FFNOInversion.eos_neutral_hydrogen!(zeros(1),state,[t],[p])[1]
            dT=1e-3*column_T[k]; dP=1e-3*column_P[k]
            atmosphere_bar.temperature[k,x,y]+=derivatives[4]*(eos_at(column_T[k]+dT,column_P[k])-eos_at(column_T[k]-dT,column_P[k]))/(2dT)
            atmosphere_bar.pgas[k,x,y]+=derivatives[4]*(eos_at(column_T[k],column_P[k]+dP)-eos_at(column_T[k],column_P[k]-dP))/(2dP)
        end
        hydrogen_bar[k,x,y,6]+=derivatives[5]
        population_bar[k,x,y,model.upper_level]+=derivatives[6]
        population_bar[k,x,y,model.lower_level]+=derivatives[7]
        atmosphere_bar.vturb[k,x,y]+=derivatives[8]
    end
    nothing
end

function FFNOInversion.formal_solve!(out,::MuspelFormalSolver,chi,eta,z)
    n=size(chi,1); source=Vector{eltype(out)}(undef,n); tmp=similar(source)
    for l in axes(chi,2)
        @inbounds for k in 1:n; source[k]=eta[k,l]/chi[k,l]; end
        Muspel.piecewise_1D_linear!(z,@view(chi[:,l]),source,tmp)
        out[l]=tmp[1]
    end
    out
end

end
