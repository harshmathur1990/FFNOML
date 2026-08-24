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

function FFNOInversion.build_muspel_line_model(atmosphere,hydrogen_populations,atom_file::AbstractString,line_index::Integer,
                                                lower_level::Integer,upper_level::Integer;
                                                voigt=(a_min=1f-4,a_max=1f1,a_n=20000,v_min=0f0,v_max=5f2,v_n=2500),
                                                include_continuum=true)
    atom=Muspel.read_atom(atom_file); line=atom.lines[line_index]
    matmos=_muspel_atmosphere(atmosphere,hydrogen_populations)
    names=("Al.yaml","C.yaml","Ca.yaml","Fe.yaml","H_6.yaml","He.yaml","KI.yaml","Mg.yaml","N.yaml","Na.yaml","NiI.yaml","O.yaml","S.yaml","Si.yaml")
    files=[joinpath(Muspel.AtomicData.get_atom_dir(),name) for name in names]
    continuum=Muspel.get_σ_itp(matmos,line.λ0,files)
    vitp=Muspel.create_voigt_itp(LinRange(Float32(voigt.a_min),Float32(voigt.a_max),voigt.a_n),
                                  LinRange(Float32(voigt.v_min),Float32(voigt.v_max),voigt.v_n))
    MuspelLineOpacityModel(line,continuum,vitp,hydrogen_populations,Int(lower_level),Int(upper_level),include_continuum)
end

function FFNOInversion.add_opacity_emissivity!(chi,eta,model::MuspelLineOpacityModel,wavelength,atmosphere,x,y,populations)
    populations===nothing && throw(ArgumentError("Muspel FFNO line populations are required"))
    hp=model.hydrogen_populations; line=model.line
    @inbounds for k in axes(chi,1)
        temp=Float32(atmosphere.temperature[k,x,y]); ne=Float32(atmosphere.ne[k,x,y])
        hi=Float32(sum(@view hp[k,x,y,1:5])); proton=Float32(hp[k,x,y,6])
        alpha_c=Muspel.α_cont(model.continuum,temp,ne,hi,proton)
        j_c=alpha_c*Muspel.blackbody_λ(line.λ0,temp)
        gamma=Muspel.calc_broadening(line.γ,temp,ne,hi)
        dwidth=Muspel.doppler_width(line.λ0,line.mass,temp)
        nu=populations[k,x,y,model.upper_level]; nl=populations[k,x,y,model.lower_level]
        gamma_energy=6.62607015e-34*299792458.0/(4pi*line.λ0*1e-9)
        for l in eachindex(wavelength)
            lambda_nm=wavelength[l]*1e9
            a=Muspel.damping(gamma,lambda_nm,dwidth)
            v=(lambda_nm-line.λ0+line.λ0*atmosphere.vz[k,x,y]/299792458.0)/dwidth
            profile=real(model.voigt(a,abs(v)))/(sqrt(pi)*dwidth)
            factor=gamma_energy*profile
            alpha=factor*(nl*line.Blu-nu*line.Bul)*1e9+(model.include_continuum ? alpha_c : 0)
            emiss=factor*nu*line.Aul*1e-3+(model.include_continuum ? j_c : 0)
            chi[k,l]+=alpha; eta[k,l]+=emiss
        end
    end
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
