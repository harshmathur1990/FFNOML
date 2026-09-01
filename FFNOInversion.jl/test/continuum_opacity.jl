using Libdl

@testset "native Julia continuum opacity" begin
    mktempdir() do dir
        suffix=Sys.isapple() ? ".dylib" : Sys.iswindows() ? ".dll" : ".so"
        library=joinpath(dir,"libwitt_native"*suffix)
        build=joinpath(@__DIR__,"..","scripts","build_wittmann_backend.jl")
        run(`$(Base.julia_cmd()) --project=$(joinpath(@__DIR__,"..")) $build $library`)
        pf=joinpath(@__DIR__,"..","..","scripts","pf_Kurucz.input")
        eos=WittmannEOS(library,pf)
        temperatures=[3200.,4500.,7730.,9000.,11999.,12000.,20000.,29999.,30000.]
        pressures=10.0 .^ range(-1,3;length=length(temperatures))
        state=continuum_state(eos,temperatures,pressures)
        wavelengths=[500.,759.,760.,1500.,1905.,2513.,3900.,5000.,10000.,30000.]
        @test all(state.rho_kg_m3.>0) && all(state.xne_cm3.>0)
        @test all(isfinite,state.partials) && all(state.partials.>=0)
        for wavelength in wavelengths
            extinction=continuum_extinction_m(state,temperatures,wavelength)
            @test all(isfinite,extinction) && all(extinction.>0)
        end

        # The original source is an oracle only; production never needs STiC.
        stic_root=abspath(get(ENV,"FFNO_STIC_ORACLE_ROOT",joinpath(@__DIR__,"..","..","..","stic")))
        cop_source=joinpath(stic_root,"src","cop.cc")
        if isfile(cop_source) && !Sys.iswindows()
            oracle=joinpath(dir,"libcop_oracle"*suffix)
            run(`$(get(ENV,"CXX","c++")) -O2 -std=c++17 -shared -fPIC -I$(dirname(cop_source)) $(joinpath(@__DIR__,"cop_oracle.cpp")) $cop_source -o $oracle`)
            handle=Libdl.dlopen(oracle); fn=Libdl.dlsym(handle,:ffno_cop_oracle)
            expected=zeros(length(wavelengths),length(temperatures)); scattering=similar(expected)
            ccall(fn,Cvoid,(Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Csize_t,Ptr{Cdouble},Csize_t,Ptr{Cdouble},Ptr{Cdouble}),
                temperatures,state.xna_cm3,state.xne_cm3,state.partials,length(temperatures),wavelengths,length(wavelengths),expected,scattering)
            actual=[first(continuum_extinction_cm(temperatures[i],wavelengths[j],state.xna_cm3[i],state.xne_cm3[i],view(state.partials,:,i))) for j=eachindex(wavelengths),i=eachindex(temperatures)]
            @test maximum(abs.(actual.-expected)./max.(abs.(expected),1e-300)) < 2e-12
            @test all(scattering.>=0)
        else
            @info "STiC oracle checkout absent; direct cop.cc parity test skipped" stic_root
        end
    end
end
