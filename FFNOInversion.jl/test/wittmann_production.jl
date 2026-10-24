include("helpers.jl")

@testset "Wittmann EOS library and the HSEMHS table" begin
    mktempdir() do dir
        suffix=Sys.isapple() ? ".dylib" : Sys.iswindows() ? ".dll" : ".so"
        library=joinpath(dir,"libwitt_test"*suffix)
        build=joinpath(@__DIR__,"..","scripts","build_wittmann_backend.jl")
        run(`$(Base.julia_cmd()) --project=$(joinpath(@__DIR__,"..")) $build $library`)
        pf=joinpath(@__DIR__,"..","..","scripts","pf_Kurucz.input")
        eos=WittmannEOS(library,pf)
        temperature=reshape([3500.,4500.,5770.,8000.,12000.],5,1,1)
        pressure=reshape([0.1,1.,10.,100.,1000.],5,1,1)
        rho=similar(pressure); ne=similar(pressure)
        thermodynamics!(rho,ne,eos,temperature,pressure)
        @test all(isfinite,rho) && all(>(0),rho) && all(isfinite,ne) && all(>(0),ne)
        # the HSEMHS equilibrium table is built from the same Wittmann source with tighter solver tolerances
        # (and FIX_GASC_H2): its density and electron density agree with the library well inside the table's
        # interpolation and tolerance budget at these photospheric points
        table_path=get(ENV,"HSEMHS_EOS_TABLE","/mn/stornext/d9/data/harshm/MHS/HSEMHS/cache/eos_table_fullz.h5")
        if isfile(table_path)
            table=FFNOInversion.HSEMHS.load_table(table_path)
            for idx in eachindex(temperature)
                lr,ln,_,_=FFNOInversion.HSEMHS.state(table,log(temperature[idx]),log(pressure[idx]))
                @test abs(exp(lr)/rho[idx]-1)<1e-3
                @test abs(exp(ln)/ne[idx]-1)<2e-2
            end
        else
            @info "HSEMHS EOS table not found; skipping the table/library comparison" table_path
        end
    end
end
