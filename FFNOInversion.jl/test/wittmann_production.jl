include("helpers.jl")

@testset "Wittmann/native-Julia production opacity" begin
    mktempdir() do dir
        suffix=Sys.isapple() ? ".dylib" : Sys.iswindows() ? ".dll" : ".so"
        library=joinpath(dir,"libwitt_test"*suffix)
        build=joinpath(@__DIR__,"..","scripts","build_wittmann_backend.jl")
        run(`$(Base.julia_cmd()) --project=$(joinpath(@__DIR__,"..")) $build $library`)
        pf=joinpath(@__DIR__,"..","..","scripts","pf_Kurucz.input")
        eos=WittmannEOS(library,pf); opacity=WittmannOpacity500(eos)

        temperature=reshape([3500.,4500.,5770.,8000.,12000.],5,1,1)
        pressure=reshape([0.1,1.,10.,100.,1000.],5,1,1)
        rho=similar(pressure); ne=similar(pressure); kappa=similar(pressure)
        thermodynamics!(rho,ne,eos,temperature,pressure)
        opacity500!(kappa,opacity,temperature,pressure,rho,ne)
        # Golden values generated with scripts/witt.py contOpacity at 5000 Angstrom.
        reference=reshape([4.70012390e-5,6.33135502e-5,8.11710008e-4,9.60884122e-2,7.62910920],5,1,1)
        @test maximum(abs.(kappa.-reference)./reference)<3e-4
        @test all(kappa.>0) && all(isfinite,kappa)

        grid=Grid3D([-5.,-4.,-3.,-2.],[0.,50e3],[0.,50e3]); shape=(4,2,2); zero3=zeros(shape)
        he=Atmosphere3D(grid,fill(5500.,shape),copy(zero3),copy(zero3),copy(zero3),copy(zero3))
        options=ForceBalanceOptions(max_iterations=300,relative_tolerance=1e-5,force_tolerance=0.6,
                                    height_tolerance_m=1.0,relaxation=0.4)
        dhe=reconstruct_force_balance!(he,HE3DBoundaryState(1e-10,1.0,:top),eos,opacity;options=options)
        @test dhe.converged && dhe.mode==:HE3D && dhe.lorentz_max_n_m3==0
        @test all(diff(he.z[:,1,1]).<0)

        bx=fill(1e-5,shape); by=zeros(shape); bz=fill(2e-5,shape)
        for k in axes(by,1); @views by[k,:,:].=(k-1)*1e-9; end
        mh=Atmosphere3D(grid,fill(5500.,shape),copy(zero3),copy(zero3),copy(zero3),copy(zero3);
                        magnetic_field=MagneticField3D(bx,by,bz))
        dmh=reconstruct_force_balance!(mh,HE3DBoundaryState(1e-10,1.0,:top),eos,opacity;options=options)
        @test dmh.converged && dmh.mode==:MHS && dmh.lorentz_max_n_m3>0
        @test maximum(abs.(mh.pgas.-he.pgas))>0
    end
end
