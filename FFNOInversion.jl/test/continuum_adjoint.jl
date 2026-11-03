using Test
using FFNOInversion
using Libdl

# The Wittmann opacities are local in depth, so their adjoint perturbs one level and recomputes only that level, and the EOS is
# the persistent backend, not a new one per call. Same answer as perturbing and recomputing the whole column.
@testset "Wittmann opacity adjoint is local in depth" begin
    mktempdir() do dir
        lib=joinpath(dir,"libwitt_adjoint.$(Libdl.dlext)")
        run(`$(Base.julia_cmd()) --project=$(joinpath(@__DIR__,"..")) $(joinpath(@__DIR__,"..","scripts","build_wittmann_backend.jl")) $lib`)
        eos=WittmannEOS(lib,joinpath(@__DIR__,"..","..","scripts","pf_Kurucz.input"))
        function column(nz)
            z=collect(range(8e5,0.0;length=nz)); shape=(nz,1,1); f(v)=reshape(collect(Float64,v),shape)
            Temp=range(4400.0,6400.0;length=nz); P=10 .^ range(1.3,4.0;length=nz)
            FFNOInversion.Atmosphere3D(Grid3D(VerticalAxis(:z_m,z),[0.0],[0.0]),f(Temp),f(100 .* (1:nz)),f(-50 .* (1:nz)),
                f(800 .* sin.(1:nz)),f(1000 .+ 30 .* (1:nz));pgas=f(P),rho=f(3e-8 .* P),ne=f(1e17 .* P.^0.6),z=f(z))
        end
        fe=joinpath(dir,"fe.list")
        write(fe,"  656.9214 -0.420 26.00   53393.669  4.0 (4F)4d g5F   38175.352  3.0 (4F)4p y3D  8.36 -4.38 -7.70FMW  0 0  0 0.000  0 0.000    0    0           1287 1324\n")
        lines=load_kurucz_linelist(fe).lines
        continuum=KuruczLTEModel(KuruczLine{Float64}[],eos;include_continuum=true)
        both=KuruczLTEModel(lines,eos;include_continuum=true)
        linesonly=KuruczLTEModel(lines,eos;include_continuum=false)

        # the persistent backend gives the same state as the per-call route
        a=column(8); T=a.temperature[:,1,1]; P=a.pgas[:,1,1]
        persistent=continuum_state(continuum.eos,T,P;threads=1); percall=continuum_state(eos,T,P;threads=1)
        @test persistent.xne_cm3==percall.xne_cm3 && persistent.partials==percall.partials && persistent.rho_kg_m3==percall.rho_kg_m3

        # one level alone gives that level's opacity in the whole column
        wave=collect(range(656.9e-9,657.2e-9;length=5))
        full=zeros(8,5),zeros(8,5); FFNOInversion._add_wittmann_kurucz!(full...,both,wave,a,1,1)
        single=zeros(1,5),zeros(1,5); FFNOInversion._add_wittmann_kurucz!(single...,both,wave,a,1,1;levels=3:3)
        @test single[1][1,:]==full[1][3,:] && single[2][1,:]==full[2][3,:]

        # the adjoint equals the whole-column numerical adjoint, on and off the vertical
        for view in (ViewGeometry(),ViewGeometry(mu=0.8,azimuth_deg=30.0,geometry=:inclined)), model in (continuum,both,linesonly)
            b=column(8); g=b.grid; b=FFNOInversion.Atmosphere3D(Grid3D(g.vertical,g.x,g.y,view),b.temperature,b.vx,b.vy,b.vz,b.vturb;pgas=b.pgas,rho=b.rho,ne=b.ne,z=b.z)
            chi_bar=reshape(sin.(1:40),8,5); eta_bar=reshape(cos.(1:40),8,5)
            local_bar=FFNOInversion.AtmosphereCotangent(b); whole=FFNOInversion.AtmosphereCotangent(b)
            FFNOInversion.add_opacity_emissivity_vjp!(local_bar,nothing,nothing,chi_bar,eta_bar,model,wave,b,1,1,nothing)
            FFNOInversion._opaque_opacity_vjp!(whole,chi_bar,eta_bar,model,wave,b,1,1,nothing)
            for name in (:temperature,:pgas,:ne,:vturb,:vx,:vy,:vz,:rho)
                x=getfield(local_bar,name); y=getfield(whole,name)
                @test maximum(abs.(x.-y))<=1e-5*max(maximum(abs.(y)),1e-30)
            end
        end

        # and it scales with the column: a production-like column is far cheaper than the whole-column adjoint
        big=column(60); wave=collect(range(656.9e-9,657.2e-9;length=30)); chi_bar=ones(60,30); eta_bar=ones(60,30)
        x=FFNOInversion.AtmosphereCotangent(big); y=FFNOInversion.AtmosphereCotangent(big)
        FFNOInversion.add_opacity_emissivity_vjp!(x,nothing,nothing,chi_bar,eta_bar,continuum,wave,big,1,1,nothing)   # compile
        fast=@elapsed FFNOInversion.add_opacity_emissivity_vjp!(x,nothing,nothing,chi_bar,eta_bar,continuum,wave,big,1,1,nothing)
        slow=@elapsed FFNOInversion._opaque_opacity_vjp!(y,chi_bar,eta_bar,continuum,wave,big,1,1,nothing)
        println("CONTINUUM_ADJOINT_60_LEVELS_30_WAVELENGTHS local=$(fast) s whole_column=$(slow) s ratio=$(slow/fast)")
        @test fast*5<slow
    end
end
