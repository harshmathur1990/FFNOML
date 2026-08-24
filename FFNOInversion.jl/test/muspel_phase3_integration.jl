using Test
using FFNOInversion
using Muspel

@testset "Phase 3 production Muspel adapter" begin
    T=Float32; nz,nx,ny=5,1,1
    grid=Grid3D(T[-5,-4,-3,-2,-1],T[0],T[0])
    shape=(nz,nx,ny); temp=reshape(T[5000,5250,5500,5750,6000],shape)
    z=reshape(T[4e5,3e5,2e5,1e5,0],shape); ne=reshape(T[5e16,7e16,1e17,1.5e17,2e17],shape); rho=fill(T(1e-7),shape)
    atmos=FFNOInversion.Atmosphere3D(grid,temp,zeros(T,shape),zeros(T,shape),zeros(T,shape),fill(T(1e3),shape);
        pgas=fill(T(100),shape),rho=rho,ne=ne,z=z)
    hp=fill(T(1e10),nz,nx,ny,6); hp[:,:,:,1].=T(1e16); hp[:,:,:,2].=T(2e14); hp[:,:,:,3].=T(1e10); hp[:,:,:,6].=T(1e15)
    atom_path=normpath(joinpath(@__DIR__,"..","..","..","multi3d","input","atoms","atom.h6_tiago2.yaml"))
    cfg=(a_min=1f-4,a_max=1f1,a_n=300,v_min=0f0,v_max=5f2,v_n=500)
    model=build_muspel_line_model(atmos,hp,atom_path,5,2,3;voigt=cfg)
    wave=T.(model.line.λ).*T(1e-9)
    cube=SpectralCube(zeros(T,length(wave),1,1,1),wave,StokesSet(:I))
    synthesize!(cube,MixedIntensitySynthesizer([OpacityContributor(model,hp)],MuspelFormalSolver()),NonPRD(),atmos)

    hi=dropdims(sum(hp[:,:,:,1:5],dims=4),dims=4)
    ma=Muspel.Atmosphere3D(1,1,nz,T[0],T[0],vec(z),temp,zeros(T,shape),zeros(T,shape),zeros(T,shape),ne,hi,Array(@view(hp[:,:,:,6])))
    col=ma[:,1,1]; buf=Muspel.RTBuffer(nz,length(wave),T)
    Muspel.calc_line_prep!(model.line,buf,col,model.continuum)
    Muspel.calc_line_1D!(model.line,buf,model.line.λ,col,vec(hp[:,:,:,3]),vec(hp[:,:,:,2]),model.voigt)
    @test isapprox(cube.data[:,1,1,1],buf.intensity;rtol=5e-4,atol=2e-6)
end
