using Test
using FFNOInversion
import Muspel

@testset "Phase 3 production Muspel adapter" begin
    T=Float32; nz,nx,ny=5,1,1
    grid=Grid3D(T[-5,-4,-3,-2,-1],T[0],T[0])
    shape=(nz,nx,ny); temp=reshape(T[5000,5250,5500,5750,6000],shape)
    z=reshape(T[4e5,3e5,2e5,1e5,0],shape); ne=reshape(T[5e16,7e16,1e17,1.5e17,2e17],shape); rho=fill(T(1e-7),shape)
    atmos=FFNOInversion.Atmosphere3D(grid,temp,zeros(T,shape),zeros(T,shape),zeros(T,shape),fill(T(1e3),shape);
        pgas=fill(T(100),shape),rho=rho,ne=ne,z=z)
    hp=fill(T(1e10),nz,nx,ny,6); hp[:,:,:,1].=T(1e16); hp[:,:,:,2].=T(2e14); hp[:,:,:,3].=T(1e10); hp[:,:,:,6].=T(1e15)
    muspel_atmos=Muspel.Atmosphere3D(nx,ny,nz,T[0],T[0],copy(z),temp,zeros(T,shape),
        zeros(T,shape),zeros(T,shape),ne,dropdims(sum(hp[:,:,:,1:5],dims=4),dims=4),
        Array(@view(hp[:,:,:,6])))
    @test muspel_atmos.z isa AbstractArray{T,3}
    atom_dir=get(ENV,"FFNO_ATOM_DIR","/mn/stornext/u3/harshm/Documents/WorkRepo/multi3d/input/atoms")
    atom_path=joinpath(atom_dir,"atom.h6_tiago2.yaml")
    cfg=(a_min=1f-4,a_max=1f1,a_n=300,v_min=0f0,v_max=5f2,v_n=500)
    model=build_muspel_line_model(atmos,hp,atom_path,5,2,3;voigt=cfg)
    wave=T.(model.line.λ).*T(1e-9)
    cube=SpectralCube(zeros(T,length(wave),1,1,1),wave,StokesSet(:I))
    synthesize!(cube,MixedIntensitySynthesizer([OpacityContributor(model,hp)],MuspelFormalSolver()),NonPRD(),atmos)

    hi=dropdims(sum(hp[:,:,:,1:5],dims=4),dims=4)
    ma=Muspel.Atmosphere3D(1,1,nz,T[0],T[0],vec(z),temp,zeros(T,shape),zeros(T,shape),zeros(T,shape),ne,hi,Array(@view(hp[:,:,:,6])))
    col=ma[:,1,1]; buf=Muspel.RTBuffer(nz,length(wave),T)
    Muspel.calc_line_prep!(model.line,buf,col,model.continuum)
    Muspel.calc_line_1D!(model.line,buf,model.line.λ,col,vec(hp[:,:,:,3]),vec(hp[:,:,:,2]),model.voigt.table)
    @test isapprox(cube.data[:,1,1,1],buf.intensity;rtol=5e-4,atol=2e-6)

    # Outside the table the Voigt function is exact (Faddeeva) instead of the linear extrapolation, which went
    # negative for damping a > a_max and gave negative line opacity in HSE-collapsed columns.
    exact(a,v)=real(Muspel.erfcx(complex(Float64(a),-Float64(v))))
    @test real(model.voigt(2f0,3f0))≈exact(2,3) rtol=1e-3
    for (a,v) in ((25f0,0f0),(25f0,40f0),(300f0,2f0),(1f0,800f0),(1f-6,0.5f0))
        @test real(model.voigt(a,v))≈exact(a,v) rtol=1e-12
        @test real(model.voigt(a,v))>0
    end
    @test real(model.voigt.table(40f0,1f0))<0   # the extrapolation this replaces (exact: +0.0141)
end
