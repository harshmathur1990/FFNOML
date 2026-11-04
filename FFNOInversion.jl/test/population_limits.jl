using Test
using FFNOInversion
using HDF5
import Muspel

# [physics] ffno_population_limit and ffno_no_inversion: what synthesis reads of an FFNO line, its gradient, and the configuration.
@testset "FFNO population limits" begin
    F=FFNOInversion
    atom=joinpath(get(ENV,"FFNO_ATOM_DIR","/mn/stornext/u3/harshm/Documents/WorkRepo/multi3d/input/atoms"),"atom.h6_tiago2.yaml")
    nz=5; nx=1; ny=1
    grid=Grid3D(VerticalAxis(:z_m,collect(range(4e6,0.0;length=nz))),[0.0],[0.0],ViewGeometry())
    temperature=reshape([9000.0,8000.0,7000.0,6000.0,5500.0],nz,1,1)
    a=Atmosphere3D(grid,temperature,zeros(nz,1,1),zeros(nz,1,1),zeros(nz,1,1),zeros(nz,1,1);z=repeat(reshape(collect(range(4e6,0.0;length=nz)),nz,1,1),1,1,1))
    a.rho=reshape([1e-9,1e-8,1e-7,1e-6,1e-5],nz,1,1); a.pgas=a.rho.*3e4; a.ne=a.rho.*1e24
    nh=a.rho./F._GIVEN_MASS_PER_H_KG
    fractions=[0.7,1e-3,1e-4,1e-5,1e-6,0.3]
    p=[nh[k,1,1]*fractions[l] for k in 1:nz,_ in 1:1,_ in 1:1,l in 1:6]
    p[1,1,1,:].*=1000.0                     # level 1: far over the limit
    p[2,1,1,3]=p[2,1,1,2]*5.0               # level 2: inverted, the total within the limit
    p[3,1,1,:].*=20.0; p[3,1,1,3]=p[3,1,1,2]*4.0   # level 3: both
    off=build_muspel_line_model(a,p,atom,5,2,3;include_continuum=false)
    # broadening reads its own copy of the hydrogen, so that the finite differences below move only what the limits act on
    on=build_muspel_line_model(a,copy(p),atom,5,2,3;include_continuum=false,population_limit=1.09,no_inversion=true)
    capped=build_muspel_line_model(a,p,atom,5,2,3;include_continuum=false,no_inversion=true)
    limited=build_muspel_line_model(a,p,atom,5,2,3;include_continuum=false,population_limit=1.09)
    ext=Base.get_extension(F,:FFNOInversionMuspelExt)
    r=off.line.Blu/off.line.Bul
    @test r≈18/8
    levels(m,k)=ext._limited_levels(m,p,a,k,1,1)
    # limits off: raw values
    @test levels(off,1)[1:2]==(p[1,1,1,2],p[1,1,1,3]) && levels(off,1)[3:4]==(1.0,false)
    # conservation: every level scaled so that the total is at the limit
    s=1.09*nh[1,1,1]/sum(p[1,1,1,:])
    @test levels(limited,1)[1]≈p[1,1,1,2]*s && levels(limited,1)[2]≈p[1,1,1,3]*s && levels(limited,1)[3]≈s
    # no inversion: the cap gives zero line opacity
    nl,nu,_,was=levels(capped,2); @test was && nu≈nl*r && abs(nl*off.line.Blu-nu*off.line.Bul)<=1e-12*nl*off.line.Blu
    # both; and a normal level is untouched
    nl,nu,scale,was=levels(on,3); @test scale<1 && was && nu≈nl*r
    @test levels(on,4)[1:2]==(p[4,1,1,2],p[4,1,1,3]) && levels(on,4)[3:4]==(1.0,false)
    # the line opacity is never negative with the limits, and is without them
    w=[656.46e-9,656.47e-9,656.48e-9]; opacity(m,q=p)=begin chi=zeros(nz,3); eta=zeros(nz,3); F.add_opacity_emissivity!(chi,eta,m,w,a,1,1,q); (chi,eta) end
    @test any(<(0),opacity(off)[1]) && all(>=(-1e-20),opacity(on)[1])
    # an atmosphere without rho has no total to compare with
    b=deepcopy(a); b.rho=nothing; @test ext._limited_levels(on,p,b,1,1,1)[3]==1.0
    # the abundance: Ca's limit is smaller by its abundance
    ca=build_muspel_line_model(a,p,joinpath(get(ENV,"FFNO_ATOM_DIR","/mn/stornext/u3/harshm/Documents/WorkRepo/multi3d/input/atoms"),"atom.ca2.yaml"),5,3,5;include_continuum=false,population_limit=1.09)
    @test ca.total_per_rho≈1.09*exp10(6.34-12)/F._GIVEN_MASS_PER_H_KG
    # the gradient against finite differences, in every level and in rho, on each kind of level
    chi_bar=reshape(collect(1.0:nz*3),nz,3).*1e3; eta_bar=reshape(collect(1.0:nz*3),nz,3).*1e-9
    # the pairing of one level: the others would only add rounding error to the differences
    pairing(q,atm,k)=begin c=zeros(nz,3); e=zeros(nz,3); F.add_opacity_emissivity!(c,e,on,w,atm,1,1,q); sum(chi_bar[k,:].*c[k,:])+sum(eta_bar[k,:].*e[k,:]) end
    atmosphere_bar=F.AtmosphereCotangent(a); population_bar=zeros(size(p))
    F.add_opacity_emissivity_vjp!(atmosphere_bar,population_bar,source->(source===p ? population_bar : zeros(size(source))),chi_bar,eta_bar,on,w,a,1,1,p)
    for k in 1:4, l in (1,2,3,6)
        h=1e-5*p[k,1,1,l]; q=copy(p); q[k,1,1,l]+=h; fp=pairing(q,a,k); q[k,1,1,l]-=2h; fm=pairing(q,a,k)
        @test population_bar[k,1,1,l]≈(fp-fm)/(2h) rtol=1e-3 atol=1e-3*maximum(abs,population_bar[k,1,1,:])
    end
    for k in (1,3)                          # the levels over the limit; elsewhere rho does not enter
        h=1e-5*a.rho[k,1,1]; plus=deepcopy(a); plus.rho[k,1,1]+=h; minus=deepcopy(a); minus.rho[k,1,1]-=h
        @test atmosphere_bar.rho[k,1,1]≈(pairing(p,plus,k)-pairing(p,minus,k))/(2h) rtol=1e-3
    end
    @test atmosphere_bar.rho[2,1,1]==0 && atmosphere_bar.rho[4,1,1]==0
    # configuration: on by default, each can be changed, and a bad value is refused
    @test F._ffno_population_limit(Dict{String,Any}())==1.09
    @test F._ffno_population_limit(Dict{String,Any}("ffno_population_limit"=>0.0))==0.0
    @test_throws ArgumentError F._ffno_population_limit(Dict{String,Any}("ffno_population_limit"=>-1.0))
end
