@testset "Log-pressure Poisson equation" begin
    grid=Grid3D([-5.,-4.,-3.,-2.,-1.],collect(0.:48000.:144000.),collect(0.:48000.:96000.))
    shape=(5,4,3);z=zeros(shape);a=1e-6
    for k in 1:5,i in 1:4,j in 1:3
        z[k,i,j]=-(k-1)*60000.0 +(i-1)*15000.0 +(j-1)*9000.
    end
    expected=exp.(-a.*z);pb=copy(expected[1,:,:]);p=repeat(reshape(pb,1,4,3),5,1,1)
    rho=a.*p./274;fx=zeros(shape);fy=zeros(shape);fz=zeros(shape)
    T=fill(5500.,shape);v=zeros(shape);atmos=Atmosphere3D(grid,T,v,copy(v),copy(v),copy(v))
    candidate=similar(p)
    stats=FFNOInversion._solve_log_pressure_serial!(candidate,p,fx,fy,fz,rho,z,atmos,pb,collect(1:5),274.;tolerance=1e-12)
    @test stats.converged && stats.residual<=1e-12
    @test candidate≈expected rtol=1e-11
    @test candidate[1,:,:]==pb
    @test stats.clipped==0
    context=serial_context();dist=DistributedAtmosphere(atmos,grid,local_tile(context,4,3))
    distributed_candidate=similar(p)
    other=FFNOInversion._solve_log_pressure!(distributed_candidate,p,fx,fy,fz,rho,z,dist,context,pb,collect(1:5),274.;tolerance=1e-12)
    @test candidate==distributed_candidate && stats==other
    # Incompatible strong magnetic forcing still produces finite positive bounded pressure.
    fz.=10.;stats=FFNOInversion._solve_log_pressure_serial!(candidate,p,fx,fy,fz,rho,z,atmos,pb,collect(1:5),274.;max_pressure_pa=100.)
    @test all(isfinite,candidate) && all(candidate.>0)
    @test minimum(candidate)>=0.5minimum(pb)
    @test maximum(candidate)<=100.
    @test stats.clipped>0
    blended=similar(p);FFNOInversion._blend_log_pressure!(blended,4p,p,0.5)
    @test blended≈2p
    @test_throws ArgumentError ForceBalanceOptions(log_pressure_iterations=0)
    @test_throws ArgumentError ForceBalanceOptions(log_pressure_tolerance=NaN)
    @test_throws ArgumentError ForceBalanceOptions(log_pressure_max_pa=-1.)
end
