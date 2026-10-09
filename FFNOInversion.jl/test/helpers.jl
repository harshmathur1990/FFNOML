function test_atmosphere(; magnetic=false, T=Float64)
    grid = Grid3D(T.([-5,-3,-1,1]), T.([0,1]), T.([0,1,2]))
    shape = (4,2,3)
    temp = fill(T(5000),shape)
    zero3 = zeros(T,shape)
    B = magnetic ? MagneticField3D(fill(T(0.01),shape),zero3,fill(T(0.02),shape)) : nothing
    Atmosphere3D(grid,temp,copy(zero3),copy(zero3),copy(zero3),copy(zero3); magnetic_field=B)
end
