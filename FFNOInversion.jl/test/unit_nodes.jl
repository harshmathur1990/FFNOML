@testset "node expansion" begin
    grid = Grid3D([-4.0,-2.0,0.0],[0.0,1.0,2.0],[0.0,1.0])
    constant_nodes = NodeField(fill(7.0,2,1,1),[-4.0,0.0])
    @test expand_nodes(constant_nodes,grid) == fill(7.0,3,3,2)

    values = Array{Float64}(undef,2,2,2)
    for iz in 1:2, ix in 1:2, iy in 1:2
        values[iz,ix,iy] = [-4.0,0.0][iz] + 2*(ix-1) + 3*(iy-1)
    end
    expanded = expand_nodes(NodeField(values,[-4.0,0.0]),grid)
    @test expanded[1,1,1] ≈ -4
    @test expanded[2,2,2] ≈ -2 + 1 + 3
    @test expanded[3,3,2] ≈ 0 + 2 + 3
end
