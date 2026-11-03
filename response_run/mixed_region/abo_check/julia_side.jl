# writes RH-cubeconvol inputs (each ABO table, 400 points incl. boundaries and outside) and this port's values
using FFNOInversion, Random
Random.seed!(1)
W=@__DIR__
for (key,name) in (((0,1),"sp"),((1,2),"pd"),((2,3),"df"))
    t=FFNOInversion.barklem_table(key...)
    for (field,f) in (("cross",t.cross),("alpha",t.alpha))
        n1,n2=size(f)
        pts=[(rand()*(n2+1)-1, rand()*(n1+1)-1) for _ in 1:400]
        append!(pts,[(0.0,0.0),(n2-1.0,n1-1.0),(0.3,n1-1.5),(n2-1.7,0.2),(n2-1.0,3.5)])
        open(joinpath(W,"$(name)_$(field).in"),"w") do io
            println(io,n1," ",n2); for i in 1:n1; println(io,join(f[i,:]," ")); end
            for (x,y) in pts; println(io,x," ",y); end
        end
        open(joinpath(W,"$(name)_$(field).jl.out"),"w") do io
            for (x,y) in pts; println(io,FFNOInversion._cubeconvol(f,x,y)); end
        end
    end
end
