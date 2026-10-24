"""FIRTEZ-style log-pressure equation adapted to our physical-edge graph.

For h=(F_L-rho*g*e_z)/P_old, solve D' W D u = D' W integral(h),
u=ln(P/1 Pa). This is a weighted graph divergence equation, not FISHPACK's
Cartesian Laplacian. The prescribed pressure plane is Dirichlet; omitted exterior
edges retain the existing natural force boundary. MPI owns only local arrays.
"""
function _solve_log_pressure!(pnew,p,fx,fy,fz,rho,z,distributed,context,pboundary,order,g;
        horizontal_weight=1.0,max_iterations=1000,tolerance=1e-8,max_pressure_pa=1e6)
    tile=distributed.tile; grid=distributed.global_grid; nz,nx,ny=size(p)
    shape=(nz,length(grid.x),length(grid.y)); boundary_k=order[1]
    hx=fx./p; hy=fy./p; hz=(fz.-rho.*g)./p
    halos=map(a->exchange_halos(_field(a,tile,shape),context,1),(hx,hy,hz,z))
    hxh,hyh,hzh,zh=halos
    weights=zeros(nz,nx,ny,6); diagonal=zeros(size(p)); rhs=zeros(size(p))
    uboundary=log.(pboundary)
    Threads.@threads :static for column in 1:nx*ny
        i=(column-1)%nx+1;j=(column-1)÷nx+1
        gi=first(tile.xrange)+i-1;gj=first(tile.yrange)+j-1
        for k in 1:nz
            k==boundary_k && continue
            for (d,dk,di,dj) in ((1,0,-1,0),(2,0,1,0),(3,0,0,-1),(4,0,0,1),(5,-1,0,0),(6,1,0,0))
                kk=k+dk;ii=gi+di;jj=gj+dj
                (1<=kk<=nz && 1<=ii<=length(grid.x) && 1<=jj<=length(grid.y)) || continue
                dz=zh[kk,i+di+1,j+dj+1]-z[k,i,j]
                dx=grid.x[ii]-grid.x[gi];dy=grid.y[jj]-grid.y[gj]
                w=(d<=4 ? horizontal_weight : 1.0)/(dx^2+dy^2+dz^2)
                edge=((hx[k,i,j]+hxh[kk,i+di+1,j+dj+1])*dx+
                      (hy[k,i,j]+hyh[kk,i+di+1,j+dj+1])*dy+
                      (hz[k,i,j]+hzh[kk,i+di+1,j+dj+1])*dz)/2
                weights[k,i,j,d]=w;diagonal[k,i,j]+=w;rhs[k,i,j]-=w*edge
                kk==boundary_k && (rhs[k,i,j]+=w*uboundary[i,j])
            end
        end
    end
    @views diagonal[boundary_k,:,:].=1
    function apply!(out,u)
        uh=exchange_halos(_field(u,tile,shape),context,1)
        fill!(out,0)
        Threads.@threads :static for column in 1:nx*ny
            i=(column-1)%nx+1;j=(column-1)÷nx+1
            for k in 1:nz
                k==boundary_k && continue
                value=diagonal[k,i,j]*u[k,i,j]
                for (d,dk,di,dj) in ((1,0,-1,0),(2,0,1,0),(3,0,0,-1),(4,0,0,1),(5,-1,0,0),(6,1,0,0))
                    w=weights[k,i,j,d]
                    (w==0 || k+dk==boundary_k) && continue
                    value-=w*uh[k+dk,i+di+1,j+dj+1]
                end
                out[k,i,j]=value
            end
        end
    end
    globaldot(a,b)=allreduce_sum(dot(vec(a),vec(b)),context)
    u=log.(p);applied=similar(u);r=similar(u);v=similar(u);direction=similar(u);product=similar(u)
    apply!(applied,u);@. r=rhs-applied
    @views r[boundary_k,:,:].=0
    @. v=r/diagonal
    copyto!(direction,v);rz=globaldot(r,v)
    rhsnorm=max(sqrt(globaldot(rhs,rhs)),eps());relative=sqrt(globaldot(r,r))/rhsnorm
    used=0;converged=relative<=tolerance
    for iteration in 1:max_iterations
        converged && break
        apply!(product,direction);denominator=globaldot(direction,product)
        isfinite(denominator) && denominator>0 || break
        alpha=rz/denominator
        @. u=u+alpha*direction
        @. r=r-alpha*product
        used=iteration;relative=sqrt(globaldot(r,r))/rhsnorm
        converged=relative<=tolerance
        converged && break
        @. v=r/diagonal
        rznew=globaldot(r,v);beta=rznew/rz
        @. direction=v+beta*direction
        rz=rznew
    end
    # Recompute the true residual to detect recurrence/roundoff drift.
    apply!(applied,u);@. r=rhs-applied
    @views r[boundary_k,:,:].=0
    relative=sqrt(globaldot(r,r))/rhsnorm;converged=relative<=tolerance
    minimum_pressure_pa=-0.5*allreduce_max(-minimum(pboundary),context)
    lower=log(minimum_pressure_pa)
    maximum_pressure_pa=max(max_pressure_pa,allreduce_max(maximum(pboundary),context))
    upper=log(maximum_pressure_pa)
    clipped=allreduce_sum(count(x->x<lower || x>upper,u),context)
    @. pnew=clamp(exp(clamp(u,lower,upper)),minimum_pressure_pa,maximum_pressure_pa)
    @views pnew[boundary_k,:,:].=pboundary
    (iterations=used,residual=relative,converged=converged,clipped=clipped,
        minimum_pressure_pa=minimum_pressure_pa,maximum_pressure_pa=maximum_pressure_pa)
end

function _solve_log_pressure_serial!(pnew,p,fx,fy,fz,rho,z,atmosphere,pboundary,order,g;kwargs...)
    context=serial_context();nz,nx,ny=size(p)
    distributed=DistributedAtmosphere(atmosphere,atmosphere.grid,local_tile(context,nx,ny))
    _solve_log_pressure!(pnew,p,fx,fy,fz,rho,z,distributed,context,pboundary,order,g;kwargs...)
end

function _blend_log_pressure!(out,candidate,current,alpha)
    @. out=exp(alpha*log(candidate)+(1-alpha)*log(current))
    out
end
