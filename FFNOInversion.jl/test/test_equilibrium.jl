# Deterministic test double for the equilibrium interface (not a production backend): an ideal gas with constant
# opacity per unit mass in hydrostatic equilibrium, column by column. On a log-tau grid P = P_top + g τ/κ and
# dz = −dτ/(κρ) (z = 0 at the top); on a height grid ln P is integrated downward from P_top with ρ = P μ m_H/(k T) and
# τ follows from κρ. Every rank applies the same formula to its columns, so results are decomposition independent.
# `invalid_above_K` makes the reconstruction invalid on every rank when any temperature exceeds it (collective).

if !isdefined(Main,:TestHydrostaticEquilibrium)

Base.@kwdef mutable struct TestHydrostaticEquilibrium <: AbstractEquilibriumBackend
    p_top::Float64 = 1.0              # Pa
    kappa_m2_kg::Float64 = 0.02
    g::Float64 = 274.0
    mu::Float64 = 1.3
    electron_fraction::Float64 = 1e-4
    invalid_above_K::Float64 = Inf
    calls::Int = 0
    closed::Bool = false
end

const _KB = 1.380649e-23
const _MH = 1.6735575e-27

function FFNOInversion.reconstruct_equilibrium!(b::TestHydrostaticEquilibrium,distributed,context)
    t0=time(); b.calls+=1
    a=distributed.local_atmosphere; T=a.temperature; nz,nx,ny=size(T)
    hot=allreduce_max(maximum(T),context)>b.invalid_above_K
    if hot
        return EquilibriumOutcome(false,:test_invalid,:column_hse,:test_hydrostatic,0,time()-t0,
            ["temperature above $(b.invalid_above_K) K"],nothing)
    end
    P=zeros(nz,nx,ny); z=zeros(nz,nx,ny); lt=zeros(nz,nx,ny)
    depth=native_depth(a.grid)
    for i in 1:nx,j in 1:ny
        if vertical_coordinate(a.grid)===:log10tau500
            tau=exp10.(depth)
            for k in 1:nz
                P[k,i,j]=b.p_top+b.g*tau[k]/b.kappa_m2_kg; lt[k,i,j]=depth[k]
            end
            for k in 2:nz
                rho1=P[k-1,i,j]*b.mu*_MH/(_KB*T[k-1,i,j]); rho2=P[k,i,j]*b.mu*_MH/(_KB*T[k,i,j])
                z[k,i,j]=z[k-1,i,j]-(tau[k]-tau[k-1])/(b.kappa_m2_kg*0.5*(rho1+rho2))
            end
        else
            P[1,i,j]=b.p_top; z[:,i,j].=depth
            for k in 2:nz
                H=_KB*0.5*(T[k-1,i,j]+T[k,i,j])/(b.mu*_MH*b.g)
                P[k,i,j]=P[k-1,i,j]*exp((depth[k-1]-depth[k])/H)
            end
            acc=0.5*b.kappa_m2_kg*P[1,i,j]*b.mu*_MH/(_KB*T[1,i,j])*1.0e5
            for k in 1:nz
                k>1 && (acc+=b.kappa_m2_kg*0.5*(P[k-1,i,j]*b.mu*_MH/(_KB*T[k-1,i,j])+P[k,i,j]*b.mu*_MH/(_KB*T[k,i,j]))*(depth[k-1]-depth[k]))
                lt[k,i,j]=log10(acc)
            end
        end
    end
    rho=@. P*b.mu*_MH/(_KB*T); ne=@. b.electron_fraction*P/(_KB*T)
    for (name,value) in ((:pgas,P),(:rho,rho),(:ne,ne),(:z,z),(:logtau500,lt))
        dest=getfield(a,name); dest===nothing ? setfield!(a,name,value) : copyto!(dest,value)
    end
    EquilibriumOutcome(true,:converged,:column_hse,:test_hydrostatic,0,time()-t0,String[],nothing)
end

FFNOInversion.close_equilibrium!(b::TestHydrostaticEquilibrium)=(b.closed=true; nothing)

end
