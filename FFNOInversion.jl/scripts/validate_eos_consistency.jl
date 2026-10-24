# Compare the HSEMHS EOS table (equilibrium ρ, nₑ) with FFNO's Wittmann library (synthesis) over a (T, P) envelope.
#
# usage: julia --project=. scripts/validate_eos_consistency.jl LIBWITT TABLE [ATMOSPHERE_OUTPUT.h5]
#
# Both are built from the same Wittmann source and partition functions; the table uses the HSEMHS shim (outer
# tolerance 1e-10, FIX_GASC_H2) and a direct/blend/hot branch policy, the library the unmodified source. Points come from
# a regular log T × log P grid covering the photosphere–chromosphere, plus — if an FFNO atmosphere output is given —
# every (T, Pgas) node of it. Differences are reported per region: direct (T < T_sw), blend (T_sw ≤ T < 1.1 T_sw), hot.
using FFNOInversion, HDF5, Printf

_quantile(v,p)=(s=sort(v); s[clamp(round(Int,p*(length(s)-1))+1,1,length(s))])

function main(args)
    length(args) in (2,3) || error("usage: julia --project=. scripts/validate_eos_consistency.jl LIBWITT TABLE [ATMOSPHERE.h5]")
    library,table_path=args[1],args[2]
    pf=normpath(joinpath(@__DIR__,"..","..","scripts","pf_Kurucz.input"))
    eos=WittmannEOS(library,pf); table=FFNOInversion.HSEMHS.load_table(table_path)
    T=Float64[]; P=Float64[]
    for lt in range(log10(3000.0),log10(3.0e4);length=120), lp in range(-3.0,5.0;length=60)
        push!(T,exp10(lt)); push!(P,exp10(lp))
    end
    if length(args)==3
        h5open(args[3],"r") do f
            haskey(f,"pgas") || error("$(args[3]) has no pgas")
            append!(T,vec(read(f["temperature"]))); append!(P,vec(read(f["pgas"])))
        end
    end
    n=length(T)
    rho=fill(NaN,n); ne=fill(NaN,n); library_failed=falses(n)
    for i in 1:n                       # point by point: the library throws for a whole batch if one cell fails
        r=zeros(1,1,1); e=zeros(1,1,1)
        try
            thermodynamics!(r,e,eos,fill(T[i],1,1,1),fill(P[i],1,1,1)); rho[i]=r[1]; ne[i]=e[1]
        catch
            library_failed[i]=true
        end
    end
    dr=fill(NaN,n); dn=fill(NaN,n); region=Vector{Symbol}(undef,n); table_out=falses(n)
    for i in 1:n
        lr,ln,_,st=FFNOInversion.HSEMHS.state(table,log(T[i]),log(P[i]))
        table_out[i]=!isfinite(lr)
        dr[i]=exp(lr)/rho[i]-1; dn[i]=exp(ln)/ne[i]-1
        region[i]=T[i]<table.T_sw ? :direct : T[i]<1.1*table.T_sw ? :blend : :hot
    end
    println(@sprintf("table T_sw = %.1f K, hot branch %s, %d points", table.T_sw, table.hot_branch, n))
    println(@sprintf("library failures: %d points; outside the table: %d points; table valid but library failed: %d",
        count(library_failed),count(table_out),count(library_failed .& .!table_out)))
    for r in (:direct,:blend,:hot)
        idx=findall(i->region[i]==r && isfinite(dr[i]) && isfinite(dn[i]),1:n); isempty(idx) && continue
        q(v)=(maximum(abs,v[idx]),_quantile(abs.(v[idx]),0.99),_quantile(abs.(v[idx]),0.5))
        println(@sprintf("%-6s n=%5d  rho rel diff max %.2e p99 %.2e median %.2e | ne rel diff max %.2e p99 %.2e median %.2e",
            r,length(idx),q(dr)...,q(dn)...))
    end
end

main(ARGS)
