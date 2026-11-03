# Van der Waals broadening by neutral hydrogen in the Anstee, Barklem & O'Mara (ABO) theory, as RH implements it
# (barklem.c, broad.c, kurucz.c, cubeconvol.c): the tables of σ (a_0², at 10 km/s) and α over the effective principal
# quantum numbers of the two levels (s-p: Anstee & O'Mara 1995; p-d: Barklem & O'Mara 1997; d-f: Barklem, O'Mara &
# Ross 1998; data/barklem, copied from RH's Atoms directory), interpolated by cubic convolution, and the conversion to
# γ/n_H = C T^((1-α)/2) (Anstee & O'Mara 1995 eq. 3). The tables hold neutral atoms only.

struct BarklemTable
    neff1::Vector{Float64}      # effective principal quantum number of the level with the lower l
    neff2::Vector{Float64}      # ... of the level with the higher l
    cross::Matrix{Float64}      # σ [a_0²], (neff1, neff2)
    alpha::Matrix{Float64}
end

const _BARKLEM_DIR=normpath(joinpath(@__DIR__,"..","data","barklem"))
const _BARKLEM_LAYOUT=Dict((0,1)=>("Barklem_spdata.dat",21,18,1.0,1.3),(1,2)=>("Barklem_pddata.dat",18,18,1.3,2.3),
                           (2,3)=>("Barklem_dfdata.dat",18,18,2.3,3.3))
const _BARKLEM_DELTA_NEFF=0.1
const _BARKLEM_TABLES=Dict{Tuple{Int,Int},BarklemTable}()
const _BARKLEM_LOCK=ReentrantLock()

"""RH's table file: three comment lines, the σ block (n1 rows of n2), two comment lines, the α block."""
function _read_barklem_table(file,n1,n2,neff1_0,neff2_0)
    numbers=Float64[]; blocks=Vector{Float64}[]
    for line in eachline(joinpath(_BARKLEM_DIR,file))
        if startswith(line,'c')
            isempty(numbers) || (push!(blocks,numbers); numbers=Float64[])
        else
            append!(numbers,parse.(Float64,split(line)))
        end
    end
    isempty(numbers) || push!(blocks,numbers)
    length(blocks)==2 && all(b->length(b)==n1*n2,blocks) || throw(ArgumentError("malformed Barklem table $file"))
    BarklemTable(neff1_0 .+ _BARKLEM_DELTA_NEFF .* (0:n1-1),neff2_0 .+ _BARKLEM_DELTA_NEFF .* (0:n2-1),
                 permutedims(reshape(blocks[1],n2,n1)),permutedims(reshape(blocks[2],n2,n1)))
end

"""The ABO table for orbital quantum numbers `l1`, `l2` of the two levels (any order), or `nothing` (only s-p, p-d,
d-f are tabulated)."""
function barklem_table(l1::Integer,l2::Integer)
    key=(min(l1,l2),max(l1,l2)); haskey(_BARKLEM_LAYOUT,key) || return nothing
    lock(_BARKLEM_LOCK) do
        get!(_BARKLEM_TABLES,key) do
            _read_barklem_table(_BARKLEM_LAYOUT[key]...)
        end
    end
end

@inline _cc_kernel(s)=(-s*(1-s*(2-s))/2,1+s*s*(3s-5)/2,s*(1+s*(4-3s))/2,s*s*(s-1)/2)

"""RH's cubic convolution (Keys 1981) of `f` (rows y, columns x) at fractional 0-based indices (x, y); points beyond the
table are linear extrapolations f(-1) = 3f(0) - 3f(1) + f(2), as in RH."""
function _cubeconvol(f::AbstractMatrix,x,y)
    ny,nx=size(f)
    i,ux = x<=0 ? (0,_cc_kernel(0.0)) : x>=nx-1 ? (nx-2,_cc_kernel(1.0)) : (floor(Int,x),_cc_kernel(x-floor(x)))
    j,uy = y<=0 ? (0,_cc_kernel(0.0)) : y>=ny-1 ? (ny-2,_cc_kernel(1.0)) : (floor(Int,y),_cc_kernel(y-floor(y)))
    column(r,c)= c<0 ? 3f[r+1,1]-3f[r+1,2]+f[r+1,3] : c>nx-1 ? 3f[r+1,nx]-3f[r+1,nx-1]+f[r+1,nx-2] : f[r+1,c+1]
    value(r,c)= r<0 ? 3column(0,c)-3column(1,c)+column(2,c) : r>ny-1 ? 3column(ny-1,c)-3column(ny-2,c)+column(ny-3,c) :
        column(r,c)
    sum(value(j-1+m,i-1+n)*ux[n+1]*uy[m+1] for m in 0:3, n in 0:3)
end

"""Effective principal quantum number of a level `ionization_j - energy_j` below the next ion's ground (charge `z` of
the ion the electron leaves, 1 for a neutral atom), with RH's reduced-mass Rydberg."""
function effective_n(ionization_j,energy_j,z,mass_amu)
    rydberg=2.1798741e-18/(1+9.1093837015e-31/(mass_amu*1.66053906660e-27))
    ionization_j>energy_j || return NaN
    z*sqrt(rydberg/(ionization_j-energy_j))
end

"""σ [a_0²] and α of a neutral atom's line from the ABO tables (RH `getBarklemcross`/`getBarklemactivecross`), for the
lower and upper levels' orbital quantum numbers and effective principal quantum numbers; `nothing` when no table applies
or the levels lie outside it (RH then uses Unsöld)."""
function barklem_sigma_alpha(l_lower,l_upper,neff_lower,neff_upper)
    (l_lower===nothing || l_upper===nothing) && return nothing
    table=barklem_table(l_lower,l_upper); table===nothing && return nothing
    neff1,neff2 = l_lower>l_upper ? (neff_upper,neff_lower) : (neff_lower,neff_upper)
    (table.neff1[1]<=neff1<=table.neff1[end] && table.neff2[1]<=neff2<=table.neff2[end]) || return nothing
    x=(neff2-table.neff2[1])/_BARKLEM_DELTA_NEFF; y=(neff1-table.neff1[1])/_BARKLEM_DELTA_NEFF
    (_cubeconvol(table.cross,x,y),_cubeconvol(table.alpha,x,y))
end

"""C [m³ s⁻¹] in γ_vdW/n_H = C T^((1-α)/2) for ABO σ [a_0²] and α (RH `getBarklemcross`; Anstee & O'Mara 1995 eq. 3, γ
the full width)."""
function barklem_coefficient(sigma,alpha,mass_amu)
    amu=1.66053906660e-27; bohr=5.29177210903e-11
    reduced=amu/(1/1.008+1/mass_amu)
    vbar=sqrt(8*1.380649e-23/(pi*reduced))
    2*(4/pi)^(alpha/2)*_gamma_function((4-alpha)/2)*vbar*bohr^2*(vbar/1e4)^(-alpha)*sigma
end

# Γ(x) for the ABO conversion (x = (4-α)/2 in [1.5, 2]): Lanczos (g = 7, n = 9), relative error < 1e-13
const _LANCZOS=(0.99999999999980993,676.5203681218851,-1259.1392167224028,771.32342877765313,-176.61502916214059,
    12.507343278686905,-0.13857109526572012,9.9843695780195716e-6,1.5056327351493116e-7)
function _gamma_function(x)
    x<0.5 && return pi/(sin(pi*x)*_gamma_function(1-x))
    x-=1; a=_LANCZOS[1]; t=x+7.5
    for i in 1:8; a+=_LANCZOS[i+1]/(x+i); end
    sqrt(2pi)*t^(x+0.5)*exp(-t)*a
end

"""Orbital quantum number of the valence electron in a Kurucz (K94) level label, as RH reads it without explicit
labels (`RLKdet_level`): the label's last word must end in a multiplicity and an upper-case term letter, and l is the
last character of the first word (e.g. "(4F)4p y3D" -> p). `nothing` when undetermined."""
function kurucz_label_orbital(label::AbstractString)
    words=split(label); isempty(words) && return nothing
    last=words[end]; length(last)>=2 || return nothing
    (isdigit(last[end-1]) && isuppercase(last[end])) || return nothing
    _orbital(words[1][end])
end

"""Orbital quantum number of the valence electron in an RH/Muspel atom level label, as RH's `determinate_abo` reads
it: the label is cut after its parity letter (E or O, last occurrence) and l is the letter of the second-last word
(e.g. "CA II 3P6 3D 2DE 5/2" -> D). `nothing` when undetermined."""
function atom_label_orbital(label::AbstractString)
    cut=findlast(c->c=='E'||c=='O',label); (cut===nothing || cut==1) && return nothing
    words=split(label[1:cut]); length(words)>=2 || return nothing
    m=match(r"^\d+([A-Za-z])",words[end-1]); m===nothing && return nothing
    _orbital(only(m.captures[1]))
end

_orbital(c::AbstractChar)=get(Dict('S'=>0,'P'=>1,'D'=>2,'F'=>3,'G'=>4,'H'=>5),uppercase(c),nothing)

const _ATOM_ENERGY_UNITS=Dict("cm^-1"=>6.62607015e-34*299792458.0*100.0,"ev"=>1.602176634e-19,"j"=>1.0)
function _atom_energy_j(level)
    e=level["energy"]; e isa AbstractDict || return Float64(e)*_ATOM_ENERGY_UNITS["cm^-1"]
    unit=lowercase(replace(string(get(e,"unit","cm^-1"))," "=>""))
    haskey(_ATOM_ENERGY_UNITS,unit) || throw(ArgumentError("atom level energy unit $(e["unit"]) not supported"))
    Float64(e["value"])*_ATOM_ENERGY_UNITS[unit]
end
_abo_value(m,keys...)=(k=findfirst(k->haskey(m,k),keys); k===nothing ? nothing :
    (v=m[keys[k]]; Float64(v isa AbstractDict ? v["value"] : v)))
_is_type(m,name)=lowercase(string(get(m,"type","")))==name

"""Atom file (Muspel YAML) for the FFNO lines with ABO van der Waals broadening resolved as RH does for atom lines
(readatom.c, barklem.c, broad.c). A `VanderWaals_ABO` entry with σ [a_0²] and α (`σ`/`sigma`, `α`/`alpha`, a number
or {value: ...}) is used as given (RH ABO_EXPL); without them σ and α come from RH's tables for a neutral atom whose
two level labels give s-p, p-d or d-f valence electrons (RH BARKLEM); otherwise the entry becomes Unsöld with its
`h_coefficient`/`he_coefficient` (default 1) and a warning (RH's fallback). Each ABO entry gets RH's Unsöld helium term
(he_coefficient 1) unless the line already has a VanderWaals_Unsold entry. Returns `atom_file` itself when it has no
ABO entry, else the path of a resolved copy."""
function resolve_abo_broadening(atom_file::AbstractString)
    data=YAML.load_file(atom_file)
    transitions=get(data,"radiative_bound_bound",Any[])
    any(t->any(m->_is_type(m,"vanderwaals_abo"),get(t,"broadening",Any[])),transitions) || return atom_file
    levels=data["atomic_levels"]; mass=Float64(data["element"]["atomic_mass"]["value"])
    for t in transitions
        mechanisms=get(t,"broadening",Any[]); any(m->_is_type(m,"vanderwaals_abo"),mechanisms) || continue
        a,b=t["transition"]; lo,up = _atom_energy_j(levels[a])<=_atom_energy_j(levels[b]) ? (a,b) : (b,a)
        stage=levels[up]["stage"]
        continuum=[_atom_energy_j(l) for l in values(levels) if l["stage"]==stage+1]
        has_unsold=any(m->_is_type(m,"vanderwaals_unsold"),mechanisms)
        resolved=Any[]
        for m in mechanisms
            if !_is_type(m,"vanderwaals_abo"); push!(resolved,m); continue; end
            sigma=_abo_value(m,"σ","sigma"); alpha=_abo_value(m,"α","alpha")
            if sigma===nothing || alpha===nothing
                abo=nothing
                if levels[lo]["stage"]==1 && !isempty(continuum)   # the tables hold neutral atoms only
                    χ∞=minimum(continuum); neff(id)=effective_n(χ∞,_atom_energy_j(levels[id]),1,mass)
                    abo=barklem_sigma_alpha(atom_label_orbital(string(levels[lo]["label"])),
                        atom_label_orbital(string(levels[up]["label"])),neff(lo),neff(up))
                end
                if abo===nothing
                    @warn "ABO broadening of $up -> $lo in $atom_file: no ABO table applies; using Unsöld (as RH)"
                    push!(resolved,Dict{Any,Any}("type"=>"VanderWaals_Unsold","h_coefficient"=>something(_abo_value(m,"h_coefficient"),1.0),
                        "he_coefficient"=>something(_abo_value(m,"he_coefficient"),1.0)))
                    has_unsold=true; continue
                end
                sigma,alpha=abo
            end
            push!(resolved,Dict{Any,Any}("type"=>"VanderWaals_ABO","σ"=>Dict{Any,Any}("value"=>sigma,"unit"=>"a_0^2"),
                "α"=>Dict{Any,Any}("value"=>alpha,"unit"=>"m/m")))
        end
        any(m->_is_type(m,"vanderwaals_abo"),resolved) && !has_unsold &&
            push!(resolved,Dict{Any,Any}("type"=>"VanderWaals_Unsold","h_coefficient"=>0.0,"he_coefficient"=>1.0))
        t["broadening"]=resolved
    end
    path=joinpath(mktempdir(),basename(atom_file))
    YAML.write_file(path,data)
    path
end
