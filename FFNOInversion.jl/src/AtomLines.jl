"""One bound-bound transition of an FFNO atom, as its atom file gives it."""
struct FFNOLine
    species::Symbol                                # key of `atom_files` and of the populations
    line_index::Int                                # position in the file's radiative_bound_bound, as Muspel's atom.lines
    lower_level::Int                               # population axis: the atom's levels by increasing energy, as Muspel sorts them
    upper_level::Int
    level_count::Int
    atomic_number::Int
    ion_stage::Int                                 # 0 = neutral, as a Kurucz line
    resonance::Bool                                # the lower level is the lowest of its ion: needs PRD for a faithful core and near wing
    atom_vacuum_nm::Float64                        # from the atom's level energies
    reference_vacuum_nm::Union{Nothing,Float64}    # laboratory wavelength; nothing: none tabulated
end

const _ELEMENT_SYMBOLS=("H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca",
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn")

"""Laboratory line wavelengths [nm] by atomic number: air above 200 nm, vacuum below (the atom files' level energies put Hα 0.16 Å, Paschen α 1.3 Å
and Ca II 0.03 Å off). Hα and Ca II 854.209 are the FFNO's reference values, the rest NIST's. Any atom's line takes the one that matches it."""
const FFNO_REFERENCE_NM=Dict(
    1=>(97.254,94.974,102.572,121.567,434.046,486.133,656.28,1281.807,1875.10,4051.15),
    20=>(393.366,396.847,849.802,854.209,866.214))
"""A reference wavelength is a line's when it lies within this fraction of the atom's wavelength."""
const FFNO_REFERENCE_TOLERANCE=1e-4

_vacuum_to_air_nm(λ)=λ<200.0 ? λ : λ^2/_air_to_vacuum_nm(λ)

function _reference_vacuum_nm(atomic_number,atom_vacuum_nm)
    candidates=_air_to_vacuum_nm.(get(FFNO_REFERENCE_NM,atomic_number,()))
    isempty(candidates) && return nothing
    nearest=argmin(c->abs(c-atom_vacuum_nm),candidates)
    abs(nearest-atom_vacuum_nm)<=FFNO_REFERENCE_TOLERANCE*atom_vacuum_nm ? nearest : nothing
end

"""Every bound-bound transition of an atom file. Levels are numbered by increasing energy, which is the order of the FFNO's population axis."""
function ffno_atom_lines(species::Symbol,atom_file::AbstractString)
    data=YAML.load_file(atom_file); levels=data["atomic_levels"]
    energy=Dict(id=>_atom_energy_j(level) for (id,level) in levels)
    order=sort!(collect(keys(levels));by=id->energy[id])
    index=Dict(id=>i for (i,id) in enumerate(order))
    z=something(findfirst(==(string(data["element"]["symbol"])),_ELEMENT_SYMBOLS),0)
    map(enumerate(get(data,"radiative_bound_bound",Any[]))) do (i,transition)
        a,b=transition["transition"]; lower,upper=energy[a]<=energy[b] ? (a,b) : (b,a)
        stage=levels[lower]["stage"]; ground=minimum(energy[id] for id in order if levels[id]["stage"]==stage)
        vacuum=_H*_C/(energy[upper]-energy[lower])*1e9
        FFNOLine(species,i,index[lower],index[upper],length(order),z,Int(stage)-1,energy[lower]==ground,vacuum,_reference_vacuum_nm(z,vacuum))
    end
end
"""Every transition of the `atom_files` (species => file), by wavelength."""
ffno_atom_lines(atom_files::AbstractDict)=sort!(reduce(vcat,[ffno_atom_lines(Symbol(s),String(f)) for (s,f) in atom_files];init=FFNOLine[]);by=l->l.atom_vacuum_nm)

"""Vacuum centre of the line's profile [nm] under `[physics] line_center`."""
ffno_line_center_nm(line::FFNOLine,line_center::Symbol=:reference)=
    line_center===:atom ? line.atom_vacuum_nm : something(line.reference_vacuum_nm,line.atom_vacuum_nm)

"""Species, levels and air wavelength of a line, for a report."""
describe_ffno_line(line::FFNOLine,line_center::Symbol=:reference)=
    "$(line.species) levels $(line.lower_level)-$(line.upper_level) $(round(_vacuum_to_air_nm(ffno_line_center_nm(line,line_center))*10;digits=2)) Å"

"""Distance beyond a region's vacuum grid [m] within which a line belongs to it: 20 × 2.5 km/s, for FFNO and Kurucz lines alike."""
_line_margin_m(vacuum)=maximum(vacuum)*20*2.5e3/_C

"""The lines of `atom_lines` (see `ffno_atom_lines`) that `region` synthesizes from FFNO populations: all whose centre lies on its grid. Preflight and
production both choose through this function. A region with an FFNO source and no such line is an error."""
function ffno_region_lines(region,atom_lines;line_center::Symbol=:reference)
    vacuum=_air_to_vacuum_nm.(wavelengths(region).*1e9).*1e-9
    lo,hi=extrema(vacuum); margin=_line_margin_m(vacuum)
    selected=[line for line in atom_lines if lo-margin<=ffno_line_center_nm(line,line_center)*1e-9<=hi+margin]
    isempty(selected) && throw(ArgumentError(string("mode = \"ffno\" in the region ",round(first(wavelengths(region))*1e10;digits=3),"-",
        round(last(wavelengths(region))*1e10;digits=3)," Å (air): no line of the atoms has its centre there. Lines: ",
        join((describe_ffno_line(line,line_center) for line in atom_lines),"; "))))
    selected
end
