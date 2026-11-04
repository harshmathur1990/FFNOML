using FFNOInversion

@testset "FFNO lines follow from the region and the atom files" begin
    dir=get(ENV,"FFNO_ATOM_DIR","/mn/stornext/u3/harshm/Documents/WorkRepo/multi3d/input/atoms")
    atoms=Dict(:H=>joinpath(dir,"atom.h6_tiago2.yaml"),:CA=>joinpath(dir,"atom.ca2.yaml"))
    lines=ffno_atom_lines(atoms)
    @test length(lines)==15 && issorted(lines;by=l->l.atom_vacuum_nm)
    @test all(l->l.reference_vacuum_nm!==nothing && l.level_count==6,lines)
    region(start_nm,step_nm,count)=SpectralRegionConfig(start_nm*1e-9,step_nm*1e-9,count,1.0,:none,nothing,[SpectralSourceConfig(:ffno,nothing)])
    found(args...;kwargs...)=[(l.species,l.line_index,l.lower_level,l.upper_level) for l in ffno_region_lines(region(args...),lines;kwargs...)]

    # the transitions the code used to name by hand, and those it could not
    @test found(656.192,0.002,88)==[(:H,5,2,3)]
    @test found(854.023,0.004267,88)==[(:CA,5,3,5)]
    @test found(121.5,0.001,300)==[(:H,1,1,2)]       # Lyman alpha: vacuum below 200 nm
    @test found(866.0,0.004,80)==[(:CA,3,2,4)]
    @test found(849.6,0.004,80)==[(:CA,4,2,5)]
    @test found(393.2,0.004,80)==[(:CA,2,1,5)]       # Ca II K
    @test found(396.7,0.004,80)==[(:CA,1,1,4)]       # Ca II H
    @test found(393.0,0.01,500)==[(:CA,2,1,5),(:CA,1,1,4)]   # a region with both has both

    # the same wavelengths as the old table, and the same centres
    halpha=only(ffno_region_lines(region(656.192,0.002,88),lines)); ca=only(ffno_region_lines(region(854.023,0.004267,88),lines))
    @test halpha.reference_vacuum_nm==FFNOInversion._air_to_vacuum_nm(656.28)
    @test ca.reference_vacuum_nm==FFNOInversion._air_to_vacuum_nm(854.209)
    @test (halpha.atomic_number,halpha.ion_stage,ca.atomic_number,ca.ion_stage)==(1,0,20,1)
    @test (halpha.resonance,ca.resonance)==(false,false)
    @test all(l->l.resonance==(l.lower_level==1),lines)
    # every line's reference is the laboratory wavelength next to the atom's, not a neighbour's
    @test all(l->abs(l.reference_vacuum_nm-l.atom_vacuum_nm)<0.15,filter(l->l.atom_vacuum_nm<2000,lines))

    # a region without a line says which lines there are
    message=try ffno_region_lines(region(630.1,0.001,46),lines); "" catch e; sprint(showerror,e) end
    @test occursin("no line of the atoms",message) && occursin("H levels 2-3 6562.8 Å",message) && occursin("CA levels 1-5 3933.66 Å",message)
    # line_center = "atom" selects and reports on the atom's own wavelength
    @test only(ffno_region_lines(region(656.192,0.002,88),lines;line_center=:atom)).line_index==5
    @test ffno_line_center_nm(halpha,:atom)==halpha.atom_vacuum_nm

    # the levels are those Muspel's sorted levels give
    @test FFNOInversion.ffno_atom_lines(:H,atoms[:H])[5].lower_level==2

    # no reference table entry for an element: the atom's wavelength
    mktempdir() do tmp
        path=joinpath(tmp,"li.yaml")
        write(path,"element: {symbol: Li}\natomic_levels:\n  a: {energy: {value: 0.0, unit: cm^-1}, g: 2, stage: 1, label: x}\n  b: {energy: {value: 14904.0, unit: cm^-1}, g: 6, stage: 1, label: y}\nradiative_bound_bound:\n  - {transition: [b, a], f_value: 0.7}\n")
        li=only(ffno_atom_lines(:LI,path))
        @test li.reference_vacuum_nm===nothing && ffno_line_center_nm(li)==li.atom_vacuum_nm && li.atomic_number==3
    end
end
