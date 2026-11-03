using Test
using FFNOInversion
using HDF5
import Muspel

# Kurucz lines inside FFNO regions: one emissivity unit for every model, RH's K94 damping defaults, no FFNO line counted
# twice, and the configured output unit and region normalization.
@testset "Mixed FFNO and Kurucz regions: units, damping, duplicates, outputs" begin
    # every model's emissivity uses Muspel's Planck function (kW m^-2 sr^-1 nm^-1)
    for λ in (393.4e-9, 656.46e-9, 854.44e-9), temp in (4000.0, 6000.0, 1e4)
        @test FFNOInversion.source_planck(λ, temp) ≈ Muspel.blackbody_λ(λ * 1e9, temp) rtol=1e-5
    end

    # K94 damping: a zero constant is "not given" (RH): radiative A_ul, no Stark. Van der Waals, in order: ABO σ.α in the
    # γ_vdW field, ABO tables (neutral, s-p/p-d/d-f labels), the list's γ_vdW, Unsöld
    si = "  853.6164 -0.910 14.00   49850.830  2.0 p3d  3F      61562.477  3.0 p5f'[7]     0.00  0.00  0.00KP  9999  0 0.000  0 0.000    0    0              0    0"
    fe = "  656.9214 -0.420 26.00   53393.669  4.0 (4F)4d g5F   38175.352  3.0 (4F)4p y3D  8.36 -4.38 -7.70FMW  0 0  0 0.000  0 0.000    0    0           1287 1324"
    h  = "  656.2797  0.710  1.00   97492.304  1.5 3p 2P        82259.158  0.5 2s 2S       8.79 -4.50 -7.60K94  0 0  0 0.000  0 0.000    0    0              0 2487"
    fe2 = replace(fe, " 26.00 " => " 26.01 ")                    # the same record as Fe II: no ABO table for ions
    explicit = replace(fe, "-7.70FMW" => "312.250W")             # σ = 312 a_0², α = 0.25, past the plain 18 characters
    dir = mktempdir(); path = joinpath(dir, "lines.list")
    write(path, join((si, fe, h, fe2, explicit), "\n") * "\n")
    lines = load_kurucz_linelist(path).lines
    l_si = only(filter(l -> l.atomic_number == 14, lines))
    l_fe, l_explicit = filter(l -> l.atomic_number == 26 && l.ion_stage == 0, lines)
    l_explicit.vdw_recipe === :abo_explicit || ((l_fe, l_explicit) = (l_explicit, l_fe))
    l_fe2 = only(filter(l -> l.atomic_number == 26 && l.ion_stage == 1, lines))
    @test l_si.ion_stage == 0 && l_si.upper_g == 7 && l_si.lower_g == 5
    @test l_si.grad_s ≈ 6.670e13 / (l_si.wavelength0_m * 1e9)^2 * 10^-0.910 / 7
    @test l_si.gstark_m3_s == 0 && l_si.gvdwaals_m3_s == 0
    @test l_si.vdw_recipe === :unsold && l_si.vdw_exponent == 0.3   # "p5f'[7]" gives no term: Unsöld
    @test 1e-15 < l_si.vdw_coefficient * 6000^0.3 < 1e-12          # γ_vdW / n_H [m^3 s^-1], the usual range
    @test l_fe.grad_s ≈ 10^8.36 && l_fe.gstark_m3_s ≈ 10^-4.38 * 1e-6
    @test l_fe.vdw_recipe === :abo_table && l_fe.gvdwaals_m3_s == 0  # 4p-4d of Fe I: the p-d table
    fe_lower = 38175.352 * FFNOInversion._HC_J_CM; fe_upper = fe_lower + FFNOInversion._H * FFNOInversion._C / l_fe.wavelength0_m
    ionpot = FFNOInversion._ionization_energy(26, 0)
    σα = barklem_sigma_alpha(1, 2, effective_n(ionpot, fe_lower, 1, 55.847), effective_n(ionpot, fe_upper, 1, 55.847))
    @test l_fe.vdw_coefficient ≈ barklem_coefficient(σα..., 55.847) && l_fe.vdw_exponent ≈ (1 - σα[2]) / 2
    @test 1e-15 < l_fe.vdw_coefficient * 6000^l_fe.vdw_exponent < 1e-12
    @test l_fe2.vdw_recipe === :list && l_fe2.gvdwaals_m3_s ≈ 10^-7.70 * 1e-6 && l_fe2.vdw_coefficient == 0
    @test l_explicit.vdw_coefficient ≈ barklem_coefficient(312.0, 0.25, 55.847) && l_explicit.vdw_exponent ≈ 0.375

    # RH reads the record with sscanf: numbers may abut, and a predicted level's negative energy counts by its magnitude
    abutting = replace(fe, " -0.420" => "-20.000", "  53393.669" => " -53393.669")
    write(path, abutting * "\n")
    l_abut = only(load_kurucz_linelist(path).lines)
    @test l_abut.loggf == -20 && l_abut.wavelength0_m ≈ l_fe.wavelength0_m
    @test l_abut.lower_energy_j ≈ l_fe.lower_energy_j && l_abut.upper_energy_j ≈ l_fe.upper_energy_j
    @test l_abut.lower_g == l_fe.lower_g && l_abut.upper_g == l_fe.upper_g

    # ABO tables (RH's cubic convolution, checked against RH's C to 1e-15 in response_run/mixed_region/abo_check):
    # nodes return the table, the level order does not matter, outside the table there is no value
    sp = FFNOInversion.barklem_table(0, 1)
    for i in (1, 4, 21), j in (1, 7, 18)
        node = barklem_sigma_alpha(0, 1, sp.neff1[i], sp.neff2[j])
        @test node[1] ≈ sp.cross[i, j]
        @test node[2] ≈ sp.alpha[i, j]
    end
    @test barklem_sigma_alpha(1, 0, 2.117, 1.627) == barklem_sigma_alpha(0, 1, 1.627, 2.117)
    @test barklem_sigma_alpha(0, 1, 3.5, 2.0) === nothing && barklem_sigma_alpha(0, 2, 1.6, 2.1) === nothing
    @test FFNOInversion.atom_label_orbital("CA II 3P6 3D 2DE 5/2") == 2 && FFNOInversion.atom_label_orbital("CA II 3P6 4S 2SE") == 0
    @test FFNOInversion.atom_label_orbital("H BF N=2") === nothing
    @test FFNOInversion.kurucz_label_orbital("(4F)4p y3D") == 1 && FFNOInversion.kurucz_label_orbital("p5f'[7]") === nothing

    # atom lines (FFNO, Muspel): ABO without σ/α comes from the tables. Na I D2 (3s 2S - 3p 2P3/2) gives Anstee &
    # O'Mara's (1995) σ = 407 a_0², α = 0.273; Muspel's ABO constant is RH's, and RH's Unsöld helium term is added
    function na_atom(broadening)
        """
        element: {symbol: Na, abundance: 6.24, atomic_mass: {value: 22.99, unit: u}}
        atomic_levels:
          lev1: {energy: {value: 0.0, unit: cm^-1}, g: 2, stage: 1, label: NA I 2P6 3S 2SE}
          lev2: {energy: {value: 16973.366, unit: cm^-1}, g: 4, stage: 1, label: NA I 2P6 3P 2PO 3/2}
          ion1: {energy: {value: 41449.451, unit: cm^-1}, g: 1, stage: 2, label: NA II GROUND TERM}
        radiative_bound_bound:
          - transition: [lev2, lev1]
            f_value: 0.641
            profile_type: Voigt
            broadening:
              - {type: Natural, value: 6.16e7, unit: s^-1}
              - $broadening
            wavelengths: {type: MULTI, nλ: 21, qmax: 100.0, q0: 3.0, qnorm: {value: 8.0, unit: km / s}}
        radiative_bound_free: []
        """
    end
    table_file = joinpath(dir, "na_table.yaml"); write(table_file, na_atom("{type: VanderWaals_ABO}"))
    resolved = resolve_abo_broadening(table_file)
    @test resolved != table_file
    nad = Muspel.read_atom(resolved).lines[1].γ
    neff_s = effective_n(41449.451 * FFNOInversion._HC_J_CM, 0.0, 1, 22.99)
    neff_p = effective_n(41449.451 * FFNOInversion._HC_J_CM, 16973.366 * FFNOInversion._HC_J_CM, 1, 22.99)
    σ, α = barklem_sigma_alpha(0, 1, neff_s, neff_p)
    @test σ ≈ 407 rtol=0.02
    @test α ≈ 0.273 atol=0.01
    @test nad.coeff[1] ≈ barklem_coefficient(σ, α, 22.99) rtol=1e-3
    @test nad.temp_exp[1] ≈ (1 - α) / 2
    @test length(nad.coeff) == 2 && nad.temp_exp[2] ≈ 0.3                       # Unsöld helium term
    explicit_file = joinpath(dir, "na_explicit.yaml")
    write(explicit_file, na_atom("{type: VanderWaals_ABO, σ: {value: 300.0}, α: {value: 0.25}}"))
    given = Muspel.read_atom(resolve_abo_broadening(explicit_file)).lines[1].γ
    @test given.coeff[1] ≈ barklem_coefficient(300.0, 0.25, 22.99) rtol=1e-3
    @test given.temp_exp[1] ≈ 0.375
    ion_file = joinpath(dir, "na_ion.yaml")                                     # Na I labels, but stage 2: no table
    write(ion_file, replace(na_atom("{type: VanderWaals_ABO}"), "stage: 1" => "stage: 2", "stage: 2, label: NA II" => "stage: 3, label: NA III"))
    fallback = @test_logs (:warn, r"no ABO table applies") Muspel.read_atom(resolve_abo_broadening(ion_file)).lines[1].γ
    @test length(fallback.coeff) == 1 && fallback.temp_exp[1] ≈ 0.3
    @test resolve_abo_broadening(explicit_file) != explicit_file
    no_abo = joinpath(dir, "na_unsold.yaml"); write(no_abo, na_atom("{type: VanderWaals_Unsold, h_coefficient: 1.0}"))
    @test resolve_abo_broadening(no_abo) == no_abo                              # no ABO entry: the file itself

    # the region's FFNO transition is dropped from its Kurucz lines, other lines are kept
    halpha = FFNOInversion.FFNO_KURUCZ_IDENTITY[:halpha]; ca = FFNOInversion.FFNO_KURUCZ_IDENTITY[:ca_ii_8542]
    kept = FFNOInversion._without_ffno_transitions(lines, [halpha])
    @test sort([l.atomic_number for l in kept]) == [14, 26, 26, 26]
    @test length(FFNOInversion._without_ffno_transitions(lines, [ca])) == length(lines)

    # output units: I_ν = I_λ λ²/c (vacuum λ), SI or cgs, divided by the region normalization. B_λ in the synthesis unit
    # becomes B_ν in SI.
    region(normalization) = SpectralRegionConfig(656.0e-9, 0.01e-9, 5, normalization, :none, nothing, SpectralSourceConfig[])
    air = wavelengths(region(1.0)); vacuum = FFNOInversion._air_to_vacuum_nm.(air .* 1e9) .* 1e-9
    temp = 5800.0; h_, c, k = 6.62607015e-34, 299792458.0, 1.380649e-23
    bnu = @. 2h_ * (c / vacuum)^3 / c^2 / expm1(h_ * c / (vacuum * k * temp))
    si_scale = output_intensity_scale([region(1.0)], :si)
    @test FFNOInversion.source_planck.(vacuum, temp) .* si_scale ≈ bnu rtol=1e-12
    @test output_intensity_scale([region(1.0)], :cgs) ≈ 1e3 .* si_scale
    two = output_intensity_scale([region(4.2e-5), region(2.0)], :cgs)
    @test two ≈ vcat(1e3 .* si_scale ./ 4.2e-5, 1e3 .* si_scale ./ 2.0)

    # configuration: intensity_units defaults to si, accepts cgs, rejects anything else
    text = read(joinpath(@__DIR__, "..", "configs", "example_intensity_nonprd.toml"), String)
    @test load_config(joinpath(@__DIR__, "..", "configs", "example_intensity_nonprd.toml")).outputs.intensity_units === :si
    for (units, ok) in (("cgs", true), ("kw", false))
        p = joinpath(dir, "c_$units.toml")
        write(p, replace(text, r"\[outputs\]\n" => "[outputs]\nintensity_units = \"$units\"\n"))
        if ok
            @test load_config(p).outputs.intensity_units === :cgs
            @test occursin("intensity_units=cgs", dry_run_summary(load_config(p)))
        else
            @test_throws ArgumentError load_config(p)
        end
    end
end

# The observation file is in the output's unit: si or cgs divided by the region normalization. A written synthesis read
# back as an observation is that synthesis, and chi-square does not depend on the unit.
@testset "Observations are read in the output unit" begin
    mktempdir() do directory
        atmosphere_path=joinpath(directory,"initial.h5"); observation_path=joinpath(directory,"observation.h5")
        synthesis_path=joinpath(directory,"synthesis.h5"); atmosphere_out=joinpath(directory,"atmosphere.h5")
        logtau=[-5.0,-3.0,-1.0]; nx,ny=2,3; shape=(3,nx,ny)
        h5open(atmosphere_path,"w") do file
            file["logtau_500"]=logtau
            for (name,value) in (("temperature",fill(5500.0,shape)),("vx",zeros(shape)),("vy",zeros(shape)),("vz",zeros(shape)))
                file[name]=FFNOInversion._with_time_zyx(value)
            end
        end
        write_config(path,units)=write(path,"""
[inputs]
observation_file = "$observation_path"
initial_atmosphere_file = "$atmosphere_path"
time_index = 1
[outputs]
synthesis_file = "$synthesis_path"
atmosphere_file = "$atmosphere_out"
intensity_units = "$units"
[equilibrium]
backend = "HSEMHS"
model = "column_hse"
eos_table_file = "eos_table.h5"
[atmosphere]
storage_order = "julia"
[atmosphere.datasets]
logtau500 = "logtau_500"
temperature = "temperature"
vx = "vx"
vy = "vy"
vz = "vz"
[grid]
dx_m = 40000.0
dy_m = 40000.0
[[regions]]
start_angstrom = 6562.4
step_angstrom = 0.1
count = 3
normalization = 4.3e-5
psf_type = "none"
[[regions]]
start_angstrom = 8541.0
step_angstrom = 0.2
count = 2
normalization = 2.0
psf_type = "none"
[physics]
redistribution = "non_prd"
[observation]
stokes = ["I"]
[observation.datasets]
intensity = "intensity"
sigma = "sigma"
wavelength_weights = "wavelength_weights"
spatial_weights = "spatial_weights"
[regularization]
[regularization.vertical]
types = [0,0,0,0,0,0,0]
regularize = 0.0
weights = [1,1,1,1,1,1,1]
[inversion]
[[inversion.controls]]
variable = "temperature"
log_tau_nodes = [-5.0,-1.0]
control_nx = 1
control_ny = 1
lower = 3000.0
upper = 8000.0
scale = 1000.0
[solver]
method = "lbfgs"
max_iterations = 1
history_length = 3
checkpoint_path = ""
[parallel]
enabled = false
threads_per_rank = 1
""")
        for units in ("cgs","si")
            path=joinpath(directory,"run_$units.toml"); write_config(path,units)
            config=load_config(path)
            nλ=length(config.synthesis.wavelength_m)
            # a blackbody in the synthesis unit is written as B_ν in the configured unit over the normalization
            temperature=[5000.0+100i+10j for _ in 1:1,i in 1:nx,j in 1:ny]
            vacuum=vcat([FFNOInversion._air_to_vacuum_nm.(wavelengths(r).*1e9).*1e-9 for r in config.regions]...)
            intrinsic=[FFNOInversion.source_planck(vacuum[l],temperature[1,i,j]) for l in 1:nλ,_ in 1:1,i in 1:nx,j in 1:ny]
            h_,c,k=6.62607015e-34,299792458.0,1.380649e-23
            factor=units=="cgs" ? 1e3 : 1.0
            normalization=vcat(fill(4.3e-5,3),fill(2.0,2))
            expected=[factor*2h_*(c/vacuum[l])^3/c^2/expm1(h_*c/(vacuum[l]*k*temperature[1,i,j]))/normalization[l]
                      for l in 1:nλ,_ in 1:1,i in 1:nx,j in 1:ny]
            atmosphere=Atmosphere3D(Grid3D(logtau,[0.0,40e3],[0.0,40e3,80e3]),fill(5500.0,shape),zeros(shape),zeros(shape),zeros(shape),zeros(shape))
            result=InversionRunResult(nothing,SpectralCube(intrinsic,config.synthesis.wavelength_m,StokesSet(:I)),atmosphere,nothing,nothing,Dict{String,Any}())
            write_inversion_outputs(config,result)
            written=h5read(synthesis_path,"intensity")
            @test written≈FFNOInversion._with_time_slyx(expected) rtol=1e-12
            @test h5open(f->read(attributes(f["intensity"])["units"]),synthesis_path)==first(FFNOInversion.INTENSITY_UNITS[Symbol(units)])
            # that file is a valid observation: it reads back as the synthesis, with sigma in the same unit
            sigma_file=fill(0.01,size(written))
            h5open(observation_path,"w") do file
                file["intensity"]=written; file["sigma"]=sigma_file
                file["wavelength_weights"]=ones(nλ,1); file["spatial_weights"]=ones(ny,nx)
            end
            observation=read_inversion_inputs(config).observation
            @test observation.spectrum.data≈intrinsic rtol=1e-12
            # chi-square of a 1% error is the same as computed in the file's unit
            model=1.01.*intrinsic; sigma_in_file=0.01.*ones(size(intrinsic))
            chi_file=sum(abs2,(model.-intrinsic).*output_intensity_scale(config.regions,config.outputs.intensity_units)./sigma_in_file)
            chi_read=sum(abs2,(model.-observation.spectrum.data)./observation.sigma)
            @test chi_read≈chi_file rtol=1e-10
            # overlapping regions must share one normalization (given twice); separate regions need not
            text=read(path,String); other=joinpath(directory,"overlap_$units.toml")
            write(other,replace(text,"start_angstrom = 8541.0"=>"start_angstrom = 6562.5"))
            @test_throws ArgumentError load_config(other)
            write(other,replace(text,"start_angstrom = 8541.0"=>"start_angstrom = 6562.5","normalization = 2.0"=>"normalization = 4.3e-5"))
            @test length(load_config(other).regions)==2
            write(other,replace(text,"start_angstrom = 8541.0"=>"start_angstrom = 6563.0"))
            @test length(load_config(other).regions)==2
        end
    end
end

# FFNO lines (Muspel): the Doppler width includes microturbulence, Δλ_D = λ0/c sqrt(2kT/m + vturb²), as in RH and the
# Kurucz path. The damping parameter uses the same width, vturb = 0 is the thermal width exactly, and the adjoint holds.
@testset "Microturbulence in the FFNO line opacity" begin
    T=Float32; nz,nx,ny=3,1,1; shape=(nz,nx,ny)
    grid=Grid3D(T[-3,-2,-1],T[0],T[0])
    state(vturb)=FFNOInversion.Atmosphere3D(grid,reshape(T[5000,5500,6000],shape),zeros(T,shape),zeros(T,shape),zeros(T,shape),
        fill(T(vturb),shape);pgas=fill(T(100),shape),rho=fill(T(1e-7),shape),ne=reshape(T[5e16,1e17,2e17],shape),
        z=reshape(T[2e5,1e5,0],shape))
    hp=fill(T(1e10),nz,nx,ny,6); hp[:,:,:,1].=T(1e20); hp[:,:,:,2].=T(1e14); hp[:,:,:,3].=T(1e10); hp[:,:,:,6].=T(1e15)
    atom_dir=get(ENV,"FFNO_ATOM_DIR","/mn/stornext/u3/harshm/Documents/WorkRepo/multi3d/input/atoms")
    voigt=(a_min=1f-4,a_max=1f1,a_n=300,v_min=0f0,v_max=5f2,v_n=500)
    model=build_muspel_line_model(state(0),hp,joinpath(atom_dir,"atom.h6_tiago2.yaml"),5,2,3;voigt,include_continuum=false)
    line=model.line
    wave=T.(line.λ0 .+ [-0.2,-0.05,0.0,0.03,0.1]).*T(1e-9)
    function opacity(a)
        chi=zeros(T,nz,length(wave)); eta=zeros(T,nz,length(wave))
        add_opacity_emissivity!(chi,eta,model,wave,a,1,1,hp)
        chi,eta
    end
    vt=T(8e3); χ,η=opacity(state(vt))
    for k in 1:nz,l in eachindex(wave)
        temp=T(state(0).temperature[k,1,1]); ne=T(state(0).ne[k,1,1]); hi=T(sum(hp[k,1,1,1:5]))
        thermal=Muspel.doppler_width(line.λ0,line.mass,temp)
        width=hypot(thermal,T(line.λ0*vt/299792458.0))
        gamma=Muspel.calc_broadening(line.γ,temp,ne,hi); λ=wave[l]*1e9
        a=Muspel.damping(gamma,λ,width); v=(λ-line.λ0)/width
        factor=6.62607015e-34*299792458.0/(4pi*line.λ0*1e-9)*real(model.voigt(a,abs(v)))/(sqrt(pi)*width)
        @test χ[k,l]≈factor*(hp[k,1,1,2]*line.Blu-hp[k,1,1,3]*line.Bul)*1e9 rtol=1e-3
    end
    # vturb broadens the line: the centre drops; vturb = 0 is unchanged by repetition
    χ0,_=opacity(state(0))
    @test all(χ[:,3].<χ0[:,3])
    @test opacity(state(0))[1]==χ0
    @test χ0≠χ
    # the adjoint's vturb term against a centred difference of the same pairing
    chi_bar=T.(reshape(sin.(1:nz*length(wave)),nz,:)); eta_bar=T.(reshape(cos.(1:nz*length(wave)),nz,:))
    function pairing(v)
        c,e=opacity(state(v))
        sum(Float64.(chi_bar).*Float64.(c))+sum(Float64.(eta_bar).*Float64.(e))
    end
    a=state(vt); bar=FFNOInversion.AtmosphereCotangent(a); population_bar=zeros(T,size(hp))
    FFNOInversion.add_opacity_emissivity_vjp!(bar,population_bar,_->zeros(T,size(hp)),chi_bar,eta_bar,model,wave,a,1,1,hp)
    h=T(50); finite=(pairing(vt+h)-pairing(vt-h))/(2h)
    @test sum(bar.vturb)≈finite rtol=0.02
    @test abs(finite)>0
end
