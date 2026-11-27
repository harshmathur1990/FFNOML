using Test
using FFNOInversion
using Libdl

# The synthesis pullback runs its columns on all threads (it ran on one: about 7 of the 7.3 minutes of an Olivia iteration,
# job 3192771). Columns are independent, so the threaded result must equal the serial one to the bit, leave the
# atmosphere as it found it (the Wittmann pullback perturbs cells in place), and match finite differences.
@testset "Synthesis pullback runs columns in parallel, bit-identical to serial" begin
    Threads.nthreads()>1 || @warn "one thread: the threaded synthesis pullback is checked only as the serial one"
    mktempdir() do dir
        lib=joinpath(dir,"libwitt_threads.$(Libdl.dlext)")
        run(`$(Base.julia_cmd()) --project=$(joinpath(@__DIR__,"..")) $(joinpath(@__DIR__,"..","scripts","build_wittmann_backend.jl")) $lib`)
        eos=WittmannEOS(lib,joinpath(@__DIR__,"..","..","scripts","pf_Kurucz.input"))
        fe=joinpath(dir,"fe.list")
        write(fe,"  656.9214 -0.420 26.00   53393.669  4.0 (4F)4d g5F   38175.352  3.0 (4F)4p y3D  8.36 -4.38 -7.70FMW  0 0  0 0.000  0 0.000    0    0           1287 1324\n")
        lines=load_kurucz_linelist(fe).lines

        T=Float64; nz,nx,ny=8,5,4; shape=(nz,nx,ny)
        z=collect(range(8e5,0.0;length=nz))
        cell(f)=[f(k,x,y) for k in 1:nz,x in 1:nx,y in 1:ny]
        temperature=cell((k,x,y)->4400.0+250k+40x-30y); pgas=cell((k,x,y)->10.0^(1.3+0.35k)*(1+0.02x))
        atmos=FFNOInversion.Atmosphere3D(Grid3D(VerticalAxis(:z_m,z),collect(0.0:4.0).*1e5,collect(0.0:3.0).*1e5),
            temperature,cell((k,x,y)->100.0k),cell((k,x,y)->-50.0x),cell((k,x,y)->800sin(k+x+y)),cell((k,x,y)->1000.0+30k);
            pgas=pgas,rho=3e-8 .* pgas,ne=1e17 .* pgas.^0.6,z=repeat(reshape(z,nz,1,1),1,nx,ny))
        populations=Dict(:H=>[1e12*(1+0.1sin(k*x*y+l)) for k in 1:nz,x in 1:nx,y in 1:ny,l in 1:3])
        populations[:H][:,:,:,3].*=1e-4
        wavelength=collect(range(656.40e-9,656.98e-9,length=12))
        synthesizer=MixedIntensitySynthesizer([
            OpacityContributor(TabulatedOpacityModel(fill(1e-5,nz,length(wavelength)),fill(2e-5,nz,length(wavelength)))),
            OpacityContributor(FFNOTransition(:H,656.461e-9,2,3,0.64,1.6735575e-27,1e8),populations[:H]),
            OpacityContributor(KuruczLTEModel(lines,eos;include_continuum=true))])
        bar=reshape([sin(0.3i) for i in 1:length(wavelength)*nx*ny],length(wavelength),1,nx,ny)

        pull(threaded)=begin
            atmosphere_bar=FFNOInversion.AtmosphereCotangent(atmos)
            population_bar=FFNOInversion._population_bar(populations)
            FFNOInversion._mixed_synthesis_vjp!(atmosphere_bar,population_bar,synthesizer,atmos,populations,wavelength,bar;
                threaded=threaded)
            (atmosphere=atmosphere_bar,populations=population_bar)
        end
        before=deepcopy(atmos)
        serial=pull(false); threaded=pull(true)
        for name in (:temperature,:pgas,:ne,:vz,:vturb,:z)
            @test getfield(threaded.atmosphere,name)==getfield(serial.atmosphere,name)
        end
        @test threaded.populations[:H]==serial.populations[:H]
        @test any(!iszero,threaded.atmosphere.temperature) && any(!iszero,threaded.atmosphere.pgas)
        @test any(!iszero,threaded.populations[:H])
        for name in (:temperature,:pgas,:ne,:vz,:vturb,:rho,:z)
            @test getfield(atmos,name)==getfield(before,name)       # perturbed cells are restored
        end

        # the threaded pullback against central differences of the synthesis, in several columns
        objective()=begin
            cube=SpectralCube(zeros(T,length(wavelength),1,nx,ny),wavelength,StokesSet(:I))
            synthesize!(cube,synthesizer,NonPRD(),atmos,populations)
            sum(cube.data.*bar)
        end
        for (field,bar_field,k,x,y) in ((atmos.temperature,threaded.atmosphere.temperature,3,1,1),
                (atmos.temperature,threaded.atmosphere.temperature,6,5,4),(atmos.pgas,threaded.atmosphere.pgas,4,2,3),
                (atmos.vz,threaded.atmosphere.vz,5,4,2))
            c=field[k,x,y]; h=1e-5*max(abs(c),1.0)
            field[k,x,y]=c+h; plus=objective(); field[k,x,y]=c-h; minus=objective(); field[k,x,y]=c
            @test bar_field[k,x,y]≈(plus-minus)/(2h) rtol=1e-4
        end
        # the Wittmann pullback perturbs the whole column at once: the same differences as one level at a time (the
        # previous form, 2 calls per field and level, below), with 2 calls per field. For vturb, ne and vz it leaves out the
        # continuum, which reads only T and P (checked here) and cancels in their differences.
        kurucz=synthesizer.contributors[3].model
        continuum_only=KuruczLTEModel(KuruczLine{Float64}[],kurucz.eos,true); lines_only=KuruczLTEModel(kurucz.lines,kurucz.eos,false)
        continuum_chi()=(c=zeros(T,nz,length(wavelength)); e=similar(c); fill!(e,0);
            FFNOInversion._add_wittmann_kurucz!(c,e,continuum_only,wavelength,atmos,2,2); (c,e))
        unperturbed=continuum_chi()
        for name in (:vturb,:ne,:vz)
            values=getfield(atmos,name); c=copy(values[:,2,2]); values[:,2,2].=1.3 .* c .+ 7.0
            @test continuum_chi()==unperturbed
            values[:,2,2].=c
        end
        levelwise!(bar,chi_bar,eta_bar,x,y)=begin
            chi=zeros(T,1,length(wavelength)); eta=similar(chi)
            pairing(k,m)=(fill!(chi,0); fill!(eta,0); FFNOInversion._add_wittmann_kurucz!(chi,eta,m,wavelength,atmos,x,y;levels=k:k);
                sum(chi[1,l]*chi_bar[k,l]+eta[1,l]*eta_bar[k,l] for l in eachindex(wavelength)))
            for (name,b,m) in ((:temperature,bar.temperature,kurucz),(:pgas,bar.pgas,kurucz),(:vturb,bar.vturb,lines_only),
                    (:ne,bar.ne,lines_only),(:vz,bar.vz,lines_only)),k in 1:nz
                values=getfield(atmos,name); c=values[k,x,y]; h=1e-5*max(abs(c),1.0)
                values[k,x,y]=c+h; p=pairing(k,m); values[k,x,y]=c-h; q=pairing(k,m); values[k,x,y]=c; b[k,x,y]+=(p-q)/(2h)
            end
        end
        chi_bar=[cos(0.1k+0.2l) for k in 1:nz,l in eachindex(wavelength)]; eta_bar=[sin(0.3k-0.1l) for k in 1:nz,l in eachindex(wavelength)]
        for (x,y) in ((1,1),(4,3))
            column=FFNOInversion.AtmosphereCotangent(atmos); reference=FFNOInversion.AtmosphereCotangent(atmos)
            FFNOInversion._local_wittmann_vjp!(column,chi_bar,eta_bar,kurucz,wavelength,atmos,x,y)
            levelwise!(reference,chi_bar,eta_bar,x,y)
            for name in (:temperature,:pgas,:vturb,:ne,:vz)
                @test getfield(column,name)[:,x,y]≈getfield(reference,name)[:,x,y] rtol=1e-12
            end
        end
        @test getfield(atmos,:temperature)==before.temperature && getfield(atmos,:pgas)==before.pgas

        for (k,x,y,l) in ((2,3,1,2),(7,1,4,2))   # the upper level (3) is 1e-4 of these: its effect is round-off
            p=populations[:H]; c=p[k,x,y,l]; h=1e-5*c
            p[k,x,y,l]=c+h; plus=objective(); p[k,x,y,l]=c-h; minus=objective(); p[k,x,y,l]=c
            @test threaded.populations[:H][k,x,y,l]≈(plus-minus)/(2h) rtol=1e-4
        end
    end

    # a failing column fails the threaded pullback with its own exception, after every task has stopped
    tasks=[Threads.@spawn(throw(ArgumentError("column failed"))),Threads.@spawn(sleep(0.2))]
    @test_throws ArgumentError FFNOInversion._wait_all_rethrow(tasks)
    @test all(istaskdone,tasks)
end
