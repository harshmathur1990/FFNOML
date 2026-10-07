using Test
using TOML
include(joinpath(@__DIR__, "..", "scripts", "prepare_olivia_environment.jl"))

@testset "Olivia thread topology has one source of truth" begin
    batch = read(joinpath(@__DIR__, "..", "scripts", "run_olivia_inversion.sbatch"),String)
    topology = Dict{String,Int}()
    for line in eachline(IOBuffer(batch))
        matched = match(r"^#SBATCH --(nodes|ntasks|ntasks-per-node)=(\d+)$",line)
        isnothing(matched) || (topology[matched.captures[1]] = parse(Int,matched.captures[2]))
    end
    @test topology["nodes"] * topology["ntasks-per-node"] == topology["ntasks"]
    @test occursin("FFNO_THREADS_PER_RANK=\"\${SLURM_CPUS_PER_TASK}\"",batch)
    @test occursin("--threads=\"\${SLURM_CPUS_PER_TASK}\"",batch)
end

@testset "Login preflight and accelerator runtime are isolated" begin
    scripts=joinpath(@__DIR__,"..","scripts")
    submission=read(joinpath(scripts,"submit_olivia_inversion.sh"),String)
    preflight=read(joinpath(scripts,"run_olivia_login_preflight.sh"),String)
    runtime=read(joinpath(scripts,"run_olivia_inversion.sbatch"),String)
    initializer=read(joinpath(scripts,"initialize_olivia_environment.sh"),String)
    guide=read(joinpath(scripts,"..","..","examples","inversion_run","START_HERE.md"),String)
    readme=read(joinpath(scripts,"..","..","examples","inversion_run","README.md"),String)
    accelerator_environment=read(joinpath(scripts,"..","..","examples","inversion_run","olivia_runtime_environment.sh"),String)
    @test occursin("module load \"\${FFNO_PREFLIGHT_STACK_MODULE:-NRIS/Login}\"",preflight)
    @test occursin("julia-depot-ffno-login-1.12.2",preflight)
    @test occursin("must be one absolute directory",preflight)
    @test occursin("Pkg.instantiate(; allow_autoprecomp=false)",preflight)
    @test first(findfirst("Pkg.instantiate",preflight)) <
        first(findfirst("exec \"\${julia_executable}\"",preflight))
    @test occursin("unset OLIVIA_ENV_SCRIPT OLIVIA_JULIA OLIVIA_PYTHON",preflight)
    @test !occursin("OLIVIA_JULIA",submission)
    @test !occursin("OLIVIA_PYTHON",submission)
    @test occursin("\"\${preflight_runner}\"",submission)
    @test occursin("scripts/witt_eos_cpp.cpp",submission)
    @test occursin("source \"\${runtime_environment}\"",runtime)
    @test occursin("\"\${initializer}\"",runtime)
    @test occursin("build_wittmann_backend.jl",initializer)
    @test occursin("Wittmann EOS load check OK",initializer)
    @test occursin("inputs/wittmann/libwitt_ffno.so",initializer)
    @test occursin("export OLIVIA_CXX=c++",accelerator_environment)
    @test first(findfirst("\"\${initializer}\"",runtime)) <
        first(findfirst("srun --kill-on-bad-exit",runtime))
    @test first(findfirst("\"\${initializer}\"",runtime)) <
        first(findfirst("mkdir -p \"\${FFNOML_RUN_DIR}/outputs\"",runtime))
    @test !occursin("Run setup_olivia_environment.sbatch",submission)
    @test !occursin("runtime_project",submission)
    @test !occursin("setup_olivia_environment.sbatch",guide)
    @test !occursin("c++ -O3",guide)
    @test !occursin("/permanent/lib/libwitt_ffno.so",readme)
end

@testset "Disposable Olivia environment" begin
    mktempdir() do root
        package = joinpath(root, "permanent-package")
        muspel = joinpath(root, "permanent-muspel")
        project = joinpath(root, "work", "julia-environment")
        mkpath(muspel)
        write(joinpath(muspel, "Project.toml"), "name = \"Muspel\"\n")
        run(pipeline(`git init -q $muspel`; stdout=devnull, stderr=devnull))
        run(`git -C $muspel add Project.toml`)
        run(`git -C $muspel -c user.name=Test -c user.email=test@example.invalid -c commit.gpgsign=false commit -qm fixture`)
        revision = strip(read(`git -C $muspel rev-parse HEAD`, String))
        for name in ("src", "ext", "scripts")
            mkpath(joinpath(package, name))
        end
        write(joinpath(package, "Project.toml"), """
        name = "Fixture"
        [sources]
        Muspel = {url = "https://example.invalid/Muspel.jl", rev = "$revision"}
        """)
        write(joinpath(package, "Manifest.toml"), """
        manifest_format = "2.0"
        [[deps.Muspel]]
        repo-url = "https://example.invalid/Muspel.jl"
        repo-rev = "$revision"
        git-tree-sha1 = "fixture"
        """)
        original = read(joinpath(package, "Manifest.toml"), String)
        prepare_olivia_environment(package, project; muspel_dir=muspel)
        @test TOML.parsefile(joinpath(project, "Project.toml"))["sources"]["Muspel"] == Dict("path"=>realpath(muspel))
        entry = only(TOML.parsefile(joinpath(project, "Manifest.toml"))["deps"]["Muspel"])
        @test entry == Dict("path"=>realpath(muspel))
        @test realpath(joinpath(project, "src")) == realpath(joinpath(package, "src"))
        @test read(joinpath(package, "Manifest.toml"), String) == original
        # Repeated setup and full work-directory expiration both recover.
        prepare_olivia_environment(package, project; muspel_dir=muspel)
        rm(dirname(project); recursive=true)
        prepare_olivia_environment(package, project; muspel_dir=muspel)
        @test isfile(joinpath(project, "Manifest.toml"))
        # Missing permanent manifest requests fresh resolution, not stale reuse.
        rm(joinpath(package, "Manifest.toml"))
        prepare_olivia_environment(package, project; muspel_dir=muspel)
        @test !isfile(joinpath(project, "Manifest.toml"))
        @test isfile(joinpath(project, "Project.toml"))
        @test_throws ErrorException prepare_olivia_environment(package, package)
        write(joinpath(muspel, "extra"), "changed revision")
        run(`git -C $muspel add extra`)
        run(`git -C $muspel -c user.name=Test -c user.email=test@example.invalid -c commit.gpgsign=false commit -qm changed`)
        @test_throws ErrorException prepare_olivia_environment(package, project; muspel_dir=muspel)
    end
end
