using Test
using TOML
include(joinpath(@__DIR__, "..", "scripts", "prepare_olivia_environment.jl"))

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
