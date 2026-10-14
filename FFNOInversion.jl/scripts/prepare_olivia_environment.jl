# Recreate a disposable Olivia project from the permanent package checkout.
using TOML

function prepare_olivia_environment(package_dir, project_dir; muspel_dir="")
    package_dir = realpath(package_dir)
    mkpath(project_dir)
    project_dir = realpath(project_dir)
    package_dir == project_dir && error("Use a separate OLIVIA_JULIA_PROJECT; the repository is the permanent source")
    project = TOML.parsefile(joinpath(package_dir, "Project.toml"))
    manifest_path = joinpath(package_dir, "Manifest.toml")
    manifest = isfile(manifest_path) ? TOML.parsefile(manifest_path) : nothing
    if !isempty(muspel_dir)
        muspel_dir = realpath(muspel_dir)
        isfile(joinpath(muspel_dir, "Project.toml")) || error("Missing Muspel Project.toml")
        revision = project["sources"]["Muspel"]["rev"]
        strip(read(`git -C $muspel_dir rev-parse HEAD`, String)) == revision ||
            error("Local Muspel checkout must be at $revision")
        project["sources"]["Muspel"] = Dict("path" => muspel_dir)
        if manifest !== nothing
            for entry in manifest["deps"]["Muspel"]
                for key in ("git-tree-sha1", "repo-rev", "repo-url")
                    pop!(entry, key, nothing)
                end
                entry["path"] = muspel_dir
            end
        end
    end
    # Check links before replacing project metadata; never overwrite user directories.
    for name in ("src", "ext", "scripts")
        source = joinpath(package_dir, name)
        isdir(source) || continue
        target = joinpath(project_dir, name)
        if ispath(target) || islink(target)
            islink(target) && realpath(target) == source ||
                error("Existing $target does not link to $source; choose a new project directory")
        else
            symlink(source, target)
        end
    end
    open(joinpath(project_dir, "Project.toml"), "w") do io
        TOML.print(io, project; sorted=true)
    end
    target_manifest = joinpath(project_dir, "Manifest.toml")
    if manifest === nothing
        # Resolve afresh rather than accidentally reusing a stale run-local manifest.
        isfile(target_manifest) && rm(target_manifest)
        println("Repository has no Manifest.toml; setup will resolve from Project.toml")
    else
        open(target_manifest, "w") do io
            TOML.print(io, manifest; sorted=true)
        end
    end
    println("Prepared disposable Julia project: $project_dir")
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || error("usage: prepare_olivia_environment.jl PACKAGE_DIR PROJECT_DIR")
    prepare_olivia_environment(ARGS...; muspel_dir=get(ENV, "OLIVIA_LOCAL_MUSPEL_DIR", ""))
end
