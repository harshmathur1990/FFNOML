"""Build the Wittmann EOS backend; continuum opacity is implemented in Julia."""
function main(args)
    length(args)<=1 || error("usage: julia scripts/build_wittmann_backend.jl [output-library]")
    package_root=normpath(joinpath(@__DIR__,"..")); repo_root=normpath(joinpath(package_root,".."))
    source=joinpath(repo_root,"scripts","witt_eos_cpp.cpp")
    isfile(source) || error("missing Wittmann source: $source")
    suffix=Sys.isapple() ? ".dylib" : Sys.iswindows() ? ".dll" : ".so"
    output=isempty(args) ? joinpath(package_root,"deps","libwitt_ffno"*suffix) : abspath(args[1])
    mkpath(dirname(output))
    compiler=get(ENV,"CXX","c++")
    run(`$compiler -O3 -std=c++17 -shared -fPIC -pthread $source -o $output`)
    println(output)
end
main(ARGS)
