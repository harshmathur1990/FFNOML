"""Build the repository's Wittmann EOS plus STiC continuum-opacity backend."""
function main(args)
    length(args)<=1 || error("usage: julia scripts/build_wittmann_backend.jl [output-library]")
    package_root=normpath(joinpath(@__DIR__,"..")); repo_root=normpath(joinpath(package_root,".."))
    source=joinpath(repo_root,"scripts","witt_eos_cpp.cpp")
    stic_root=abspath(get(ENV,"FFNO_STIC_ROOT",joinpath(repo_root,"..","stic")))
    cop=joinpath(stic_root,"src","cop.cc")
    isfile(source) || error("missing Wittmann source: $source")
    isfile(cop) || error("missing STiC continuum source: $cop (set FFNO_STIC_ROOT to the STiC checkout)")
    suffix=Sys.isapple() ? ".dylib" : Sys.iswindows() ? ".dll" : ".so"
    output=isempty(args) ? joinpath(package_root,"deps","libwitt_ffno"*suffix) : abspath(args[1])
    mkpath(dirname(output))
    compiler=get(ENV,"CXX","c++")
    run(`$compiler -O3 -std=c++17 -shared -fPIC -pthread $source $cop -o $output`)
    println(output)
end
main(ARGS)
