# Build the Wittmann EOS library (WittBifrostEOS's libwitt: the C++ EOS with HSEMHS's shim and the synthesis entry points) at the given
# path, by default deps/libwitt_ffno.<ext>; continuum opacity is implemented in Julia. Run it on the node type that will load the library.
if Base.find_package("WittBifrostEOS")!==nothing
    import WittBifrostEOS
else
    import FFNOInversion; const WittBifrostEOS=FFNOInversion.WittBifrostEOS    # an environment with FFNOInversion only (an Olivia run's)
end
function main(args)
    length(args)<=1 || error("usage: julia scripts/build_wittmann_backend.jl [output-library]")
    package_root=normpath(joinpath(@__DIR__,".."))
    suffix=Sys.isapple() ? ".dylib" : Sys.iswindows() ? ".dll" : ".so"
    output=isempty(args) ? joinpath(package_root,"deps","libwitt_ffno"*suffix) : abspath(args[1])
    mkpath(dirname(output))
    built=WittBifrostEOS.build_libwitt(mktempdir(dirname(output)))
    mv(built,output;force=true); rm(dirname(built);recursive=true)
    println(output)
end
main(ARGS)
