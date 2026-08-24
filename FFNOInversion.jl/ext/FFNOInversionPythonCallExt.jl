module FFNOInversionPythonCallExt

using FFNOInversion
using PythonCall

function FFNOInversion.load_python_ffno_model(module_name::AbstractString,factory_name::AbstractString,
                                              metadata::FFNOInversion.PopulationMetadata;
                                              python_path::Union{Nothing,AbstractString}=nothing,
                                              factory_kwargs=Dict{String,Any}())
    if python_path!==nothing
        sys=pyimport("sys"); sys.path.insert(0,abspath(python_path))
    end
    module_object=pyimport(module_name); factory=pygetattr(module_object,factory_name)
    backend=factory(;Dict(Symbol(k)=>v for (k,v) in factory_kwargs)...)
    if pyhasattr(backend,"describe")
        description=pyconvert(Dict,backend.describe())
        Tuple(Symbol.(description["input_channels"]))==metadata.input_channels || throw(ArgumentError("Python checkpoint input-channel metadata differs from Julia metadata"))
        Tuple(String.(description["level_names"]))==metadata.level_names || throw(ArgumentError("Python checkpoint level map differs from Julia metadata"))
        String(description["checkpoint_hash"])==metadata.checkpoint_hash || throw(ArgumentError("Python checkpoint hash differs from Julia metadata"))
        Symbol(description["output_representation"])==metadata.output_representation || throw(ArgumentError("Python output representation differs from Julia metadata"))
    end
    FFNOInversion.PythonFFNOModel(backend,metadata,0)
end

function FFNOInversion.predict_populations!(out::AbstractArray{T,4},model::FFNOInversion.PythonFFNOModel{Py},
                                            atmosphere::FFNOInversion.Atmosphere3D,cache=nothing) where T
    size(out,4)==length(model.metadata.level_names) || throw(DimensionMismatch("population level count differs from checkpoint metadata"))
    size(out)[1:3]==size(atmosphere.temperature) || throw(DimensionMismatch("population shape differs from atmosphere"))
    features=FFNOInversion.population_features(atmosphere); z=Float32.(atmosphere.z)
    dx=FFNOInversion._spacing(atmosphere.grid.x,:x); dy=FFNOInversion._spacing(atmosphere.grid.y,:y)
    predicted=pyconvert(Array,model.backend.predict(features,z,dx,dy))
    size(predicted)==size(out) || throw(DimensionMismatch("Python FFNO returned $(size(predicted)); expected $(size(out)) canonical (nz,nx,ny,nlevels)"))
    all(isfinite,predicted)&&all(>(0),predicted) || throw(ErrorException("Python FFNO returned non-positive or non-finite populations"))
    out.=T.(predicted); model.calls+=1; out
end

end
