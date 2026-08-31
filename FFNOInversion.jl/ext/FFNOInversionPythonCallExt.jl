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

function FFNOInversion.population_vjp!(feature_bar::AbstractArray{T,4},z_bar::AbstractArray{T,3},
        model::FFNOInversion.PythonFFNOModel{Py},atmosphere::FFNOInversion.Atmosphere3D,
        population_bar::AbstractArray{T,4}) where T
    size(feature_bar)==(6,size(atmosphere.temperature)...) || throw(DimensionMismatch(
        "FFNO feature cotangent shape differs from atmosphere"))
    size(z_bar)==size(atmosphere.temperature) || throw(DimensionMismatch(
        "FFNO z cotangent shape differs from atmosphere"))
    size(population_bar)[1:3]==size(atmosphere.temperature) || throw(DimensionMismatch(
        "FFNO population cotangent grid differs from atmosphere"))
    size(population_bar,4)==length(model.metadata.level_names) || throw(DimensionMismatch(
        "FFNO population cotangent level count differs from checkpoint metadata"))
    features=FFNOInversion.population_features(atmosphere); z=Float32.(atmosphere.z)
    dx=FFNOInversion._spacing(atmosphere.grid.x,:x); dy=FFNOInversion._spacing(atmosphere.grid.y,:y)
    result=pyconvert(Dict,model.backend.vjp(features,z,dx,dy,Float32.(population_bar)))
    feature_gradient=result["features"]; z_gradient=result["z_scale"]
    size(feature_gradient)==size(feature_bar) || throw(DimensionMismatch(
        "Python FFNO VJP returned an invalid feature-gradient shape"))
    size(z_gradient)==size(z_bar) || throw(DimensionMismatch(
        "Python FFNO VJP returned an invalid z-gradient shape"))
    feature_bar.=T.(feature_gradient); z_bar.=T.(z_gradient)
    all(isfinite,feature_bar)&&all(isfinite,z_bar) || throw(ErrorException(
        "Python FFNO VJP returned NaN or Inf"))
    model.calls+=1
    feature_bar,z_bar
end

end
