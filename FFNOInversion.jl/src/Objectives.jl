struct ResidualLayout
    stokes::StokesSet
end

function residual!(r::AbstractVector{T}, layout::ResidualLayout, synthetic::SpectralCube{T}, observation::ObservationCube{T}) where T
    layout.stokes.components == synthetic.stokes.components == observation.spectrum.stokes.components || throw(DimensionMismatch("residual Stokes layout mismatch"))
    length(r) == length(observation.spectrum.data) || throw(DimensionMismatch("residual vector length mismatch"))
    for i in eachindex(synthetic.data)
        r[i] = observation.inversion_weights[i] * (synthetic.data[i] - observation.spectrum.data[i]) / observation.sigma[i]
    end
    r
end
