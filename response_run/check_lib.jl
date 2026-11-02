"""Atmosphere (nz, nx, ny) and populations (nz, nx, ny, level) as written by write_inversion_outputs."""
function read_forward_atmosphere(path)
    h5open(path,"r") do file
        field(name)=haskey(file,name) ? permutedims(read(file[name])[1,:,:,:],(1,3,2)) : nothing
        grid=Grid3D(VerticalAxis(:log10tau500,read(file["logtau_500"])),read(file["x_m"]),read(file["y_m"]))
        atmosphere=Atmosphere3D(grid,field("temperature"),field("vx"),field("vy"),field("vz"),field("vturb");
            pgas=field("pgas"),rho=field("rho"),ne=field("ne"),z=field("z"),logtau500=field("logtau500"),
            chi500=field("chi500"))
        populations=Dict(Symbol(name)=>permutedims(read(file["populations"][name]),(2,4,3,1))
            for name in keys(file["populations"]))
        atmosphere,populations
    end
end

"""Width of each log tau500 cell (trapezoid weights), so that Σ_k R_k Δ_k δv_k is the intensity change."""
function cell_widths(logtau)
    n=length(logtau); d=zeros(n)
    d[1]=(logtau[2]-logtau[1])/2; d[n]=(logtau[n]-logtau[n-1])/2
    for k in 2:n-1; d[k]=(logtau[k+1]-logtau[k-1])/2; end
    d
end

"""Step and chain factor per depth: R is ∂I/∂v for T, vz, vturb and ∂I/∂ln v for ne and pgas. vz and vturb use a
fixed step because Muspel works in Float32, where a m/s Doppler shift is below the wavelength resolution."""
function perturbation(variable,value)
    variable===:temperature && return 1e-3*value,1.0
    variable in (:vz,:vturb) && return 250.0,1.0
    1e-3*value,value
end

function opacity!(chi,eta,synthesizer,wavelength,atmosphere,x,y)
    fill!(chi,0); fill!(eta,0)
    for contributor in synthesizer.contributors
        F.add_opacity_emissivity!(chi,eta,contributor.model,wavelength,atmosphere,x,y,contributor.populations)
    end
end

