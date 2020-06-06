export ElasticPropagatorParams, ElasticPropagator, ElasticSource, ElasticReceiver,
       AcousticPropagatorParams, AcousticPropagator, AcousticSource, AcousticReceiver

@with_kw mutable struct ElasticPropagatorParams
    # number of grids along x,y axis and time steps
    NX::Int64 = 101
    NY::Int64 = 641 
    NSTEP::Int64 = 2000 * 2 

    # size of grid cell and time step
    DELTAX::Float64 = 10.
    DELTAY::Float64 = 10.
    DELTAT::Float64 = 2.e-3 / 2

    # PML boundary conditon 
    f0::Float64 = 5. # source frequency
    vp_ref::Float64 = 2000. 
    USE_PML_XMIN::Bool = true
    USE_PML_XMAX::Bool = true
    USE_PML_YMIN::Bool = true
    USE_PML_YMAX::Bool = true
    NPOINTS_PML::Int64 = 12
    NPOWER::Float64 = 2.
    K_MAX_PML::Float64 = 1.
    ALPHA_MAX_PML::Float64 = 2. * π * ( f0/2.) 
    Rcoef::Float64 = 0.001 # Relative reflection coefficient
    
    # display params
    IT_DISPLAY::Int64 = 0
end

mutable struct ElasticSource
    srci::Union{Array{Int64, 1},PyObject}
    srcj::Union{Array{Int64, 1},PyObject}
    srctype::Union{Array{Int64, 1},PyObject}
    srcv::Union{Array{Float64, 2},PyObject}
end

mutable struct ElasticReceiver
    rcvi::Union{Array{Int64, 1},PyObject}
    rcvj::Union{Array{Int64, 1},PyObject}
    rcvtype::Union{Array{Int64, 1},PyObject}
    rcvv::Union{Array{Float64, 2},PyObject,Missing}
end

mutable struct ElasticPropagator
    param::ElasticPropagatorParams
    src::ElasticSource
    vx::PyObject
    vy::PyObject
    sigmaxx::PyObject
    sigmayy::PyObject
    sigmaxy::PyObject
    mem::PyObject
end

@with_kw mutable struct AcousticPropagatorParams
    # number of grids along x,y axis and time steps
    NX::Int64 = 101
    NY::Int64 = 641 
    NSTEP::Int64 = 2000 * 2 

    # size of grid cell and time step
    DELTAX::Float64 = 10.
    DELTAY::Float64 = 10.
    DELTAT::Float64 = 2.e-3 / 2

    # PML boundary conditon 
    USE_PML_XMIN::Bool = true
    USE_PML_XMAX::Bool = true
    USE_PML_YMIN::Bool = true
    USE_PML_YMAX::Bool = true
    NPOINTS_PML::Int64 = 12
    NPOWER::Int64 = 2
    damping_x::Union{Missing,Float64} = missing
    damping_y::Union{Missing,Float64} = missing
    Rcoef::Float64 = 0.001 # Relative reflection coefficient
    vp_ref::Float64 = 2000. 

    # Auxilliary Data
    Σx::Array{Float64} = []
    Σy::Array{Float64} = []
    IJ::Array{Int64} = []
    IJn::Array{Int64}  = []
    IJp::Array{Int64} = []
    IpJ::Array{Int64} = []
    IpJp::Array{Int64} = []
    IpJn::Array{Int64} = []
    InJ::Array{Int64} = []
    InJn::Array{Int64} = []
    InJp::Array{Int64} = []
    
    # display params
    IT_DISPLAY::Int64 = 0
end

mutable struct AcousticSource
    srci::Union{Array{Int32, 1},Array{Int64, 1}, PyObject}
    srcj::Union{Array{Int32, 1},Array{Int64, 1}, PyObject}
    srcv::Union{Array{Float64, 2},PyObject}
end

mutable struct AcousticReceiver
    rcvi::Union{Array{Int64, 1},PyObject}
    rcvj::Union{Array{Int64, 1},PyObject}
    rcvv::Union{Array{Float64, 2},PyObject,Missing}
end

"""
    AcousticPropagator

A structure holding all the information for acoustic simulation
"""
mutable struct AcousticPropagator
    param::AcousticPropagatorParams
    src::AcousticSource
    u::PyObject
    # auxillary variables for PML
    φ::PyObject
    ψ::PyObject
end
