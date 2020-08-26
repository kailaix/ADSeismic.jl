module ADSeismic

    using ADCME
    using ADCMEKit
    using Optim
    using PyPlot
    using PyCall
    using SparseArrays
    using LinearAlgebra
    using Statistics
    using Parameters
    using DelimitedFiles
    using MAT
    using Printf

    np = PyNULL()
    function __init__()
        copy!(np, pyimport("numpy"))
    end

    include("Struct.jl")
    include("Core.jl")
    include("IO.jl")
    include("NN.jl")
    include("Optim.jl")
    include("Utils.jl")
    include("ThirdParty.jl")

end