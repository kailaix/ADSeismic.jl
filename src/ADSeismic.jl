module ADSeismic

    using ADCME
    using PyPlot
    using PyCall
    using SparseArrays
    using LinearAlgebra
    using DelimitedFiles
    using Statistics
    using Parameters
    using MAT
    using ProgressMeter
    using Optim
    using Printf
    using Conda


    np = PyNULL()

    add_source = nothing
    get_receive = nothing
    acoustic_wave_op = nothing
    function __init__()
        global add_source, get_receive, acoustic_wave_op
        copy!(np, pyimport("numpy"))
        # load custom operators
        add_source = load_op_and_grad("$(@__DIR__)/../deps/CustomOps/SourceOps/build/libAddSource","add_source",
            multiple=true)
        get_receive = load_op_and_grad("$(@__DIR__)/../deps/CustomOps/ReceiveOps/build/libGetReceive", "get_receive")
        acoustic_wave_op = load_op_and_grad("$(@__DIR__)/../deps/CustomOps/AcousticWaveSource/build/libAcousticSource", "acoustic_source")
    end

    include("Struct.jl")
    include("Core.jl")
    include("Utils.jl")
    include("Io.jl")
    include("Optim.jl")
    
end