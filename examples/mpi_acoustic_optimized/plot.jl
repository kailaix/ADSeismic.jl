using PyPlot 
using ADSeismic
using ADCME
using JLD2

mpi_init()

M = N = 5
n = 100
@info M 
param = MPIAcousticPropagatorParams(NX = n*M, NY = n*N, n = n,
     Rcoef=0.2, DELTAX=10, DELTAY=10, DELTAT=0.05, NSTEP = 2000) 
compute_PML_Params!(param; check_mpi_size = false)

Vals = Array{Float64, 3}[]
for i = 1:M 
    for j = 1:N 
        r = (i-1)*N + j - 1
        @load "data/dat$(r)-$(M*N).jld2" u
        push!(Vals, u)
    end
end

p = visualize_wavefield(Vals, param)
saveanim(p, "wavefield.gif")