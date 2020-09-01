using Revise
using ADSeismic
using ADCME 
using JLD2


param = ElasticPropagatorParams(NX=100, NY=100, NSTEP=1000, DELTAT=1e-4, 
    DELTAX=1.0, DELTAY=1.0)

idx1 = (1:50) 
idx2 = (51:100) 
V = zeros(param.NSTEP+1, param.NX+2, param.NY+2)
@load "data/dat0-4.jld2" Vx 
V[:, idx1, idx1] = Vx 
@load "data/dat1-4.jld2" Vx 
V[:, idx1, idx2] = Vx 
@load "data/dat2-4.jld2" Vx 
V[:, idx2, idx1] = Vx 
@load "data/dat3-4.jld2" Vx 
V[:, idx2, idx2] = Vx

close("all")
visualize_wavefield(V, param)

