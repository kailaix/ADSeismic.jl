using Revise
using ADSeismic
using ADCME 
using JLD2
using Test


param = ElasticPropagatorParams(NX=10, NY=10, NSTEP=10, DELTAT=1e-4, 
    DELTAX=1.0, DELTAY=1.0)
nv = 10

idx1 = (1:5) 
idx2 = (6:10) 
V = zeros(param.NSTEP+1, param.NX, param.NY)
@load "data/dat0-4.jld2" Vx param
param00 = param
V[:, idx1, idx1] = Vx 
@load "data/dat1-4.jld2" Vx param
param01 = param
V[:, idx1, idx2] = Vx 
@load "data/dat2-4.jld2" Vx param
param10 = param
V[:, idx2, idx1] = Vx 
@load "data/dat3-4.jld2" Vx param
param11 = param
V[:, idx2, idx2] = Vx

V = V[nv,:,:]
@load "data/dat0-1.jld2" Vx param
Vx = Vx[nv,:,:]

@test V≈Vx 

V = zeros(param.NSTEP+1, param.NX, param.NY)
@load "data/dat0-4.jld2" Vy 
V[:, idx1, idx1] = Vy 
@load "data/dat1-4.jld2" Vy 
V[:, idx1, idx2] = Vy 
@load "data/dat2-4.jld2" Vy 
V[:, idx2, idx1] = Vy 
@load "data/dat3-4.jld2" Vy 
V[:, idx2, idx2] = Vy
V = V[nv,:,:]
@load "data/dat0-1.jld2" Vy 
Vy = Vy[nv,:,:]
@test V≈Vy 


V = zeros(param.NSTEP+1, param.NX, param.NY)
@load "data/dat0-4.jld2" Sxx 
V[:, idx1, idx1] = Sxx 
@load "data/dat1-4.jld2" Sxx 
V[:, idx1, idx2] = Sxx 
@load "data/dat2-4.jld2" Sxx 
V[:, idx2, idx1] = Sxx 
@load "data/dat3-4.jld2" Sxx 
V[:, idx2, idx2] = Sxx
V = V[nv,:,:]
@load "data/dat0-1.jld2" Sxx 
Sxx = Sxx[nv,:,:]
@test V≈Sxx 


V = zeros(param.NSTEP+1, param.NX, param.NY)
@load "data/dat0-4.jld2" Syy 
V[:, idx1, idx1] = Syy 
@load "data/dat1-4.jld2" Syy 
V[:, idx1, idx2] = Syy 
@load "data/dat2-4.jld2" Syy 
V[:, idx2, idx1] = Syy 
@load "data/dat3-4.jld2" Syy 
V[:, idx2, idx2] = Syy
V = V[nv,:,:]
@load "data/dat0-1.jld2" Syy 
Syy = Syy[nv,:,:]
@test V≈Syy 

V = zeros(param.NSTEP+1, param.NX, param.NY)
@load "data/dat0-4.jld2" Sxy 
V[:, idx1, idx1] = Sxy 
@load "data/dat1-4.jld2" Sxy 
V[:, idx1, idx2] = Sxy 
@load "data/dat2-4.jld2" Sxy 
V[:, idx2, idx1] = Sxy 
@load "data/dat3-4.jld2" Sxy 
V[:, idx2, idx2] = Sxy
V = V[nv,:,:]
@load "data/dat0-1.jld2" Sxy 
Sxy = Sxy[nv,:,:]
@test V≈Sxy 

# close("all")
# visualize_wavefield(V, param)

