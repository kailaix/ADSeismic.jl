using PyPlot 
using JLD2
using DelimitedFiles
using ADSeismic

n = 200
@load "data/dat0-1.jld2" U
U0 = U 

Uk = zeros(size(U0,1), n, n)
@load "data/dat0-4.jld2" U
Uk[:,1:n÷2,1:n÷2] = U

@load "data/dat1-4.jld2" U
Uk[:,1:n÷2,n÷2+1:end] = U

@load "data/dat2-4.jld2" U
Uk[:,n÷2+1:end, 1:n÷2] = U

@load "data/dat3-4.jld2" U
Uk[:,n÷2+1:end, n÷2+1:end] = U 

@show maximum(abs.(U0 - Uk))

