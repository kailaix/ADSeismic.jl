using PyPlot 
using JLD2
using DelimitedFiles

n = 30
@load "data/dat0-1.jld2" rcvv U sigma tau rcvi rcvj φ ψ
rcvv0 = rcvv 
sigma0 = sigma
tau0 = tau
U0 = U 
Phi0 = φ
Psi0 = ψ
rcvi0 = rcvi
rcvj0 = rcvj

Uk = zeros(size(U0,1), n, n)
Phik = zeros(size(U0,1), n, n)
Psik = zeros(size(U0,1), n, n)

@load "data/dat0-4.jld2" rcvv U sigma tau rcvi rcvj φ ψ
rcvv1 = rcvv 
sigma1 = sigma[1:end-1, 1:end-1]
tau1 = tau[1:end-1, 1:end-1]
φ1 = φ
ψ1 = ψ

U1 = U 
Uk[:,1:n÷2,1:n÷2] = U1
Phik[:,1:n÷2,1:n÷2] = φ1
Psik[:,1:n÷2,1:n÷2] = ψ1

rcvi1 = rcvi
rcvj1 = rcvj

@load "data/dat1-4.jld2" rcvv U sigma tau rcvi rcvj φ ψ
rcvv2 = rcvv 
sigma2 = sigma[1:end-1, 2:end]
tau2 = tau[1:end-1, 2:end]
U2 = U 
Uk[:,1:n÷2,n÷2+1:end] = U2
rcvi2 = rcvi
rcvj2 = rcvj
φ2 = φ
ψ2 = ψ

Phik[:,1:n÷2,n÷2+1:end] = φ2
Psik[:,1:n÷2,n÷2+1:end] = ψ2


@load "data/dat2-4.jld2" rcvv U sigma tau rcvi rcvj φ ψ
rcvv3 = rcvv 
sigma3 = sigma[2:end, 1:end-1]
tau3 = tau[2:end, 1:end-1]
U3 = U 
Uk[:,n÷2+1:end, 1:n÷2] = U3
rcvi3 = rcvi
rcvj3 = rcvj
φ3 = φ
ψ3 = ψ

Phik[:,n÷2+1:end, 1:n÷2] = φ3
Psik[:,n÷2+1:end, 1:n÷2] = ψ3

@load "data/dat3-4.jld2" rcvv U sigma tau rcvi rcvj φ ψ
rcvv4 = rcvv 
sigma4 = sigma[2:end, 2:end]
tau4 = tau[2:end, 2:end]
U4 = U 
Uk[:,n÷2+1:end, n÷2+1:end] = U4
rcvi4 = rcvi
rcvj4 = rcvj
φ4 = φ
ψ4 = ψ

Phik[:,n÷2+1:end, n÷2+1:end] = φ4
Psik[:,n÷2+1:end, n÷2+1:end] = ψ4

@show maximum(abs.(U0 - Uk))

S = [sigma1 sigma2; sigma3 sigma4]
T = [tau1 tau2; tau3 tau4]

@show maximum(abs.(sigma0-S))
@show maximum(abs.(tau0-T))

rcvv_mpi = [rcvv1 rcvv2]

idx0 = @. (rcvj0 - 1) * n + rcvi0 
rcvv0_ref = reshape(U0, size(U0,1), :)[:, idx0]

idx1 = @. (rcvj1 - 1) * (n÷2) + rcvi1
rcvv1_ref = reshape(U1, size(U1,1), :)[:, idx1]


@show maximum(abs.(rcvv_mpi - rcvv0))

# writedlm(stdout, Uk[8,1:15,16:end])

# writedlm(stdout, U0[8,1:15,16:end])
# close("all")
# pcolormesh(rcvv_mpi, cmap="gray")
# colorbar()
# savefig("receiver_mpi.png")
