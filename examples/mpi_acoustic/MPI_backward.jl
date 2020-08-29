using ADSeismic
using ADCMEKit
using ADCME
using PyPlot
using JLD2
using PyCall
using DelimitedFiles
r = mpi_init()

n = 100
M = N = Int(round(sqrt(mpi_size())))
@info M 
param = MPIAcousticPropagatorParams(NX = n*M, NY = n*N, n = n, Rcoef=0.2, DELTAX=1000/(n*M), DELTAY=1000/(n*N), DELTAT=0.01)
compute_PML_Params!(param)

srci = [n*param.M÷5]
srcj = [n*param.N÷2]
srcv = hcat([Ricker(param, 100.0, 500.0)]...)
src = MPIAcousticSource(param, srci,srcj,srcv)


rcvj = Array(20:n*param.M-19)
rcvi = ones(Int64, length(rcvj))*((n*param.N)÷5)

xs = Float64[]
ys = Float64[]
mask = ones(n, n)
for i = 1:n 
    for j = 1:n 
        x = i*param.DELTAX + (param.II-1) * n * param.DELTAX
        y = j*param.DELTAY + (param.JJ-1) * n * param.DELTAY 
        if x<=100 || x>=900 || y<=100 || y>=900
            mask[i,j] = 0.0
        end
        push!(xs, x)
        push!(ys, y)
    end
end

θ = placeholder(fc_init([2, 20, 20, 20, 1]))
θ_ = mpi_bcast(θ)
c = fc([xs ys], [20, 20, 20, 1], θ_) * 1000.0 
c = reshape(c, (n, n))
c = c .* mask + 1000.0 * (1 .- mask)

propagator = MPIAcousticPropagatorSolver(param, src, c)
MPISimulatedObservation!(propagator, receiver)
@load "data/dat$(mpi_rank())-$(mpi_size()).jld2" rcvv

if isnothing(rcvv)
    @info mpi_rank(), "nothing"
    local_loss = 0.0 * sum(propagator.u)
else
    @info mpi_rank(), "value"
    local_loss = sum((receiver.rcvv - rcvv )^2)
end

loss = mpi_sum(local_loss)
G = gradients(loss, θ)

sess = Session(); init(sess)

ll = run(sess, local_loss)
@info mpi_rank(), ll
l = run(sess, loss)
g = run(sess, G)


if mpi_rank()==0
    println("Loss = $l")
    writedlm(stdout, g)
end

if mpi_size()==1
    close("all")
    pcolormesh( (1:param.n)*param.DELTAX, (1:param.n)*param.DELTAY, g')
    gca().invert_yaxis()
    colorbar()
    xlabel("x")
    ylabel("y")
    savefig("gradient.png")
end

mpi_finalize()