using ADSeismic
using ADCME
using PyPlot
using JLD2
using PyCall
using DelimitedFiles
using Statistics
mpi_init()

M = N = Int(round(sqrt(mpi_size())))
n_total = 1000
@info M 
n = n_total÷M
param = MPIAcousticPropagatorParams(NX = n*M, NY = n*N, n = n,
     Rcoef=0.2, DELTAX=10, DELTAY=10, DELTAT=0.05, NSTEP = 2000) 
compute_PML_Params!(param)


srci = [n*param.M÷5]
srcj = [n*param.N÷2]
srcv = hcat([Ricker(param, 100.0, 500.0)]...)
src = MPIAcousticSource(param, srci,srcj,srcv)

rcvj = Array(20:n*param.M-19)
rcvi = ones(Int64, length(rcvj))*((n*param.N)÷5)
receiver =  MPIAcousticReceiver(param, rcvi, rcvj)

xs = Float64[]
ys = Float64[]
mask = ones(n, n)
for i = 1:n 
    for j = 1:n 
        x = i*param.DELTAX + (param.II-1) * n * param.DELTAX
        y = j*param.DELTAY + (param.JJ-1) * n * param.DELTAY 
        idx = i + (param.II-1) * n
        idy = j + (param.JJ-1) * n 
        if idx<=param.NPOINTS_PML || idx>=param.N*n-param.NPOINTS_PML || 
            idy<=param.NPOINTS_PML || idy>=param.M*n-param.NPOINTS_PML
            mask[i,j] = 0.0
        end
        push!(xs, x)
        push!(ys, y)
    end
end

init_guess = fc_init([2, 20, 20, 20, 1])
θ = placeholder(init_guess)
θ_ = mpi_bcast(θ)
c = fc([xs ys], [20, 20, 20, 1], θ_) * 1000.0 
c = reshape(c, (n, n))
c = c .* mask + 1000.0 * (1 .- mask)

propagator = MPIAcousticPropagatorSolver(param, src, c)
MPISimulatedObservation!(propagator, receiver)
@load "data/dat$(mpi_rank())-$(mpi_size()).jld2" rcvv

local_loss = sum(propagator.u[end,:]) * 1e-20
# if !isnothing(rcvv)
#     @info mpi_rank(), "value"
#     global local_loss += sum((receiver.rcvv - rcvv )^2) 
# end

loss = mpi_sum(local_loss)
G = gradients(loss, θ)

sess = Session(); init(sess)
 
@info "Runing loss..."
l = run(sess, loss)

init(sess)
@info "Runing grads..."
g = run(sess, G, θ=>init_guess)

timer_fwd = []
timer_bwd = []
for i = 1:10
    @info "forward, $i"
    init(sess)
    d = @timed run(sess, loss)
    push!(timer_fwd, d[2])
end

for i = 1:10
    @info "backward, $i"
    init(sess)
    d = @timed run(sess, G)
    push!(timer_bwd, d[2])
end

timer_fwd = mean(timer_fwd)
timer_bwd = mean(timer_bwd)

if mpi_rank()==0
    open("timing.txt", "a") do io 
        writedlm(io, [mpi_size() timer_fwd timer_bwd])
    end
end 

mpi_finalize()