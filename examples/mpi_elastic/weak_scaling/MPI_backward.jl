using Revise
using ADSeismic
using ADCME 
using JLD2
using DelimitedFiles
ADCME.options.customop.verbose = false
mpi_init()
M = N = Int(sqrt(mpi_size()))
n = 100

param = MPIElasticPropagatorParams(NX= n * M, NY= n * N, n=n, NSTEP=2000, DELTAT=1e-4/2, 
                        DELTAX=1.0, DELTAY=1.0)
compute_PML_Params!(param)
vp = 3300.
vs = 3300. / 1.732
rho = 2800.
λ, ρ, μ = compute_default_properties(param, vp, vs, rho)
source = Ricker(param, 50.0, 200.0) 

srci = [div(param.NX,5)]
srcj = [div(param.NY,2)]
srctype = [0]
srcv = reshape(source, :, 1)
src = MPIElasticSource(param, srci, srcj, srctype, srcv)


rcvj = Array(20:n*param.M-19)
rcvi = ones(Int64, length(rcvj))*((n*param.N)÷5)
rcvtype = zeros(Int64, length(rcvj))
receiver =  MPIElasticReceiver(param, rcvi, rcvj, rcvtype)

xs = Float64[]
ys = Float64[]
for i = 1:n + 4
    for j = 1:n + 4
        x = (i - 2)*param.DELTAX + (param.II-1) * n * param.DELTAX
        y = (j - 2)*param.DELTAY + (param.JJ-1) * n * param.DELTAY 
        push!(xs, x)
        push!(ys, y)
    end
end


init_guess = fc_init([2, 20, 20, 20, 1])
θ = placeholder(init_guess)
θ_ = mpi_bcast(θ)
c = fc([xs ys], [20, 20, 20, 1], θ_) * 1e10
λ = reshape(c, (n+4, n+4))


propagator = MPIElasticPropagatorSolver(param, src, ρ, λ, μ)
MPISimulatedObservation!(propagator, receiver)

@load "data/dat$(mpi_rank())-$(mpi_size()).jld2" rcvv


sess = Session(); init(sess)
@info "starting..."
rcvv, Vx = run(sess, [receiver.rcvv, propagator.vx])

local_loss = sum(propagator.vx[end,:]) * 1e-20
if !isnothing(rcvv)
    @info mpi_rank(), "value"
    global local_loss += sum((receiver.rcvv - rcvv )^2) 
end

loss = mpi_sum(local_loss)
G = gradients(loss, θ)

sess = Session(); init(sess)

@info "running loss"
run(sess, loss)
@info "running gradients"
run(sess, G)

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