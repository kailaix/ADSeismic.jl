using ADSeismic
using ADCMEKit
using ADCME
using PyPlot
using JLD2
using PyCall
r = mpi_init()

n = 100
M = N = Int(round(sqrt(mpi_size())))
@info M 
param = MPIAcousticPropagatorParams(NX = n*M, NY = n*N, n = n, Rcoef=0.2, DELTAX=1000/(n*M), DELTAY=1000/(n*N), DELTAT=0.01)

compute_PML_Params!(param)

srci = [n*param.JJ÷2]
srcj = [n*param.II÷5]
srcv = hcat([Ricker(param, 100.0, 500.0)]...)
src = MPIAcousticSource(param, srci,srcj,srcv)

rcvi = Array(20:n*param.JJ-19)
rcvj = ones(Int64, length(rcvi))*(n*param.II÷5)
receiver =  MPIAcousticReceiver(param, rcvi, rcvj)

cv = Variable(1.0)
c = constant(1000*ones(n, n))*cv

propagator = MPIAcousticPropagatorSolver(param, src, c)
MPISimulatedObservation!(propagator, receiver)
@load "data/dat$(mpi_rank())-$(mpi_size()).jld2" rcvv

if isa(receiver.rcvv, PyObject)
    local_loss = 0.0 * sum(propagator.u)
else
    local_loss = sum((receiver.rcvv )^2)
end

loss = mpi_sum(local_loss)
G = gradients(loss, c)

sess = Session(); init(sess)

ll = run(sess, local_loss)
@info mpi_rank(), ll
l = run(sess, loss)
g = run(sess, G)


if mpi_rank()==0
    @info l
end

# if mpi_rank()==0
#     close("all")
#     pcolormesh((rcvi.-1)*param.DELTAX, (0:param.NSTEP)*param.DELTAT,  rcvv, cmap="gray")
#     colorbar()
#     xlabel("Location")
#     ylabel("Time")
#     savefig("receiver.png")

#     close("all")
#     visualize_wavefield(U, param)

#     close("all")
#     visualize_model(c, param)
#     savefig("model.png")
# end

# mpi_finalize()