using Revise
using ADSeismic
using ADCME 
using JLD2
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
λ, ρ, μ = compute_lame_parameters(param, vp, vs, rho)
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

propagator = MPIElasticPropagatorSolver(param, src, ρ, λ, μ)
MPISimulatedObservation!(propagator, receiver)

sess = Session(); init(sess)
@info "starting..."
rcvv, Vx = run(sess, [receiver.rcvv, propagator.vx])

@info "saving..."
@save "data/dat$(mpi_rank())-$(mpi_size()).jld2" rcvv

if mpi_size()>1
    mpi_finalize()  
end
# if mpi_rank()==0 && mpi_size()==1
#     close("all")
#     pcolormesh(rcvv, cmap="gray")
#     colorbar()
#     xlabel("Location")
#     ylabel("Time")
#     savefig("receiver.png")

#     close("all")
#     visualize_wavefield(Vx, param)

# end