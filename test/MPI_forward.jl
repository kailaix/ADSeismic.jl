using ADSeismic
using ADCMEKit
using ADCME
using PyPlot
using DelimitedFiles
using JLD2
r = mpi_init()

M = N = Int(round(sqrt(mpi_size())))
n = 30÷M
@info M 
param = MPIAcousticPropagatorParams(NX = n*M, NY = n*N, n = n,
     Rcoef=0.2, DELTAX=20, DELTAY=20, DELTAT=0.01, NSTEP=100) #, USE_PML_XMIN = false, USE_PML_XMAX = false, 
    #  USE_PML_YMIN = false, USE_PML_YMAX = false)
compute_PML_Params!(param)

srci = [n*param.M÷2]
srcj = [n*param.N÷2]
srcv = hcat([Ricker(param, 100.0, 500.0)]...)
srcv = ones(size(srcv)...)
src = MPIAcousticSource(param, srci,srcj,srcv)

rcvj = Array(1:n*param.M)
rcvi = ones(Int64, length(rcvj))*((n*param.N)÷5)

receiver =  MPIAcousticReceiver(param, rcvi, rcvj)

c = 1000*ones(n, n)
# center = n*M÷2
# width = n*M ÷ 8
# for i = 1:n 
#     for j = 1:n 
#         ii = i + (param.II-1) * n
#         jj = j + (param.JJ-1) * n
#         x = i*param.DELTAX + (param.II-1) * n * param.DELTAX
#         y = j*param.DELTAY + (param.JJ-1) * n * param.DELTAY 
#         if center -width <= ii <= center + width && center -width <=jj<=center +width
#             c[i, j] = 2000.
#         end
#     end
# end

propagator = MPIAcousticPropagatorSolver(param, src, c)
MPISimulatedObservation!(propagator, receiver)

sess = Session(); init(sess)

U = run(sess, propagator.u)
rcvv = run(sess, receiver.rcvv)
φ = run(sess, propagator.φ)
ψ = run(sess, propagator.ψ)

sigma, tau = param.Σx, param.Σy
rcvi = receiver.rcvi
rcvj = receiver.rcvj
@save "data/dat$(mpi_rank())-$(mpi_size()).jld2" U rcvv sigma tau rcvi rcvj φ ψ

# if mpi_rank()==0
#     writedlm(stdout, rcvv)
# end

if mpi_size()==1
    close("all")
    pcolormesh(rcvv, cmap="gray")
    colorbar()
    xlabel("Location")
    ylabel("Time")
    savefig("receiver.png")

    close("all")
    visualize_wavefield(U, param)

    close("all")
    visualize_model(c, param)
    savefig("model.png")
end

mpi_finalize()