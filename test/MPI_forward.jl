using ADSeismic
using ADCMEKit
using ADCME
using PyPlot
using JLD2
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

c = 1000*ones(n, n)
for i = 1:n 
    for j = 1:n 
        x = i*param.DELTAX + (param.II-1) * n * param.DELTAX
        y = j*param.DELTAY + (param.JJ-1) * n * param.DELTAY 
        if 400<=x<=600 && 400<=y<=600
            c[i, j] = 2000.
        end
    end
end

propagator = MPIAcousticPropagatorSolver(param, src, c)
MPISimulatedObservation!(propagator, receiver)

sess = Session(); init(sess)

U = run(sess, propagator.u)
rcvv = run(sess, receiver.rcvv)


@save "data/dat$(mpi_rank())-$(mpi_size()).jld2" U rcvv


if mpi_size()==1
    close("all")
    pcolormesh((rcvi.-1)*param.DELTAX, (0:param.NSTEP)*param.DELTAT,  rcvv, cmap="gray")
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